/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.encryptor;

import static org.opensearch.ml.common.CommonValue.CREATE_TIME_FIELD;
import static org.opensearch.ml.common.CommonValue.MASTER_KEY;
import static org.opensearch.ml.common.CommonValue.ML_CONFIG_INDEX;

import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

import javax.crypto.spec.SecretKeySpec;

import com.google.common.collect.ImmutableMap;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.action.DocWriteRequest;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.engine.VersionConflictEngineException;

import com.amazonaws.encryptionsdk.AwsCrypto;
import com.amazonaws.encryptionsdk.CommitmentPolicy;
import com.amazonaws.encryptionsdk.CryptoResult;
import com.amazonaws.encryptionsdk.jce.JceMasterKey;

import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.engine.indices.MLIndicesHandler;

@Log4j2
public class EncryptorImpl implements Encryptor {

    public static final String MASTER_KEY_NOT_READY_ERROR =
        "The ML encryption master key has not been initialized yet. Please retry after waiting for 10 seconds.";
    private ClusterService clusterService;
    private Client client;
    private volatile String masterKey;
    private MLIndicesHandler mlIndicesHandler;

    public EncryptorImpl(ClusterService clusterService, Client client, MLIndicesHandler mlIndicesHandler) {
        this.masterKey = null;
        this.clusterService = clusterService;
        this.client = client;
        this.mlIndicesHandler = mlIndicesHandler;
    }

    public EncryptorImpl(String masterKey) {
        this.masterKey = masterKey;
    }

    @Override
    public void setMasterKey(String masterKey) {
        this.masterKey = masterKey;
    }

    @Override
    public String getMasterKey() {
        return masterKey;
    }

    @Override
    public String encrypt(String plainText) {
        final AwsCrypto crypto = AwsCrypto.builder().withCommitmentPolicy(CommitmentPolicy.RequireEncryptRequireDecrypt).build();
        byte[] bytes = Base64.getDecoder().decode(masterKey);
        // https://github.com/aws/aws-encryption-sdk-java/issues/1879
        JceMasterKey jceMasterKey = JceMasterKey.getInstance(new SecretKeySpec(bytes, "AES"), "Custom", "", "AES/GCM/NOPADDING");

        final CryptoResult<byte[], JceMasterKey> encryptResult = crypto
                .encryptData(jceMasterKey, plainText.getBytes(StandardCharsets.UTF_8));
        return Base64.getEncoder().encodeToString(encryptResult.getResult());
    }

    @Override
    public String decrypt(String encryptedText) {
        final AwsCrypto crypto = AwsCrypto.builder().withCommitmentPolicy(CommitmentPolicy.RequireEncryptRequireDecrypt).build();

        byte[] bytes = Base64.getDecoder().decode(masterKey);
        JceMasterKey jceMasterKey = JceMasterKey.getInstance(new SecretKeySpec(bytes, "AES"), "Custom", "", "AES/GCM/NOPADDING");

        final CryptoResult<byte[], JceMasterKey> decryptedResult = crypto
                .decryptData(jceMasterKey, Base64.getDecoder().decode(encryptedText));
        return new String(decryptedResult.getResult());
    }

    @Override
    public void encrypt(Map<String, String> rawCredentials, ActionListener<Map<String, String>> listener) {
        initMasterKey(ActionListener.wrap(inited -> {
            Map<String, String> encryptedCredentials = new HashMap<>();
            for (String key : rawCredentials.keySet()) {
                encryptedCredentials.put(key, encrypt(rawCredentials.get(key)));
            }
            listener.onResponse(encryptedCredentials);
        }, e -> {
            log.error("Failed to init ML encryption master key", e);
            listener.onFailure(e);
        }));
    }

    @Override
    public void decrypt(Map<String, String> encryptedCredentials, ActionListener<Map<String, String>> listener) {
        initMasterKey(ActionListener.wrap(inited -> {
            Map<String, String> decryptedCredentials = new HashMap<>();
            for (String key : encryptedCredentials.keySet()) {
                decryptedCredentials.put(key, decrypt(decryptedCredentials.get(key)));
            }
            listener.onResponse(decryptedCredentials);
        }, e -> {
            log.error("Failed to init ML encryption master key", e);
            listener.onFailure(e);
        }));
    }

    @Override
    public String generateMasterKey() {
        byte[] keyBytes = new byte[32];
        new SecureRandom().nextBytes(keyBytes);
        String base64Key = Base64.getEncoder().encodeToString(keyBytes);
        return base64Key;
    }

    private void initMasterKey(ActionListener<Boolean> listener) {
        if (masterKey != null) {
            listener.onResponse(null);
            return;
        }

        mlIndicesHandler.initMLConfigIndex(ActionListener.wrap(r -> {
            GetRequest getRequest = new GetRequest(ML_CONFIG_INDEX).id(MASTER_KEY);
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                client.get(getRequest, ActionListener.wrap(getResponse -> {
                    if (!getResponse.isExists()) {
                        IndexRequest indexRequest = new IndexRequest(ML_CONFIG_INDEX).id(MASTER_KEY);
                        final String generatedMasterKey = generateMasterKey();
                        indexRequest.source(ImmutableMap.of(MASTER_KEY, generatedMasterKey, CREATE_TIME_FIELD, Instant.now().toEpochMilli()));
                        indexRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                        indexRequest.opType(DocWriteRequest.OpType.CREATE);
                        client.index(indexRequest, ActionListener.wrap(indexResponse -> {
                            log.info("ML encryption master key indexed successfully");
                            this.masterKey = generatedMasterKey;
                            log.info("ML encryption master key initialized successfully");
                            listener.onResponse(Boolean.TRUE);
                        }, e -> {
                            if (e instanceof VersionConflictEngineException) {
                                GetRequest getMasterKeyRequest = new GetRequest(ML_CONFIG_INDEX).id(MASTER_KEY);
                                try (ThreadContext.StoredContext threadContext = client.threadPool().getThreadContext().stashContext()) {
                                    client.get(getMasterKeyRequest, ActionListener.wrap(getMasterKey -> {
                                        if (getMasterKey.isExists()) {
                                            final String masterKey = (String) getResponse.getSourceAsMap().get(MASTER_KEY);
                                            this.masterKey = masterKey;
                                            log.info("ML encryption master key already initialized, no action needed");
                                            listener.onResponse(Boolean.TRUE);
                                        }
                                    }, error -> {
                                        log.debug("Failed to get ML encryption master key", e);
                                        listener.onFailure(new ResourceNotFoundException(MASTER_KEY_NOT_READY_ERROR));
                                    }));
                                }
                            } else {
                                log.debug("Failed to save ML encryption master key", e);
                                listener.onFailure(new ResourceNotFoundException(MASTER_KEY_NOT_READY_ERROR));
                            }
                        }));
                    } else {
                        final String masterKey = (String) getResponse.getSourceAsMap().get(MASTER_KEY);
                        this.masterKey = masterKey;
                        log.info("ML encryption master key already initialized, no action needed");
                        listener.onResponse(Boolean.TRUE);
                    }
                }, e -> {
                    log.debug("Failed to get ML encryption master key from config index", e);
                    listener.onFailure(new ResourceNotFoundException(MASTER_KEY_NOT_READY_ERROR));
                }));
            }
        }, e -> {
            log.debug("Failed to init ML config index", e);
            listener.onFailure(new ResourceNotFoundException(MASTER_KEY_NOT_READY_ERROR));}));
    }

    private void initMasterKey1(ActionListener<Boolean> listener) {
        if (masterKey != null) {
            listener.onResponse(null);
            return;
        }

        mlIndicesHandler.initMLConfigIndex(ActionListener.wrap(r -> {
            IndexRequest indexRequest = new IndexRequest(ML_CONFIG_INDEX).id(MASTER_KEY);
            final String generatedMasterKey = generateMasterKey();
            indexRequest.source(ImmutableMap.of(MASTER_KEY, generatedMasterKey, CREATE_TIME_FIELD, Instant.now().toEpochMilli()));
            indexRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            indexRequest.opType(DocWriteRequest.OpType.CREATE);
            client.index(indexRequest, ActionListener.wrap(indexResponse -> {
                log.info("ML encryption master key indexed successfully");
                this.masterKey = generatedMasterKey;
                log.info("ML encryption master key initialized successfully");
                listener.onResponse(Boolean.TRUE);
            }, e -> {
                if (e instanceof VersionConflictEngineException) { //TODO
                    GetRequest getMasterKeyRequest = new GetRequest(ML_CONFIG_INDEX).id(MASTER_KEY);
                    try (ThreadContext.StoredContext threadContext = client.threadPool().getThreadContext().stashContext()) {
                        client.get(getMasterKeyRequest, ActionListener.wrap(getMasterKeyResponse -> {
                            if (getMasterKeyResponse.isExists()) {
                                final String masterKey = (String) getMasterKeyResponse.getSourceAsMap().get(MASTER_KEY);
                                this.masterKey = masterKey;
                                log.info("ML encryption master key already initialized, no action needed");
                                listener.onResponse(Boolean.TRUE);
                            } else {
                                log.warn("Failed to get ML encryption master key", e);
                                listener.onFailure(new ResourceNotFoundException(MASTER_KEY_NOT_READY_ERROR));
                            }
                        }, error -> {
                            log.warn("Failed to get ML encryption master key", e);
                            listener.onFailure(new ResourceNotFoundException(MASTER_KEY_NOT_READY_ERROR));
                        }));
                    }
                } else {
                    log.warn("Failed to save ML encryption master key", e);
                    listener.onFailure(new ResourceNotFoundException(MASTER_KEY_NOT_READY_ERROR));
                }
            }));
        }, e -> {
            log.warn("Failed to init ML config index", e);
            listener.onFailure(new ResourceNotFoundException(MASTER_KEY_NOT_READY_ERROR));
        }));
    }
}
