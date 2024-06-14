/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.encryptor;

import org.opensearch.core.action.ActionListener;

import java.util.Map;

public interface Encryptor {

    /**
     * Takes plaintext and returns encrypted text.
     *
     * @param plainText plainText.
     * @return String encryptedText.
     */
    String encrypt(String plainText);

    /**
     * Takes encryptedText and returns plain text.
     *
     * @param encryptedText encryptedText.
     * @return String plainText.
     */
    String decrypt(String encryptedText);

    /**
     * Encrypt map of raw credentials.
     * @param rawCredentials raw credentials, key is credential name, value is credential in plain text.
     * @param listener return map of encrypted credentials.
     */
    void encrypt(Map<String, String> rawCredentials, ActionListener<Map<String, String>> listener);

    /**
     * Encrypt map of decrypted credentials.
     * @param encryptedCredential decrypted credentials, key is credential name, value is encrypted credential.
     * @param listener return map of decrypted credentials.
     */
    void decrypt(Map<String, String> encryptedCredential, ActionListener<Map<String, String>> listener);

    /**
     * Set up the masterKey for dynamic updating
     *
     * @param masterKey masterKey to be set.
     */
    void setMasterKey(String masterKey);

    String getMasterKey();

    String generateMasterKey();

}
