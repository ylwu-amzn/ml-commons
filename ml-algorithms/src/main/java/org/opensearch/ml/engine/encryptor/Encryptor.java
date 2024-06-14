/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.encryptor;

import org.opensearch.core.action.ActionListener;

public interface Encryptor {

    /**
     * Takes plaintext and returns encrypted text.
     *
     * @param plainText plainText.
     * @return String encryptedText.
     */
    void encrypt(String plainText, ActionListener<String> listener);

    /**
     * Takes encryptedText and returns plain text.
     *
     * @param encryptedText encryptedText.
     * @return String plainText.
     */
    void decrypt(String encryptedText, ActionListener<String> listener);

    /**
     * Set up the masterKey for dynamic updating
     *
     * @param masterKey masterKey to be set.
     */
    void setMasterKey(String masterKey);

    String getMasterKey();

    String generateMasterKey();

}
