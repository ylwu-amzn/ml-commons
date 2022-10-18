/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.utils;

import lombok.extern.log4j.Log4j2;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.ArrayList;

@Log4j2
public class FileUtils {

    public static ArrayList<String> readAndFragment(File file, Path outputPath, int chunkSize) throws IOException {
        int fileSize = (int) file.length();
        ArrayList<String> nameList = new ArrayList<>();
        try (InputStream inStream = new BufferedInputStream(new FileInputStream(file))){
            int numberOfChunk = 0;
            int totalBytesRead = 0;
            while (totalBytesRead < fileSize) {
                String partName = numberOfChunk + "";
                int bytesRemaining = fileSize - totalBytesRead;
                if (bytesRemaining < chunkSize) {
                    chunkSize = bytesRemaining;
                }
                byte[] temporary = new byte[chunkSize];
                int bytesRead = inStream.read(temporary, 0, chunkSize);
                if (bytesRead > 0) {
                    totalBytesRead += bytesRead;
                    numberOfChunk++;
                }
                Path partFileName = outputPath.resolve(partName + "");
                write(temporary, partFileName.toString());
                nameList.add(partFileName.toString());
            }
        }
        return nameList;
    }

    public static void write(byte[] data, String destinationFileName) throws IOException {
        File file = new File(destinationFileName);
        write(data, file, false);
    }

    public static void write(byte[] data, File destinationFile, boolean append) throws IOException {
        org.apache.commons.io.FileUtils.createParentDirectories(destinationFile);
        try (OutputStream output = new BufferedOutputStream(new FileOutputStream(destinationFile, append))){
            output.write(data);
        }
    }

    /**
     * Delete file quietly.
     * @param path file path
     */
    public static void deleteFileQuietly(Path path) {
        File file = new File(path.toUri());
        if (file.exists()) {
            org.apache.commons.io.FileUtils.deleteQuietly(file);
        }
    }

}
