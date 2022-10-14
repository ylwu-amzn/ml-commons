/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.utils;

import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.common.io.ByteSource;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.exception.MLException;

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
import java.util.HashSet;
import java.util.Set;

@Log4j2
public class FileUtils {

    /**
     * Calculate file sha256 hash value.
     * @param file
     * @return hash code
     * @throws IOException
     */
    public static String calculateFileHash(File file) throws IOException {
        ByteSource byteSource = com.google.common.io.Files.asByteSource(file);
        HashCode hc = byteSource.hash(Hashing.sha256());
        return hc.toString();
    }

    /**
     * Split file into smaller chunks.
     * @param file file
     * @param outputPath output path
     * @param chunkSize chunks size in bytes
     * @return array of chunk files
     * @throws IOException
     */
    public static ArrayList<String> splitFile(File file, Path outputPath, int chunkSize) throws IOException {
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

    /**
     * Write bytes into file
     * @param data bytes data
     * @param destinationFileName destination file name
     * @throws IOException
     */
    public static void write(byte[] data, String destinationFileName) throws IOException {
        File file = new File(destinationFileName);
        write(data, file, false);
    }

    /**
     * Write bytes into file by choosing with append option.
     * @param data bytes data
     * @param destinationFile destination file
     * @param append append to destination file or not
     * @throws IOException
     */
    public static void write(byte[] data, File destinationFile, boolean append) throws IOException {
        org.apache.commons.io.FileUtils.createParentDirectories(destinationFile);
        try (OutputStream output = new BufferedOutputStream(new FileOutputStream(destinationFile, append))){
            output.write(data);
        }
    }

    /**
     * Merge multiple files to one file.
     * @param sourceFiles source files
     * @param destinationFile destination file
     */
    public static void mergeFiles(File[] sourceFiles, File destinationFile) {
        boolean failed = false;
        for (int i = 0; i< sourceFiles.length ; i++) {
            File f = sourceFiles[i];
            try (InputStream inStream = new BufferedInputStream(new FileInputStream(f))) {
                if (!failed) {
                    int fileLength = (int) f.length();
                    byte fileContent[] = new byte[fileLength];
                    inStream.read(fileContent, 0, fileLength);

                    write(fileContent, destinationFile, true);
                }
                org.apache.commons.io.FileUtils.deleteQuietly(f);
                if (i == sourceFiles.length - 1) {
                    org.apache.commons.io.FileUtils.deleteQuietly(f.getParentFile());
                }
            } catch (IOException e) {
                log.error("Failed to merge file " + f.getAbsolutePath() + " to " + destinationFile.getAbsolutePath());
                failed = true;
            }
        }
        if (failed) {
            org.apache.commons.io.FileUtils.deleteQuietly(destinationFile);
            throw new MLException("Failed to merge model chunks");
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

    /**
     * List files under all paths with non-recursive way.
     * @param paths file paths
     * @return all files under input paths
     */
    public static Set<String> getFileNames(Path... paths) {
        Set<String> allFileNames = new HashSet<>();
        for (Path path : paths) {
            File f = new File(path.toUri());
            if (f.exists()) {
                String[] fileNames = f.list();
                if (fileNames != null && fileNames.length > 0) {
                    for (String name : fileNames) {
                        allFileNames.add(name);
                    }
                }
            }
        }
        return allFileNames;
    }
}
