package org.opensearch.ml.engine.utils;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
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

@Log4j2
public class MLFileUtils {

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
        FileUtils.createParentDirectories(destinationFile);
        try (OutputStream output = new BufferedOutputStream(new FileOutputStream(destinationFile, append))){
            output.write(data);
        }
    }

    public static void mergeFiles(File[] files, File mergedFile) {
        boolean failed = false;
        for (int i = 0; i< files.length ; i++) {
            File f = files[i];
            try (InputStream inStream = new BufferedInputStream(new FileInputStream(f))) {
                if (!failed) {
                    int fileLength = (int) f.length();
                    byte fileContent[] = new byte[fileLength];
                    inStream.read(fileContent, 0, fileLength);

                    write(fileContent, mergedFile, true);
                }
                FileUtils.deleteQuietly(f);
                if (i == files.length - 1) {
                    FileUtils.deleteQuietly(f.getParentFile());
                }
            } catch (IOException e) {
                log.error("Failed to merge file " + f.getAbsolutePath() + " to " + mergedFile.getAbsolutePath());
                failed = true;
            }
        }
        if (failed) {
            FileUtils.deleteQuietly(mergedFile);
            throw new MLException("Failed to merge model chunks");
        }
    }

//    public static String getFileMD5(File file) throws IOException
//    {
//        //Get file input stream for reading the file content
//        FileInputStream fis = new FileInputStream(file);
//
//        //Create byte array to read data in chunks
//        byte[] byteArray = new byte[1024];
//        int bytesCount = 0;
//
//        //Read file data and update in message digest
//        while ((bytesCount = fis.read(byteArray)) != -1) {
//            MessageDigest messageDigest = MessageDigest.getInstance("MD5");
//            messageDigest.update(byteArray, 0, bytesCount);
//        };
//
//        //close the stream; We don't need it now.
//        fis.close();
//
//        //Get the hash's bytes
//        byte[] bytes = messageDigest.digest();
//
//        //This bytes[] has bytes in decimal format;
//        //Convert it to hexadecimal format
//        StringBuilder sb = new StringBuilder();
//        for(int i=0; i< bytes.length ;i++)
//        {
//            sb.append(Integer.toString((bytes[i] & 0xff) + 0x100, 16).substring(1));
//        }
//
//        //return complete hash
//        return sb.toString();
//    }
}
