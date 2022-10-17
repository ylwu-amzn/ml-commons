/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.breaker;

import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.exception.MLException;

import java.io.File;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;

/**
 * A circuit breaker for disk usage.
 */
@Log4j2
public class DiskCircuitBreaker extends ThresholdCircuitBreaker<Long> {

    public static final long DEFAULT_DISK_SHORTAGE_THRESHOLD = 10L;
    public static final String DEFAULT_DISK_DIR = "/";
    private String diskDir;

    public DiskCircuitBreaker(Path path) {
        super(DEFAULT_DISK_SHORTAGE_THRESHOLD);
        if (path != null) {
            this.diskDir = path.toString();
        } else {
            this.diskDir = DEFAULT_DISK_DIR;
        }
    }

    @Override
    public boolean isOpen() {
        try {
            return AccessController.doPrivileged((PrivilegedExceptionAction<Boolean>) () -> {
                double availableDiskSpace = new File(diskDir).getFreeSpace() / 1024 / 1024 / 1024.0;
                return availableDiskSpace < getThreshold();  // in GB
            });
        } catch (PrivilegedActionException e) {
            throw new MLException("failed to run disk circuit breaker", e);
        }
    }
}
