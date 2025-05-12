/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.httpclient;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.time.Duration;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import io.netty.channel.ChannelFactory;
import io.netty.channel.ChannelOption;
import io.netty.channel.socket.nio.NioSocketChannel;
import lombok.extern.log4j.Log4j2;
import software.amazon.awssdk.http.async.AsyncExecuteRequest;
import software.amazon.awssdk.http.async.SdkAsyncHttpClient;
import software.amazon.awssdk.http.nio.netty.NettyNioAsyncHttpClient;
import software.amazon.awssdk.http.nio.netty.SdkEventLoopGroup;

@Log4j2
public class MLHttpClientFactory {

    /**
     * Custom HTTP client implementation that ensures connections are made from the eth1 interface
     */
    private static class Eth1BoundHttpClient implements SdkAsyncHttpClient {
        private final SdkAsyncHttpClient delegate;
        private final InetAddress localAddress;

        public Eth1BoundHttpClient(Duration connectionTimeout, Duration readTimeout, int maxConnections, InetAddress localAddress) {
            this.localAddress = localAddress;
            try {
                // Create a custom channel factory that will bind outgoing connections to our eth1 address
                ChannelFactory<NioSocketChannel> channelFactory = () -> {
                    NioSocketChannel channel = new NioSocketChannel() {
                        @Override
                        protected boolean doConnect(java.net.SocketAddress remoteAddress, java.net.SocketAddress localAddress) throws Exception {
                            // Force outgoing connections to use our eth1 address
                            if (localAddress == null) {
                                localAddress = new java.net.InetSocketAddress(Eth1BoundHttpClient.this.localAddress, 0);
                            }
                            return super.doConnect(remoteAddress, localAddress);
                        }
                    };
                    return channel;
                };

                // Create a custom EventLoopGroup builder with our channel factory
                SdkEventLoopGroup.Builder eventLoopGroupBuilder = SdkEventLoopGroup.builder()
                        .channelFactory(channelFactory);

                // Build the delegate HTTP client with our custom event loop group
                this.delegate = AccessController.doPrivileged(
                        (PrivilegedExceptionAction<SdkAsyncHttpClient>) () -> NettyNioAsyncHttpClient.builder()
                                .connectionTimeout(connectionTimeout)
                                .readTimeout(readTimeout)
                                .maxConcurrency(maxConnections)
                                .eventLoopGroupBuilder(eventLoopGroupBuilder)
                                .putChannelOption(ChannelOption.SO_REUSEADDR, true)
                                .build()
                );
            } catch (Exception e) {
                log.error("Failed to create eth1-bound HTTP client", e);
                throw new RuntimeException("Failed to initialize eth1-bound HTTP client", e);
            }
        }

        @Override
        public CompletableFuture<Void> execute(AsyncExecuteRequest request) {
            return delegate.execute(request);
        }

        @Override
        public void close() {
            delegate.close();
        }

        @Override
        public String clientName() {
            return "Eth1Bound-" + delegate.clientName();
        }
    }

    private static SdkAsyncHttpClient createDefaultHttpClient(Duration connectionTimeout, Duration readTimeout, int maxConnections) {
        try {
            return AccessController.doPrivileged(
                    (PrivilegedExceptionAction<SdkAsyncHttpClient>) () -> NettyNioAsyncHttpClient.builder()
                            .connectionTimeout(connectionTimeout)
                            .readTimeout(readTimeout)
                            .maxConcurrency(maxConnections)
                            .build()
            );
        } catch (PrivilegedActionException e) {
            log.error("Failed to create HTTP client", e);
            return null;
        }
    }

    public static SdkAsyncHttpClient getAsyncHttpClient(Duration connectionTimeout, Duration readTimeout, int maxConnections) {
        try {
            // Try to get the IPv4 address of eth1
            InetAddress localAddress = getEth1Address();
            if (localAddress != null) {
                log.info("Creating HTTP client bound to eth1 interface with address: {}", localAddress.getHostAddress());
                return new Eth1BoundHttpClient(connectionTimeout, readTimeout, maxConnections, localAddress);
            } else {
                log.warn("Could not find eth1 interface or IPv4 address. Using default interface for HTTP client.");
                return createDefaultHttpClient(connectionTimeout, readTimeout, maxConnections);
            }
        } catch (Exception e) {
            log.error("Failed to create HTTP client with eth1 binding, falling back to default", e);
            return createDefaultHttpClient(connectionTimeout, readTimeout, maxConnections);
        }
    }

    private static InetAddress getEth1Address() {
        try {
            NetworkInterface eth1 = NetworkInterface.getByName("eth1");
            if (eth1 != null) {
                Enumeration<InetAddress> addresses = eth1.getInetAddresses();
                while (addresses.hasMoreElements()) {
                    InetAddress address = addresses.nextElement();
                    if (address instanceof Inet4Address) {
                        return address;
                    }
                }
            }
        } catch (Exception e) {
            log.error("Error finding eth1 interface", e);
        }
        return null;
    }

    /**
     * Validate the input parameters, such as protocol, host and port.
     * @param protocol The protocol supported in remote inference, currently only http and https are supported.
     * @param host The host name of the remote inference server, host must be a valid ip address or domain name and must not be localhost.
     * @param port The port number of the remote inference server, port number must be in range [0, 65536].
     * @param connectorPrivateIpEnabled The port number of the remote inference server, port number must be in range [0, 65536].
     * @throws UnknownHostException Allow to use private IP or not.
     */
    public static void validate(String protocol, String host, int port, AtomicBoolean connectorPrivateIpEnabled)
        throws UnknownHostException {
        if (protocol != null && !"http".equalsIgnoreCase(protocol) && !"https".equalsIgnoreCase(protocol)) {
            log.error("Remote inference protocol is not http or https: " + protocol);
            throw new IllegalArgumentException("Protocol is not http or https: " + protocol);
        }
        // When port is not specified, the default port is -1, and we need to set it to 80 or 443 based on protocol.
        if (port == -1) {
            if (protocol == null || "http".equals(protocol.toLowerCase(Locale.getDefault()))) {
                port = 80;
            } else {
                port = 443;
            }
        }
        if (port < 0 || port > 65536) {
            log.error("Remote inference port out of range: " + port);
            throw new IllegalArgumentException("Port out of range: " + port);
        }
        validateIp(host, connectorPrivateIpEnabled);
    }

    private static void validateIp(String hostName, AtomicBoolean connectorPrivateIpEnabled) throws UnknownHostException {
        InetAddress[] addresses = InetAddress.getAllByName(hostName);
        if ((connectorPrivateIpEnabled == null || !connectorPrivateIpEnabled.get()) && hasPrivateIpAddress(addresses)) {
            log.error("Remote inference host name has private ip address: " + hostName);
            throw new IllegalArgumentException("Remote inference host name has private ip address: " + hostName);
        }
    }

    private static boolean hasPrivateIpAddress(InetAddress[] ipAddress) {
        for (InetAddress ip : ipAddress) {
            if (ip instanceof Inet4Address) {
                byte[] bytes = ip.getAddress();
                if (bytes.length != 4) {
                    return true;
                } else {
                    int firstOctets = bytes[0] & 0xff;
                    int firstInOctal = parseWithOctal(String.valueOf(firstOctets));
                    int firstInHex = Integer.parseInt(String.valueOf(firstOctets), 16);
                    if (firstInOctal == 127 || firstInHex == 127) {
                        return bytes[1] == 0 && bytes[2] == 0 && bytes[3] == 1;
                    } else if (firstInOctal == 10 || firstInHex == 10) {
                        return true;
                    } else if (firstInOctal == 172 || firstInHex == 172) {
                        int secondOctets = bytes[1] & 0xff;
                        int secondInOctal = parseWithOctal(String.valueOf(secondOctets));
                        int secondInHex = Integer.parseInt(String.valueOf(secondOctets), 16);
                        return (secondInOctal >= 16 && secondInOctal <= 32) || (secondInHex >= 16 && secondInHex <= 32);
                    } else if (firstInOctal == 192 || firstInHex == 192) {
                        int secondOctets = bytes[1] & 0xff;
                        int secondInOctal = parseWithOctal(String.valueOf(secondOctets));
                        int secondInHex = Integer.parseInt(String.valueOf(secondOctets), 16);
                        return secondInOctal == 168 || secondInHex == 168;
                    }
                }
            }
        }
        return Arrays.stream(ipAddress).anyMatch(x -> x.isSiteLocalAddress() || x.isLoopbackAddress() || x.isAnyLocalAddress());
    }

    private static int parseWithOctal(String input) {
        try {
            return Integer.parseInt(input, 8);
        } catch (NumberFormatException e) {
            return Integer.parseInt(input);
        }
    }
}
