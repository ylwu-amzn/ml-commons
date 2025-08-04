# Upload file
```
scp -i ~/Downloads/ostest_362973614626.pem $F ubuntu@44.244.78.88:/home/ubuntu/code/ylwu/ml-commons/$F; git add $F
```

# Build
```
cd ~/code/ylwu/ml-commons; ./gradlew clean; ./gradlew assemble
```

# Redeploy

> cd ~/os/3.2/opensearch-3.2.0/; rm -rf plugins/opensearch-ml; bin/opensearch-plugin install file:///home/ubuntu/code/ylwu/ml-commons/plugin/build/distributions/opensearch-ml-3.2.0.0-SNAPSHOT.zip -b; ./opensearch-tar-install.sh

