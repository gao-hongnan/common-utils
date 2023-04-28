curl -o google-cloud-sdk.tar.gz \
https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-428.0.0-darwin-arm.tar.gz \
&& tar -xvf google-cloud-sdk.tar.gz \
&& ./google-cloud-sdk/install.sh --install-python TRUE --path-update TRUE --quiet


