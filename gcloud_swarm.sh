#!/bin/sh

# Ensure a SERVICE_AGENT_PASSPHRASE is available when running this.
#SERVICE_AGENT_PASSPHRASE=secret

# Either install gcloud globally, or download and unzip locally
#alias gcloud=./google-cloud-sdk/bin/gcloud

# Connect kubctl to cluster
gcloud container clusters get-credentials serotonin-cluster --zone=us-central1-c

# Prepare the image
IMAGE=gcr.io/southerd/serotonin:alpha2-${RANDOM}
gcloud builds submit --tag $IMAGE

# Run the image for each flag
for F in $(ls flags) ; do 
    FLAGS=${F#flags.}
    POD=$(echo serotonin-${FLAGS//[^-a-zA-Z0-9]/-} | tr '[:upper:]' '[:lower:]')
    if [[ $(kubctl get pod $POD) ]]; then kubectl delete pod $POD; fi
    kubectl run $POD \
        --image=$IMAGE \
        --replicas=1 \
        --restart="Never" \
        --requests="memory=512Mi" \
        --env="BUILD_ID=$FLAGS" \
        --env="SERVICE_AGENT_PASSPHRASE=${SERVICE_AGENT_PASSPHRASE}"
done
