#!/bin/sh

# Check for global gcloud, or download and install locally.
# https://cloud.google.com/sdk/docs/downloads-versioned-archives
CLOUD_SDK=https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-292.0.0-linux-x86.tar.gz
if [[ ! -x gcloud ]] ; then
    if [[ ! -d ./google-cloud-sdk ]] ; then
        echo "No gcloud available, fetching locally"
        curl --silent "$CLOUD_SDK" | tar x
    fi
    echo "Using local gcloud"
    alias gcloud=./google-cloud-sdk/bin/gcloud
fi

# Authenticate to gcloud with an account that has access to the project chosen below.
ACCOUNT="$(./google-cloud-sdk/bin/gcloud auth list --filter="status:ACTIVE" --format="value(account)")"
if [[ -z "$ACCOUNT" ]] ; then
    echo "No gcloud account, launching auth flow"
    gcloud auth init
else
    echo "Using gcloud account $ACCOUNT"
fi

# Install kubectl globally, or download a local copy (WARNING: Linux specific. Probably.).
if [[ ! -x kubctl ]] ; then
    echo "kubectl not found, fetching locally"
    KUBE_STABLE="$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)"
    curl --silent -LO https://storage.googleapis.com/kubernetes-release/release/${KUBE_STABLE}/bin/linux/amd64/kubectl
fi

# Select a project, cluster, zone, and bucket
PROJECT="${PROJECT:-serotoninmodel}"
CLUSTER="${CLUSTER:-serotonin-cluster}"
ZONE="${ZONE:-us-central1-c}"
BUCKET="${BUCKET:-serotoninmodel}"

# Ensure a SERVICE_AGENT_PASSPHRASE is available when running this.
# If you're running in the project serotonin-model, look in the
# serotoninmodel_config bucket for the keyfile and passphrase.
SERVICE_AGENT_KEY="${SERVICE_AGENT_FILE:-serotoninmodel_agent.json.gpg}"
#SERVICE_AGENT_PASSPHRASE="${SERVICE_AGENT_PASSPHRASE:-this is the wrong secret}"
if [ -z "$SERVICE_AGENT_PASSPHRASE" ]; then
    echo "Missing SERVICE_AGENT_PASSPHRASE"
    exit 1
fi 

# Memory per image. Empirically determined from watching prior runs.
# Scales linearly on population count and number of steps (time / tau).
# Format is from Kubernete's pod requests syntax. Currently set at a
# reasonable bound for totalTime=1000 tau=0.1 popCount=40
MEM="512Mi"

# Connect kubctl to cluster
gcloud container clusters get-credentials "${CLUSTER}" --project="${PROJECT}" --zone="$ZONE"

TAG=$RANDOM

# Prepare the image. Attach a random tag at the end to ensure a new image each swarm.
IMAGE="gcr.io/${PROJECT}/serotonin:alpha2-${TAG}"
gcloud builds submit --tag "$IMAGE" --project="${PROJECT}"

# Run the image for each flag
for F in $(ls flags) ; do 
    FLAGS="${F#flags.}"
    NAME="$(echo $FLAGS | md5)" # Have to hash flags so that the pod name isn't > 63 chars
    POD="serotonin-$TAG-$NAME"
    kubectl run "$POD" \
        --image="$IMAGE" \
        --replicas=1 \
        --restart="Never" \
        --requests="memory=$MEM" \
        --labels="BUILD_ID=$FLAGS" \
        --env="BUILD_ID=$FLAGS" \
        --env="BUCKET=${BUCKET}" \
        --env="SERVICE_AGENT_FILE=${SERVICE_AGENT_FILE}" \
        --env="SERVICE_AGENT_PASSPHRASE=${SERVICE_AGENT_PASSPHRASE}"
done
