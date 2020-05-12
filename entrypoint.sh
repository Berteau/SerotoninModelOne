#/bin/sh

# Required environment variables:
#
# PARAMS: the hashed parameters details corresponding to a flags file in ./flags/flags.${BUILD_ID}
# SERVICE_AGENT_PASSPHRASE: the passphrase to use at runtime to decrypt the service agent credentials.
#    Request access at https://docs.google.com/document/d/1s-9nAPxLcYlVT9nviQrpx6WRAAiyCeccosvBDe0O-RQ

echo "Using build id ${PARAMS}"

FLAGFILE=./flags/flags.${BUILD_ID}

python3 ./app.py --mode=PLOT --flagfile="$FLAGFILE"

gpg -d --batch --passphrase="${SERVICE_AGENT_PASSPHRASE}" service_agent.json.gpg > service_agent.json
gcloud auth activate-service-account --key-file=service_agent.json
gsutil -m cp -r figures/* gs://serotonin
