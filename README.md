# Serotonin Model

## Local run
*   Have python3 3.6 or later
*   `run_local.sh` for Linux systems
	*   Uses './.venv' for the virtual environment, ignored by git
	*   Generates images into ./figures folder, storing multiple
		runs in ./figures/local_build/{ISO_TIMESTAMP_OF_RUN}.

## Docker run locally
*   `run_docker` runs in a local docker container and uploads to GCP.
    *   Create a local `envfile` to pass two flags in:
    *   BUILD_ID= sets the ID for this run.
	*   SERVICE_AGENT_PASSPHRASE= sets the secret key to decrypt and
	    utilize a service account for uploading results to GCP.
    *   Results will be uploaded to
		https://storage.cloud.google.com/serotonin using the same
		{BUILD_ID}/{ISO_TIMESTAMP_OF_RUN} folder structure.
*   `run_docker_local_figures.bat` and `run_docker_local_figures.sh`
    both run docker locally, while mounting `./figures` to the
	container. Use the appropriate one for your system.

## Swarm run
*   Install gcloud. Either install it globally, or download and extract
    the `google-cloud-sdk` folder into the root directory.
*   Initialize gcloud auth and select the appropriate project. David has
    the details. He will also grant your google cloud accoutn appropriate
	permissions.
*   In `generate_flags.py`, update the `RANGES` dictionary to generate
    the set of flags of interest.
*   Delete any `flags/` directory, if it exists.
*   Run `python3 generate_flags.py` to create the `flags/` directory
	for this parameter space run. 
*   Run `gcloud_swarm.sh` to schedule the params for this flag set on
    a kubernetes cluster.
    *   If using a local gcloud install, uncomment the `alias gcloud=`
	    line in `gcloud_swarm.sh`
    *   Set `SERVICE_AGENT_PASS_PHRASE` in the environment, as with
	    `run_docker`.
*   Monitor https://console.cloud.google.com/kubernetes/workload?project=southerd
    and https://storage.cloud.google.com/serotonin for results and failures.