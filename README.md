# Serotonin Model

## Local run
*   Have python3 3.6 or later
*   `run_local.sh` for Linux systems
	*   Uses './.venv' for the virtual environment, ignored by git
	*   Generates images into ./figures folder, storing multiple
		runs in ./figures/local_build/{ISO_TIMESTAMP_OF_RUN}.

## Docker run locally
*   `run_docker` runs in a local docker container and uploads to GCP.
    *   Create a local `envfile` to pass these flags in:
		*   `BUILD_ID` sets the ID for this run. Defaults to `default`.
		*   `SERVICE_AGENT_FILE` sets the encrypted file containing the
			GCP service account credentials. Required.
		*   `SERVICE_AGENT_PASSPHRASE` sets the secret key to decrypt and
			utilize a service account for uploading results to GCP. Required.
    *   Results will be uploaded to
		https://storage.cloud.google.com/serotonin using the same
		`{BUILD_ID}/{ISO_TIMESTAMP_OF_RUN}` folder structure.
*   `run_docker_local_figures.bat` and `run_docker_local_figures.sh`
    both run docker locally, while mounting `./figures` to the
	container. Use the appropriate one for your system.

## Swarm run
*   Follow instructions in `gcloud_swarm.sh` to configure an environment.
*   In `generate_flags.py`, update the `RANGES` dictionary to generate
    the set of flags of interest.
*   Delete any `flags/` directory, if it exists.
*   Run `python3 generate_flags.py` to create the `flags/` directory
	for this parameter space run. 
*   Run `gcloud_swarm.sh` to schedule the params for this flag set on
    a kubernetes cluster.
	*   Ensure the correct service agent key is available, encrypted,
	    and select it with `SERVICE_AGENT_FILE`.
    *   Set `SERVICE_AGENT_PASSPHRASE` in the environment, as with
	    `run_docker`.
*   Monitor https://console.cloud.google.com/kubernetes/workload?project=serotonin-model
    and https://storage.cloud.google.com/serotoninmodel for results and failures.