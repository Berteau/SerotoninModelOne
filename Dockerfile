# Use a google cloud sdk base image which has gsutil and python3
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim

WORKDIR /usr/lib/serotonin

# Before adding the entire working directory, install pip requirements.
# This keeps it in a separate phase, so it doesn't need to re-run pip every time any
# other code changes.
ADD requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Add the entire project, set some environment variables that will be used for flag
# default values, and run in PLOT mode
ADD . .
ENV FIGURES_DIRECTORY=/usr/lib/serotonin/figures

ENTRYPOINT "./entrypoint.sh"
