#!/bin/bash
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
	python3 -m venv .venv
fi

source ./.venv/bin/activate

python3 -m pip install -r ./requirements.txt

python3 app.py --mode=PLOT $@
