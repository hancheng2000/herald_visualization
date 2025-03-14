#!/bin/bash

# Set to 1 if batch_mpr2csv.py should be run after syncing data
run_mpr2csv_bool=1
origin=""
dest=""

# Sync files into 
rsync -auP "$origin" "$dest"

conda activate herald

# Update to most recent version of mpr2csv.py
# git fetch
# git checkout -f -m origin/main -- ../mpr2csv.py

# Run batch processing script, send output to terminal and a log file
python batch_mpr2csv.py | tee log.txt