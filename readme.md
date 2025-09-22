This package is designed for plotting the data in herald project.
The example notebook represents a typical workflow that aims to be universal for all relevant electrochemical testings in the project:
1. locate into the folder with all 4 tests
2. convert mpr to csv files
3. plot the GITT/cycling profile of different cycles

Note that the plotting functions return fig and ax in case you want further modifications to the plot (e.g. add theoretical OCV for comparison), and you need to save the figure yourself (save_png is provided as an option in function input)

# Batch exporting csv files from mpr files
Navigate to the inner `herald_visualization` directory and run `batch_mpr2csv.py`. The first time you run it, it will prompt you to input the path to your data directory. Using the local path convention for your system, type the full path to the directory where relevant test data is stored. Examples include:

Mac: `/Users/eschl/MIT Dropbox/Christopher Eschler/MIT/_Grad/Thesis/Data/PROPEL-1K/Electrochemical_Testing`

Windows: `C:\Users\eschl\MIT Dropbox\Christopher Eschler\MIT\_Grad\Thesis\Data\PROPEL-1K\Electrochemical_Testing`

By default, this script will only run mpr2csv in directories with data files that are newer than the latest csv export within the directory. If you wish to run mpr2csv in every available data directory instead, add the `-a` switch, i.e. `python batch_mpr2csv.py -a` (warning: this may take a long time to run). If you are running from a bash terminal, you can simultaneously save the outputs to a log file using the command `python -u batch_mpr2csv.py | tee batch_mpr2csv.log`.

# Syncing data from lab computers
## Installation
### Install cygwin and rclone
Cygwin is needed in order to run bash commands on Windows systems. Install it from here: https://www.cygwin.com/. Leave the installation directory as the default. During the installation process, on the step labeled Select Packages, set View to Full, search for `rsync` in the available modules, and select the latest available version in the New column.

The `rclone` utility is available from https://rclone.org. Download the .zip file, extract it, and copy `rclone.exe` into `C:/cygwin64/bin`. Once this is done, open a Cygwin terminal and run `rclone config` to setup the remote directory. Name the remote directory `Dropbox`. You will need to login to the Dropbox on the lab computer in order to complete this.

### Write the sync scripts
Two scripts are needed in order to automate syncing to Dropbox on a Windows system: a batch script and a bash script. To make the bash script, open Notepad++ (and/or download and install it if necessary). Make a new file and under Edit > EOL Conversion, select "Unix (LF)".
The contents of the file are as follows:

    #!/bin/bash
    origin="{origin location}"
    cache="{cache location}"
    dest="{destination location}"
    exec > >(tee "$origin/sync_data.log") 2>&1 # Creates a log file from the bash script output and outputs to stdout
    echo "Starting cache: $(date)" # Starts the log file with the date and time of execution
    rsync -avuP --exclude "sync_data.*" "$origin" "$cache" # Copies files from origin to cache, ignoring the sync_data scripts and log
    cd "$cache"

    echo "Starting sync: $(date)"
    rclone copy . "$dest" -v --dropbox-batch-mode sync --transfers 8 # Uses rclone to copy the cache folder into the Dropbox directory

A few variables will need to be set to tell the scripts where data is located and where Dropbox is. Using UNIX-style paths starting at the `/cygdrive` mount point*, set the value of `origin` to the path in which the test data is being stored locally (do not include the curly braces shown above, but do use quotes around the path). Set the value of `cache` to a new folder that you create (recommendation below). Set the value of `dest` to the path where the test data will be backed up, e.g. the Dropbox, as specified by the rclone setup. This should look something like:
    
    origin="/cygdrive/c/Users/USERNAME/Documents/EC-Lab/Data/HERALD/PROPEL-1K/"
    cache="/cygdrive/c/Users/USERNAME/Documents/HERALD_cache"
    dest="Dropbox:Data/PROPEL-1K/Electrochemical_Testing"

Naturally, the actual paths will depend on the particular system. Save this file as `sync_data.sh` in the origin directory.

The batch script should contain the following:

    @echo off
    C:\cygwin64\bin\bash.exe -l -c "{UNIX-style path to bash script}"

The .exe path on the second line points to the location of bash on the system and should not need to be changed if cygwin is installed in the default location. The path to the bash script will depend on where it is saved, e.g. `"/cygdrive/c/Users/USERNAME/Documents/EC-Lab/Data/HERALD/PROPEL-1K/sync_data.sh"`. Save the batch script in the same directory as the bash script as `sync_data.bat`. Make sure the file extension is `.bat`, not `.txt`.

*Essentially, all paths used in cygwin will begin with `/cygdrive/c/...`. From there, the path is the same as the Windows path, but using forward slashes (/) instead of backslashes (\\).

### Running the script
The script can be run manually by opening a cygwin terminal, navigating to the directory with the sync_data.sh script, and running `bash sync_data.sh`. To automate running the script, open Windows Task Scheduler and select `Create Task...` under the Actions menu. Choose a name in General. Under Triggers, select New... and set up the desired trigger conditions. For instance, select Daily and Repeat task every: 6 hours (you can type custom times in the box) to sync every 6 hours. Under Actions, select New... and make a Start a program action. For Program/script, input the path to the batch script you created. This can be saved as a new task and it will run automatically.

### Why do this instead of telling the software to save directly into the Dropbox?
Dropbox does not handle frequent changes to large files well. It generates a cache under `{Dropbox main folder}/.dropbox.cache` that tracks changes to files. If large files are being written to every few seconds, this cache grows enormously and will eventually fill all remaining disk space causing a cascade of issues, including Dropbox no longer being able to sync. It is much more manageable to make syncing to Dropbox a less frequent operation and rsync provides a powerful, elegant tool to do so.