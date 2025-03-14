# Installation
## Install cygwin and 
Cygwin is needed in order to run bash commands on Windows systems. Install it from here: https://www.cygwin.com/. Leave the installation directory as the default. During the installation process, on the step labeled Select Packages, set `View` to `Full`,search for `rsync` and `git` in the available modules, and select the latest available version for each in the `New` column.

## Install python and clone git repo
Now the appropriate python scripts will need to be downloaded from the git repository into an appropriate location and installed. Note that the file structure used within cygwin is POSIX-style and starts from the mount point /cygdrive. So the Windows path C:\Users\X becomes /cygdrive/c/Users/X. 

Within a cygwin terminal, run the following commands (replacing USERNAME with the appropriate path):

    mkdir /cygdrive/c/Users/USERNAME/git
    cd /cygdrive/c/Users/USERNAME/git
    git clone https://github.com/hancheng2000/herald_visualization.git
    cd herald_visualization
    pip install .

## Setting variables
A few variables will need to be set to tell the scripts where data is located and where Dropbox is.

In `sync_data/sync_data.sh`, set the value of `origin` to the path in which the test data is being stored locally. Set the value of `dest` to the path where the test data will be backed up, e.g. the Dropbox. This should look something like:
    
    source="/cygdrive/c/Users/USERNAME/Documents/EC-Lab/Data/Chris/PROPEL-1K/"
    dest="/cygdrive/c/Users/USERNAME/Dropbox/Data/PROPEL-1K/Electrochemical_Testing"

Naturally, the actual paths will depend on the particular system. If you would like the script to convert the newly copied .mpr files into .csv files as well, set `run_mpr2csv_bool=1`, otherwise 0. It is recommended (for performance reasons) to only do this on one of the systems that is syncing into the Dropbox, as it will analyze all newly acquired data each time the script is run.

In `sync_data/batch_mpr2csv.py`, set the value of `root_dir` to the path of the data directory, in the Windows format, e.g. `C:/Users/USERNAME/Dropbox/Data/PROPEL-1K/Electrochemical_Testing`. This will essentially be the Windows version of the path used previously for `dest`.

## Running the script
The script can be run manually by opening a cygwin terminal, navigating to the `sync_data` directory, and running `bash sync_data.sh`.

The script can be automated by adding a task to Windows Task Scheduler.