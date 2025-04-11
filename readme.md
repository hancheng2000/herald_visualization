This package is designed for plotting the data in herald project.
The example notebook represents a typical workflow that aims to be universal for all relevant electrochemical testings in the project:
1. locate into the folder with all 4 tests
2. convert mpr to csv files
3. plot the GITT/cycling profile of different cycles

Note that the plotting functions return fig and ax in case you want further modifications to the plot (e.g. add theoretical OCV for comparison), and you need to save the figure yourself (save_png is provided as an option in function input)

# Syncing data from lab computers
## Installation
### Install cygwin 
Cygwin is needed in order to run bash commands on Windows systems. Install it from here: https://www.cygwin.com/. Leave the installation directory as the default. During the installation process, on the step labeled Select Packages, set View to Full, search for `rsync` in the available modules, and select the latest available version in the New column.

### Write the sync script
Open Notepad++ (and/or download and install it if necessary). Make a new file and under Edit > EOL Conversion, select "Unix (LF)".
The contents of the file are as follows:

    #!/bin/bash
    origin={origin location}
    dest={destination location}
    rsync -auP "$origin" "$dest"

A few variables will need to be set to tell the scripts where data is located and where Dropbox is. Using UNIX-style paths starting at the `/cygdrive` mount point*, set the value of `origin` to the path in which the test data is being stored locally (do not include the curly braces shown above, but do use quotes around the path). Set the value of `dest` to the path where the test data will be backed up, e.g. the Dropbox. This should look something like:
    
    origin="/cygdrive/c/Users/USERNAME/Documents/EC-Lab/Data/Chris/PROPEL-1K/"
    dest="/cygdrive/c/Users/USERNAME/MIT Dropbox/Riley Hargrave/Data/PROPEL-1K/Electrochemical_Testing"

Naturally, the actual paths will depend on the particular system. Save this file as `sync_data.sh` in your desired location (typically I put it in the origin directory).

*Essentially, all paths used in cygwin will begin with `/cygdrive/c/...`. From there, the path is the same as the Windows path, but using forward slashes (/) instead of backslashes (\\).

### Running the script
The script can be run manually by opening a cygwin terminal, navigating to the directory with the sync_data.sh script, and running `bash sync_data.sh`. To automate running the script, open Windows Task Scheduler and select `Create Task...` under the Actions menu. Choose a name in General. Under Triggers, select New... and set up the desired trigger conditions. For instance, select Daily and Repeat task every: 6 hours (you can type custom times in the box) to sync every 6 hours. Under Actions, select New... and make a Start a program action. For Program/script, find the bash executable installed by cygwin, which will typically be at `C:\cygwin64\bin\bash.exe`. Under Add arguments, type the location of the script, in quotes using UNIX notation, e.g. `"/cygdrive/c/Users/USERNAME/Documents/EC-Lab/Data/Chris/sync_data.sh"`. This can be saved as a new task and it will run automatically.

### Why do this instead of telling the software to save directly into the Dropbox?
Dropbox does not handle frequent changes to large files well. It generates a cache under `{Dropbox main folder}/.dropbox.cache` that tracks changes to files. If large files are being written to every few seconds, this cache grows enormously and will eventually fill all remaining disk space causing a cascade of issues, including Dropbox no longer being able to sync. It is much more manageable to make syncing to Dropbox a less frequent operation and rsync provides a powerful, elegant tool to do so.