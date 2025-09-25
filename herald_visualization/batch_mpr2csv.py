import os, glob, sys, argparse
from herald_visualization.mpr2csv import cycle_mpr2csv

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--all', help="Analyze even if no new data is found since last analysis.", action='store_true')
parser.add_argument('-d', '--dry-run', help="Do not export csv after analysis.", action='store_false')
parser.add_argument('-o', '--opt-timeout', help="Wait x sec for user response at prompts. Set to -1 to wait indefinitely. Default 10. (Not implemented)", default=10.0, type=float)
parser.add_argument('-r', '--recent', help="Only analyze tests with new data from the past x days.", type=float)
parser.add_argument('-s', '--warn-size', help="Warn and wait for input if data filesize exceeds this value (MB). (Not implemented)", type=float)
args = parser.parse_args()
export_csv = args.dry_run # Set flag for exporting csv (False if -d arg is given)
if args.recent:
    import time
    # Determine the earliest Unix timestamp within the previous args.recent days
    earliest_time = time.time() - (args.recent*86400)

base_path = os.getcwd()

# Check for a file defining the path to the local system's data directory
# If it exists, use that path as the default in id_to_path
# Otherwise, ask the user to specify the path and save it into a file
data_path_file = 'data_path.txt'
if os.path.isfile(data_path_file):
    f = open(data_path_file, 'r')
    root_dir = f.readline()
    f.close()
else:
    print("The id_to_path function needs the path to your test data directory in order to search.")
    print(f"This can be edited by changing {data_path_file}.")
    root_dir = input("Input your data directory path: ")
    f = open(data_path_file, 'w')
    f.write(root_dir)
    f.close()

glob_list = glob.glob(r'**/*CC[0-9][0-9][0-9][A-Z]*/', root_dir=root_dir)
glob_list.sort() # Sorting alphanumerically makes it easier to determine how far along the batch is while running
run_count = 0
for path in glob_list:
    full_path = os.path.join(root_dir, path)
    try:
        output_files = glob.glob(os.path.join(full_path, 'outputs', '*.csv'))
        if len(output_files) > 0:
            processed_time = max([os.path.getmtime(file) for file in output_files])
        else:
            processed_time = 0 # Makes program consider the (nonexistent) summary file as outdated
        data_files = glob.glob(os.path.join(full_path, '*.mpr')) + glob.glob(os.path.join(full_path, '*.csv'))
        # Looks for both .mpr files from EC-Lab and .csv files from BT-Export
        if args.warn_size: # If flag is set to warn above a certain file size
            total_data_size = sum([os.path.getsize(file) for file in data_files])/(1024**2) # Calculate total size of data files in MB
            # TODO: implement warning size
        latest_data_time = max([os.path.getmtime(file) for file in data_files])
        # Only look at data newer than the exported files, unless --all is set
        if latest_data_time > processed_time or args.all:
            # If --recent is set, only consider tests with data newer than requested, otherwise all tests from above
            if not args.recent or latest_data_time > earliest_time:
                cycle_mpr2csv(full_path, export_csv=export_csv)
                run_count += 1
    except:
        # Return to the base path if there's an error, otherwise we're left stranded in a random dir
        os.chdir(base_path)
print(f"\nLocated {len(glob_list)} data paths.")
print(f"Ran cycle_mpr2csv in {run_count} data paths.")