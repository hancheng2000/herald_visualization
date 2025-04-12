import os, glob, sys
from herald_visualization.mpr2csv import cycle_mpr2csv

if len(sys.argv) > 1 and sys.argv[1] == '-a': # -a can be entered as a switch following the script name when executing
    process_all_files = True # Analyze every file located
else:
    process_all_files = False # Only analyze fresh data
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
        latest_data_time = max([os.path.getmtime(file) for file in data_files])
        if latest_data_time > processed_time or process_all_files:
            cycle_mpr2csv(full_path)
            run_count += 1
    except:
        # Return to the base path if there's an error, otherwise we're left stranded in a random dir
        os.chdir(base_path)
print(f"Located {len(glob_list)} data paths.")
print(f"Ran mpr2csv in {run_count} data paths.")