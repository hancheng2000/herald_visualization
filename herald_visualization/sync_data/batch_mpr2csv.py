import os, glob
from mpr2csv import cycle_mpr2csv

base_path = os.getcwd()
# Directory where all desired data resides
root_dir = '/Users/eschl/MIT Dropbox/Christopher Eschler/MIT/_Grad/Thesis/Data/PROPEL-1K/Electrochemical_Testing'
glob_list = glob.glob(r'**/*CC[0-9][0-9][0-9][A-Z]*/', root_dir=root_dir)
run_count = 0
for path in glob_list:
    full_path = os.path.join(root_dir, path)
    try:
        output_files = glob.glob(os.path.join(full_path, 'outputs', '*.csv'))
        if len(output_files) > 0:
            processed_time = max([os.path.getmtime(file) for file in output_files])
        else:
            processed_time = 0 # Makes program consider the (nonexistent) summary file as outdated
        latest_data_time = max([os.path.getmtime(file) for file in glob.glob(os.path.join(full_path, '*.mpr'))])
        if latest_data_time > processed_time:
            cycle_mpr2csv(full_path)
            run_count += 1
    except:
        # Return to the base path if there's an error, otherwise we're left stranded in a random dir
        os.chdir(base_path)
print(f"Located {len(glob_list)} data paths.")
print(f"Ran mpr2csv in {run_count} data paths.")