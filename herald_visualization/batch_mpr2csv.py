import os, glob
from mpr2csv import cycle_mpr2csv

base_path = os.getcwd()
data_path = ''
for path in glob.glob(r'**/*CC[0-9][0-9][0-9][A-Z]*/', root_dir=data_path):
    try:
        output_files = glob.glob(os.path.join(path, 'outputs', '*.csv'))
        if len(output_files) > 0:
            processed_time = max([os.path.getmtime(file) for file in output_files])
        else:
            processed_time = 0 # Makes program consider the (nonexistent) summary file as outdated
        latest_data_time = max([os.path.getmtime(file) for file in glob.glob(os.path.join(path, '*.mpr'))])
        if latest_data_time > processed_time:
            cycle_mpr2csv(path)
    except:
        os.chdir(base_path)