# Cycling Analysis, based on Navani
# Author: Chris Eschler, Hancheng Zhao, 2024

import pandas as pd
import glob
import re
import os
import herald_visualization.echem as ec
import json

# Functions

# Check for a file defining the path to the local system's data directory
# If it exists, use that path as the default in id_to_path
# Otherwise, ask the user to specify the path and save it into a file
data_path_file = 'data_path.txt'
if os.path.isfile(data_path_file):
    f = open(data_path_file, 'r')
    data_path = f.readline()
    f.close()
else:
    print("The id_to_path function needs the path to your test data directory in order to search.")
    print(f"This can be edited by changing {data_path_file}.")
    data_path = input("Input your data directory path: ")
    f = open(data_path_file, 'w')
    f.write(data_path)
    f.close()

def id_to_path(cellid, root_dir=data_path):
    """
    Find the correct directory path to a data folder from the cell ID
    """
    glob_str = os.path.join('**', '*'+cellid+'*/')
    paths = glob.glob(glob_str, root_dir=root_dir, recursive=True)
    if len(paths) == 1:
        return os.path.join(root_dir, paths[0])
    elif len(paths) == 0:
        print(f"No paths matched for {cellid}")
    else:
        print(f"Too many paths matched for {cellid}: {paths}")

def import_data_using_listfile(df, listfile='stitch.txt'):
    """
    Import data files using an explicit list in a text file.
    Each line in the listfile is a filename.
    optionally followed by 'T' (no quotations needed).
    - T: Offset time values in the imported file by the latest timestamp in previously imported data.
        Allows for restarted tests to be stitched back together in time.
    """
    data_filenames = []
    offset_bools = []

    try:
        with open(listfile, 'r') as file:
            lines = file.readlines()
            # If listfile is empty, skip analysis
            if lines == []:
                print(f"Analysis skipped due to empty {listfile}.")
                return [], pd.DataFrame()
            
            for line in lines:
                # Separate each line by whitespace into the filename and the flags for which fields to offset (i.e. T for time)
                line = line.strip().split(None, maxsplit=1)
                filename = line[0]
                # Keep track of whether a time offset needs to be added when each file is loaded
                if len(line) > 1 and 'T' in line[1].upper():
                    offset_bools += [True]
                else:
                    offset_bools += [False]

                # If the filename does not correspond to a valid file, prompt the user to continue without it                        
                if not os.path.isfile(filename):
                    inp = input(f"File {filename} from {listfile} could not be found. Enter 'y' if you wish to skip it and continue anyway.")
                    if inp.lower() == 'y':
                        print(f"File {filename} skipped. Analysis continuing.")
                    else:
                        print("Analysis terminated by user.")
                        return [], pd.DataFrame()
                # If the filename is already in data_filenames, prompt the user to continue
                elif filename in data_filenames:
                    inp = input(f"Filename {filename} is duplicated in {listfile}. Enter 'y' if you wish for it to be duplicated in the analysis. Otherwise, the repeat instance will be ignored.")
                    if inp.lower() == 'y':
                        data_filenames += [filename]
                # Else filename corresponds to a filename that is not already in data_filenames
                else:
                    data_filenames += [filename]
    
    except IOError:
        # If listfile cannot be opened
        print(f"Unable to open/read {listfile}. Analysis skipped.")
        return [], pd.DataFrame()
    
    for i, (file, bool) in enumerate(zip(data_filenames, offset_bools)):
        # Calculate time_offset from existing data in df
        if bool:
            try:
                time_offset = df['Time'].max()
            except KeyError: # Takes care of case where True appears before there is data present
                time_offset = 0.0
        else:
            time_offset = 0.0
        # Only do the full processing step on the last file to load, in order to speed up loading
        if i == len(data_filenames) - 1:
            df = ec.echem_file_loader(file, df_to_append_to=df, time_offset=time_offset, calc_cycles_and_cap=True)
        else:
            df = ec.echem_file_loader(file, df_to_append_to=df, time_offset=time_offset, calc_cycles_and_cap=False)
    
    print(f"Imported using {listfile}.")
    return data_filenames, df

def import_data_using_pattern(df, extension):
    """
    Import data files by searching the data directory for files matching a pattern.
    It is expected that all of the files found by this method belong to the same test (not a restarted test).
    Otherwise, a listfile should be used to explicitly indicate a restarted test.
    """
    data_filenames = []

    # Keyword expected to be in each filename to be analyzed (if there are multiple data files)
    search_keyword = '_GCPL_'
    match_string = f'*{search_keyword}*.{extension}'
    data_filenames += glob.glob(match_string)
    # # Order files by time of creation so that they get added in sequence to the dataframe
    # data_filenames.sort(key=os.path.getctime)
    
    # Tests that should not be used even if they are the only ones found
    blacklist = ['EIS_', '_OCV_', 'summary_cycle']
    # Check whether no matches with search_keyword were found
    if len(data_filenames) == 0:
        # If only one data file was made by the test, it should be used even if it lacks the search_keyword
        # as long as it does not contain any of the blacklisted keywords
        data_filenames = [file for file in glob.glob(f'*.{extension}') if not any(blacklist_item in file for blacklist_item in blacklist)]
        if len(data_filenames) == 0:
            print(f"No .{extension} files found by auto-search.")
            return [], pd.DataFrame()
        elif len(data_filenames) > 1:
            print(f"Too many .{extension} files found by auto-search. Use a listfile to indicate which files to analyze.")
            print(data_filenames)
            return [], pd.DataFrame()

    for i, file in enumerate(data_filenames):
        # Only do the full processing step on the last file to load, in order to speed up loading
        if i == len(data_filenames) - 1:
            df = ec.echem_file_loader(file, df_to_append_to=df, calc_cycles_and_cap=True)
        else:
            df = ec.echem_file_loader(file, df_to_append_to=df, calc_cycles_and_cap=False)

    return data_filenames, df

def settings_filename_from_data_filename(data_filename):
    # Determine the settings filename that EC-Lab or BT-Export would autogenerate for a given data filename
    file_extension = os.path.splitext(data_filename)[-1].lower()
    if file_extension == '.mpr':
        # Files created by EC-Lab
        mpr_regex = r'(_\d{2}_[a-zA-Z]+)?_C\d{2}.mpr'
        return re.sub(mpr_regex, '.mps', data_filename)
    elif file_extension == '.csv':
        # Files exported by BT-Export have the same filename for data and settings
        return re.sub('.csv', '.json', data_filename)
    else:
        print("Unsupported filetype for settings file.")
        return None

def import_settings(settings_filename):
    # Imports info about the cell from the settings file.
    # Key: variable in the settings file
    # Val: string in the associated line of the settings file
    # Handles both .mps files from EC-Lab and .json files from BT-Export
    file_extension = os.path.splitext(settings_filename)[-1].lower()
    val_regex = r'\d+\.?\d*' # Regex for numeric values
    cell_props = {}

    try:
        if file_extension == '.mps':
            settings_keys = {
                'active_material_mass': "Mass of active material",
                'x_at_mass': " at x = ",
                'empty_mol_weight': "Molecular weight of active material",
                'interc_weight': "Atomic weight of intercalated ion",
                'x_at_start': "Acquisition started at",
                'e_per_ion': "Number of e- transfered per intercalated ion",
                'surface_area': "Electrode surface area"
            }

            with open(settings_filename, 'r', errors='ignore') as file:
                # Iterate through settings file to fill in active material mass and other cell parameters
                i = 0
                while True:
                    i += 1
                    line = file.readline()
                    if not line:
                        break
                    else:
                        new_key = [key for key, val in settings_keys.items() if val in line]
                        if new_key:
                            match = re.findall(val_regex, line)
                            # The final matching string is used
                            cell_props[new_key[0]] = float(match[-1])         
                return cell_props
        elif file_extension == '.json':
            settings_keys = {
                'active_material_mass': "mass of active material",
                'x_at_mass': "x mass",
                'empty_mol_weight': "molecular weight",
                'interc_weight': "intercalated ion molecular weight of active material",
                'x_at_start': "x0",
                'e_per_ion': "number of electrons",
                'surface_area': "surface area"
            }
            with open(settings_filename, 'r', errors='ignore') as file:
                settings = json.load(file)
                for key, val in settings_keys.items():
                    match = re.findall(val_regex, settings['dutType'][val])
                    cell_props[key] = float(match[-1])
                return cell_props

    except:
        print("Unable to read settings file.")
        
def total_AM_mass(cell_props):
    """
    Uses cell properties imported from a settings file to convert the cathode AM mass
    (inputted into EC-lab at test start) into a total AM mass, i.e. unlithiated to lithiated.
    Assumes that e_per_ion is the total number of electrons transferred for a complete reaction,
    e.g. 3 for 3Li + FeF3 <-> Fe + 3LiF.
    """
    interc_weight = cell_props['interc_weight']
    x_at_mass = cell_props['x_at_mass']
    empty_mol_weight = cell_props['empty_mol_weight']
    e_per_ion = cell_props['e_per_ion']
    active_material_mass = cell_props['active_material_mass']

    # Molar mass of material in starting condition
    starting_mol_weight = empty_mol_weight + (x_at_mass * interc_weight)

    # Molar mass of fully intercalated cathode material
    full_mol_weight = empty_mol_weight + (interc_weight * e_per_ion)

    # Mass of fully intercalated cathode material
    full_active_material_mass = active_material_mass * full_mol_weight / starting_mol_weight
    return full_active_material_mass
    
# convert all relevant data to .csv
def cycle_mpr2csv(
        dir_name,
        listfile='stitch.txt'
    ):
    df = pd.DataFrame()
    home_dir = os.getcwd()
    os.chdir(dir_name)
    print(f"\nRunning in {dir_name}")
    
    # Import data and settings
    if os.path.isfile(listfile):
        data_filenames, df = import_data_using_listfile(df=df, listfile=listfile)
    elif len(glob.glob('*.mpr')) > 0:
        data_filenames, df = import_data_using_pattern(df=df, extension='mpr')
    elif len(glob.glob('*.csv')) > 0:
        data_filenames, df = import_data_using_pattern(df=df, extension='csv')
    
    # Break if data import is unsuccessful
    if df.empty:
        return None
    # Otherwise continue
    print(f"Data file(s): {data_filenames}")

    # Check that all test files came from the same settings file, i.e. test
    settings_filenames = [settings_filename_from_data_filename(filename) for filename in data_filenames]
    if len(set(settings_filenames)) > 1:
        print("Auto-search found files from multiple tests. The first test's settings file is being used. A listfile may be needed if not already present.")
    # Assume that the cell characteristics are the same in all settings files,
    # only look at settings file associated with first data file
    settings_filename = settings_filenames[0]
    cell_props = import_settings(settings_filename)
    print(f"Settings file: {settings_filename}")

    # Post-process
    if cell_props:
        print(f"Cell properties: {cell_props}")
        mass = cell_props['active_material_mass']
        full_mass = total_AM_mass(cell_props)
        area = cell_props['surface_area']
        df = ec.df_post_process(df, 
                                mass=mass,
                                full_mass=full_mass, 
                                area=area)
    else:
        print(f"No cell properties imported.")

    # Export navani-processed dataframe as a .csv for later use
    # Path for .csv with same filename as .mps or .json
    os.makedirs('outputs', exist_ok=True)
    settings_extension = os.path.splitext(settings_filename)[-1].lower()
    output_filename = settings_filename.removesuffix(settings_extension) + '.csv'
    data_csv_filename = os.path.join('outputs', output_filename)
    df.to_csv(data_csv_filename)
    print(f"CSV exported to: {data_csv_filename}")

    # Export a cycle summary .csv if multiple half cycles are present
    # TODO: catch error when properties are not present
    if df['half cycle'].max() >= 1:
        cycle_summary = ec.cycle_summary(df, mass=mass, full_mass=full_mass, area=area)
        cycle_summary_csv_filename = os.path.join('outputs', 'cycle_summary.csv')
        cycle_summary.to_csv(cycle_summary_csv_filename)
        # print(f"Cycle summary CSV exported to: {cycle_summary_csv_filename}")
    
    os.chdir(home_dir)
    return df

def eis_mpr2csv(
    dir_name,
    state = 'gitt',
):
    """
    Convert EIS mpr files to csv files
    Args:
    - dir_name: Directory containing the mpr files
    - state: either 'gitt' or 'as-built'. In the case of 'gitt', the *03_PEIS*.mpr file is used. In the case of 'as-built', the *01_PEIS*.mpr file is used.
    """    
    from galvani import BioLogic
    if state == 'gitt':
        mpr_files = glob.glob(dir_name+'/*03_PEIS*.mpr')
    elif state == 'as-built':
        mpr_files = glob.glob(dir_name+'/*01_PEIS*.mpr')
    else:
        raise NotImplementedError("Only 'gitt' and 'as-built' states are supported.")
    mpr_file = mpr_files[0]
    data = BioLogic.MPRfile(mpr_file)
    df = pd.DataFrame(data.data)
    file_name = mpr_file.split('/')[-1].replace('.mpr', '.csv')
    home_dir = os.getcwd()
    df.to_csv(os.path.join(dir_name, 'outputs', file_name), index=False)
    return df