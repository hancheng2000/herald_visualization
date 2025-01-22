# Cycling Analysis, based on Navani
# Author: Chris Eschler, Hancheng Zhao, 2024

import pandas as pd
import glob
import re
import os
import sys
import herald_visualization.echem as ec

# Functions
def append_import_to_df(df, filename, offset_flags=''):
    """
    Add the contents of a file to an existing DataFrame, possibly offsetting the time and capacity in the new data.
    The offsets will be based on the max time and latest capacity in the existing DataFrame.

    Args:
    - df (pandas.DataFrame): The DataFrame to be added to.
    - filename (str): The filepath that should be imported and added to the DataFrame.
    - offset_flags (str, opt): String containing T and/or C to indicate that time and/or capacity should be offset.

    Returns:
    - pandas.DataFrame: A DataFrame with the data imported from filename appended to df.
    """
    time_offset = 0
    capacity_offset = 0

    # Files larger than this many bytes will throw a warning when being imported
    # giving the opportunity to abort analysis
    filesize_warn_threshold = 200e6

    filesize = os.stat(filename).st_size
    if filesize > filesize_warn_threshold:
        inp = input(f"File {filename} contains {filesize} bytes. Press 'y' if you would like to continue analyzing it.")
        if inp.lower() == 'y':
            print("Analysis continuing.")
        else:
            print("Analysis terminated.")
            sys.exit(1)

    # TODO: catch errors from empty df
    if 't' in offset_flags.lower():
        time_offset = df['time/s'].max()
    
    if 'c' in offset_flags.lower():
        df = df.sort_values(by='time/s', ignore_index=True)
        capacity_offset = df['Q charge/discharge/mA.h'].iloc[-1]

    new_data = ec.multi_file_biologic(filename, time_offset=time_offset, capacity_offset=capacity_offset)
    df = pd.concat([df, new_data], ignore_index=True)
    return df

def import_data_using_listfile(df, listfile='stitch.txt'):
    """
    Import data files using an explicit list in a text file.
    Each line in the listfile is a filename.
    optionally followed by 'T' and/or 'C' (no quotations needed).
    - T: Offset time values in the imported file by the latest timestamp in previously imported data.
        Allows for restarted tests to be stitched back together in time.
    - C: Offset capacity values in the imported file by the latest capacity in previously imported data.
        Use if a test was restarted after interruption within a cycle.
    """
    data_filenames = []

    try:
        with open(listfile, 'r') as file:
            lines = file.readlines()
            # If listfile is empty, skip analysis
            if lines == []:
                print(f"Analysis skipped due to empty {listfile}.")
                sys.exit(1)
            else:
                for line in lines:
                    # Separate each line by whitespace into the filename and the flags for which fields to offset (i.e. T for time and C for capacity)
                    line = line.strip().split(None, maxsplit=1)
                    filename = line[0]
                    if len(line) > 1:
                        offset_flags = line[1]
                    else:
                        offset_flags = ''

                    # If the filename does not correspond to a valid file, prompt the user to continue without it                        
                    if not os.path.isfile(filename):
                        inp = input(f"File {filename} from {listfile} could not be found. Enter 'y' if you wish to skip it and continue anyway.")
                        if inp.lower() == 'y':
                            print(f"File {filename} skipped. Analysis continuing.")
                        else:
                            print("Analysis terminated by user.")
                            sys.exit(1)
                    # If the filename is already in data_filenames, prompt the user to continue
                    elif filename in data_filenames:
                        inp = input(f"Filename {filename} is duplicated in {listfile}. Enter 'y' if you wish for it to be duplicated in the analysis. Otherwise, the repeat instance will be ignored.")
                        if inp.lower() == 'y':
                            df = append_import_to_df(df, filename, offset_flags=offset_flags)
                            data_filenames += [filename]
                    # Else filename corresponds to a filename that is not already in data_filenames
                    else:
                        df = append_import_to_df(df, filename, offset_flags=offset_flags)
                        data_filenames += [filename]
                print(f"Imported using {listfile}.")
                return data_filenames, df
    except IOError:
        # If listfile cannot be opened
        print(f"Unable to open/read {listfile}. Analysis skipped.")
        sys.exit(1)

def import_data_using_pattern(df):
    """
    Import data files by searching the data directory for files matching a pattern.
    It is expected that all of the files found by this method belong to the same test (not a restarted test).
    Otherwise, a listfile should be used to explicitly indicate a restarted test.
    """
    data_filenames = []

    # Keyword expected to be in each filename to be analyzed (if there are multiple .mpr files)
    search_keyword = '_GCPL_'
    match_string = '*'+search_keyword+'*.mpr'
    data_filenames += glob.glob(match_string)
    # # Order files by time of creation so that they get added in sequence to the dataframe
    # data_filenames.sort(key=os.path.getctime)

    # Check whether no matches with search_keyword were found
    if len(data_filenames) == 0:
        # If only one .mpr file was made by the test, it should be used even if it lacks the search_keyword
        data_filenames += glob.glob('*.mpr')
        if len(data_filenames) == 0:
            print("No .mpr files found by auto-search.")
            sys.exit(1)
        elif len(data_filenames) > 1:
            print("Too many .mpr files found by auto-search. Use a listfile to indicate which files to analyze.")
            sys.exit(1)

    for filename in data_filenames:
        df = append_import_to_df(df, filename)
    return data_filenames, df

def import_settings(data_filenames):
    """
    Imports info about the cell from the .mps settings file.
    """
    # Back-generate the .mps filename that would create each file and check that they are the same
    mpr_regex = r'(_\d{2}_[a-zA-Z]+)?_C\d{2}.mpr'
    mps_filenames = [re.sub(mpr_regex, '.mps', filename) for filename in data_filenames]
    if len(set(mps_filenames)) > 1:
        print("Auto-search found files from multiple tests. The first test's settings file is being used.")
    
    # Assume that the cell characteristics are the same in all settings files,
    # only look at settings file associated with first data file
    settings_filename = mps_filenames[0]

    # Key: variable in the settings file
    # Val: string in the associated line of the settings file
    settings_keys = {
        'active_material_mass': "Mass of active material",
        'x_at_mass': " at x = ",
        'empty_mol_weight': "Molecular weight of active material",
        'interc_weight': "Atomic weight of intercalated ion",
        'x_at_start': "Acquisition started at",
        'e_per_ion': "Number of e- transfered per intercalated ion",
        'surface_area': "Electrode surface area"
    }
    val_regex = r'\d+\.?\d*'
    cell_props = {}

    try:
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
            
            return cell_props, settings_filename
      
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

def id_to_path(cellid, root_dir='../..'):
    """
    Find the correct directory path to a data folder from the cell ID
    """
    glob_str = os.path.join('**', '*'+cellid+'*/')
    paths = glob.glob(glob_str, root_dir=root_dir, recursive=True)
    if len(paths) == 1:
        return os.path.join(root_dir, paths[0])
    elif len(paths) == 0:
        print("No paths matched")
    else:
        print(f"Too many paths matched: {paths}")
    
# convert all relevant data to .csv
def cycle_mpr2csv(
        dir_name,
        listfile='stitch.txt',
    ):
    df = pd.DataFrame()
    home_dir = os.getcwd()
    os.chdir(dir_name)
    print("Running in", os.getcwd())
    
    # Import data and settings
    if os.path.isfile(listfile):
        data_filenames, df = import_data_using_listfile(df=df)
    else:
        data_filenames, df = import_data_using_pattern(df=df)
    print(f"Data file(s): {data_filenames}")
    cell_props, settings_filename = import_settings(data_filenames)
    print(f"Settings file: {settings_filename}")

    # Post-process
    df = df.sort_values(by='time/s', ignore_index=True)
    df = ec.biologic_processing(df)
    if cell_props:
        print(f"Cell properties: {cell_props}")
        mass = None
        area = None
        if 'active_material_mass' in cell_props.keys():
            mass = cell_props['active_material_mass']
            full_mass = total_AM_mass(cell_props)
        if 'surface_area' in cell_props.keys():
            area = cell_props['surface_area']
        df = ec.df_post_process(df, 
                                mass=mass,
                                full_mass=full_mass, 
                                area=area)
    else:
        print(f"No cell properties imported.")
        df = ec.df_post_process(df)
    # print(df)

    # Export navani-processed dataframe as a .csv for later use
    # Path for .csv with same filename as .mps, based on .mpr
    os.makedirs('outputs', exist_ok=True)
    output_filename = settings_filename.replace('.mps','.csv')
    data_csv_filename = os.path.join('outputs', output_filename)
    df.to_csv(data_csv_filename)
    print(f"CSV exported to: {data_csv_filename}")

    # Export a cycle summary .csv if multiple half cycles are present
    if df['half cycle'].max() >= 1:
        cycle_summary = ec.cycle_summary(df, mass=mass, full_mass=full_mass, area=area)
        cycle_summary_csv_filename = os.path.join('outputs', 'cycle_summary.csv')
        cycle_summary.to_csv(cycle_summary_csv_filename)
        print(f"Cycle summary CSV exported to: {cycle_summary_csv_filename}")
        
    # # Print cell metrics
    # number_of_cycles = len(cycle_summary)
    # print(f"Number of cycles: {number_of_cycles}")
    # print(f"UCV (1st cycle): {cycle_summary.iloc[0]['UCV']}")
    # print(f"LCV (1st cycle): {cycle_summary.iloc[0]['LCV']}")
    
    # if cell_props:
    #     print(f"Spec. cap. (1st cycle, cathode AM): {cycle_summary.iloc[0]['Specific Discharge Capacity']}")
    #     print(f"GED (1st cycle, cathode AM): {cycle_summary.iloc[0]['Specific Discharge Energy']}")
    #     # Some metrics can only be determined if there are at least 2 cycles in the test
    #     if number_of_cycles >= 2:
    #         print(f"Cycle stability (2nd discharge energy/1st discharge energy): {cycle_summary.iloc[1]['Specific Discharge Energy']/cycle_summary.iloc[0]['Specific Discharge Energy']}")
    #         print(f"Discharge efficiency (1st discharge energy/2nd charge energy): {cycle_summary.iloc[0]['Specific Discharge Energy']/cycle_summary.iloc[1]['Specific Charge Energy']}")
    
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


# # Split data into sections based on current to figure out when relaxation is happening
# i = 0
# while i < len(data):

#     while np.array(data['control/V/mA'])[i] < 0.0:
#         i=i+1
#     start_indices.append(i)

#     while np.array(data['control/V/mA'])[i] == 0.0 and i <= len(data):
#         i=i+1
#         if i == len(data):
#             break
#     end_indices.append(i)


# # Plot relaxation curves for each GITT relaxation
# os.makedirs('relaxations', exist_ok=True)
# for i in range(len(start_indices)):
#     x_data = data['time/s'][start_indices[i]:end_indices[i]]/3600 - data['time/s'][start_indices[i]]/3600
#     y_data = data['Ewe/V'][start_indices[i]:end_indices[i]]
#     np.savetxt('relaxations/'+str(i)+'.csv', np.transpose([x_data, y_data]), delimiter=',')
#     # popt, pcov = curve_fit(func, x_data, y_data, p0=popt, method='lm', maxfev=10000)
#     # print(popt)
#     # plt.plot(x_data, func(x_data, *popt), 'r-', label='fit')
#     plt.plot(x_data, y_data, label='Data', color='black')
#     plt.ylim(voltage_limits)
#     plt.xlabel('Time (hr)')
#     plt.ylabel('Voltage (v)')
#     plt.title(str(i+1)+'th relaxation')
#     plt.savefig('relaxations/'+str(i)+'.png')
#     plt.close()

# # Plot pseudo-OCV curve
# plt.plot(-1*data['(Q-Qo)/mA.h'][np.array(end_indices)], data['Ewe/V'][np.array(end_indices)])
# plt.ylim(voltage_limits)
# plt.xlabel('Discharge capacity (mAh)')
# plt.ylabel('Voltage (V)')
# plt.title('Pseudo-OCV')
# plt.savefig('plots/pseudo_OCV.png')
# plt.close()
