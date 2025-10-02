from galvani import MPRfile
from galvani import res2sqlite as r2s

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import sqlite3
import os
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path

# MAIN FUNCTION
# TODO: correct output for non-BioLogic files
def echem_file_loader(filepath, 
                      df_to_append_to=None, 
                      time_offset=0.0,
                      calc_cycles_and_cap=True,
                      debounce=True):
    """
    Loads a variety of electrochemical filetypes and tries to construct the most useful measurements in a
    consistent way, with consistent column labels. Outputs a dataframe with these constructed columns:
    
    - Time: (unit: s) time from beginning of test
    - dt: (unit: s) time since previous data point
    - Cycle change: indicates an increment in cycle count
    - half cycle: discharge and charge count as separate half cycles; rests are included with preceding data
    - full cycle: defined as charge followed by discharge
    - Voltage: (unit: V) cell voltage
    - Current: (unit: mA) cell current, +ve for charge and -ve for discharge
    - state: 1=charging, 0=rest, -1=discharging
    - Power: (unit: mW) cell power, +ve for charge and -ve for discharge
    - Capacity: (unit: mAh) capacity passed in a given half cycle (resets to 0 at beginning of each half cycle)
    - Q-Q0: (unit: mAh) total capacity passed up to a given point in a test (increases during charge, decreases during discharge)
    
    Parameters:
        filepath (str): The path to the electrochemical file.
        The following parameters are only implemented for Biologic files:
        df_to_append_to (pd.DataFrame): DataFrame to append the newly loaded data to the end of, for combining multiple raw data files.
        time_offset (float): Amount to increase the time values of newly loaded data in order to stitch together raw data files.
        calc_cycles_and_cap (bool): True if the cycles and capacity should be calculated after appending the new data.
            Set to True after the final piece of raw data has been imported when stitching files.
        debounce (bool): Runs debouncer function to remove extraneous half cycles.
    
    Returns:
        pandas.DataFrame: A dataframe with the constructed columns.

    The 2 dataframes that occur within each handler are raw (raw data as imported from the test file) and df (the processed data).
    """
    extension = os.path.splitext(filepath)[-1].lower()
    
    # Biologic .mpr file
    if extension == '.mpr':
        gal_file = MPRfile(os.path.join(filepath))
        raw = pd.DataFrame(data=gal_file.data)
        df = biologic_processing(raw)

    # arbin .res file - uses an sql server and requires mdbtools installed
    # sudo apt get mdbtools for windows and mac
    elif extension == '.res': 
        Output_file = 'placeholder_string'
        r2s.convert_arbin_to_sqlite(os.path.join(filepath), Output_file)
        dat = sqlite3.connect(Output_file)
        query = dat.execute("SELECT * From Channel_Normal_Table")
        cols = [column[0] for column in query.description]
        df = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        dat.close()
        df = arbin_res(df)

    # Currently .txt files are assumed to be from an ivium cycler - this may need to be changed
    # These have time, current and voltage columns only
    elif extension == '.txt':
        raw = pd.read_csv(os.path.join(filepath), sep='\t')
        # Checking columns are an exact match
        if set(['time /s', 'I /mA', 'E /V']) - set(raw.columns) == set([]):
            df = ivium_processing(raw)
        else:
            raise ValueError('Columns do not match expected columns for an ivium .txt file')

    # Landt and Arbin can output .xlsx and .xls files
    elif extension in ['.xlsx', '.xls']:
        if extension == '.xlsx':
            xlsx = pd.ExcelFile(os.path.join(filepath), engine='openpyxl')
        else:
            xlsx = pd.ExcelFile(os.path.join(filepath))

        names = xlsx.sheet_names
        # Use different Landt processing if all exported as one sheet (different versions of Landt software)
        if len(names) == 1:
            raw = xlsx.parse(0)
            df = new_landt_processing(raw)

        # If Record is a sheet name, then it is a Landt file
        elif "Record" in names[0]:
            df_list = [xlsx.parse(0)]
            if not isinstance(df_list, list) or not isinstance(df_list[0], pd.DataFrame):
                raise RuntimeError("First sheet is not a dataframe; cannot continue parsing {filepath=}")
            col_names = df_list[0].columns

            for sheet_name in names[1:]:
                if "Record" in sheet_name:
                    if len(xlsx.parse(sheet_name, header=None)) != 0:
                        df_list.append(xlsx.parse(sheet_name, header=None))
            for sheet in df_list:
                if not isinstance(sheet, pd.DataFrame):
                    raise RuntimeError("Sheet is not a dataframe; cannot continue parsing {filepath=}")
                sheet.columns = col_names
            raw = pd.concat(df_list)
            raw.set_index('Index', inplace=True)
            df = old_landt_processing(raw)

        # If Channel is a sheet name, then it is an arbin file
        else:
            df_list = []
            # Remove the Channel_Chart sheet if it exists as it's arbin's charting sheet
            if 'Channel_Chart' in names:
                names.remove('Channel_Chart')
            for count, name in enumerate(names):
                if 'Channel' in name and 'Chart' not in name:
                    df_list.append(xlsx.parse(count))
            if len(df_list) > 0:
                raw = pd.concat(df_list)
                df = arbin_excel(raw)
            else:
                raise ValueError('Names of sheets not recognised')
            
    # Neware files are .nda or .ndax
    elif extension in (".nda", ".ndax"):
        df = neware_reader(filepath)

    # If the file is a csv previously processed by navani or exported by BT-Export
    # Check for the columns that are expected in each case
    elif extension == '.csv':
        # Import just the first row to check which format the columns fit
        df_col = pd.read_csv(filepath, nrows=0, sep=None, engine='python')
        navani_expected_columns = ['Time', 'Current', 'Voltage', 'half cycle', 'full cycle', 'state', 'Capacity', 'Q']
        btexport_expected_columns = ['Sample Index', 'Time / s', 'U / V', 'I / A', 'Q / C']
        if all(col in df_col.columns for col in navani_expected_columns):
            df = pd.read_csv(filepath, 
                            index_col=0,
                            dtype={'Time': np.float64, 'Capacity': np.float64, 'Voltage': np.float64, 'Current': np.float64,
                                    'full cycle': np.int32, 'half cycle': np.int32, 'state': np.int16})
        elif all(col in df_col.columns for col in btexport_expected_columns):
            raw = pd.read_csv(filepath,
                             index_col=0,
                             sep=';',
                             usecols=btexport_expected_columns,
                             dtype={'Time / s': np.float64, 'U / V': np.float64, 'I / A': np.float64, 'Q / C': np.float64})
            df = bt_export_processing(raw)
        else:
            raise ValueError('Columns do not match expected columns for navani-processed csv')

    # If it's a filetype not seen before raise an error
    else:
        print(extension)
        raise RuntimeError(f"Filetype {extension} not recognised.")

    # Offset time and concatenate if df_to_append_to exists
    # If df_to_append_to is None, this will just return df
    df['Time'] += time_offset
    df = pd.concat([df_to_append_to, df], ignore_index=True)
    df.index.name = 'id'

    # Stop here if the cycles and capacity don't need to be calculated yet
    if not calc_cycles_and_cap:
        return df

    # Sort by time before calculating cycle numbers, etc.
    # In case multiple points have the same time value, sort by id
    df.sort_values(by=['Time','id'], inplace=True)

    # Calculate Q (running total capacity) to use as a check for half cycle alternation
    df['Q'] = df['dQ'].cumsum()

    # Calculate cycle numbers
    rest_mask = (df['state'] == 0)
    df['cycle change'] = False
    df['rest count'] = rest_mask.cumsum() # Helper column
    df['cycle change candidate'] = False # Helper column
    if (~rest_mask).any(): # If there is non-rest data
        # When state changes from 1 to -1 or -1 to 1, it is a candidate for a cycle change
        # But could also be caused by current transients
        # There should be rest between each charge and discharge half cycle, so check for an increment in number of rest points
        df.loc[~rest_mask, 'cycle change candidate'] = df.loc[~rest_mask, 'state'].ne(df.loc[~rest_mask, 'state'].shift())
        df.loc[df['cycle change candidate'], 'cycle change'] = df.loc[df['cycle change candidate'], 'rest count'].ne(df.loc[df['cycle change candidate'], 'rest count'].shift())
        df.drop(columns=['rest count', 'cycle change candidate'], inplace=True) # Clean up helper columns
    else:
        # If nothing is found for not_rest_idx, then all points are at rest
        # Therefore all the following values can be set to 0
        df['half cycle'] = 0
        df['full cycle'] = 0
        df['Q'] = 0
        df['Capacity'] = 0
        df['Power'] = 0
        print("No charge or discharge data found.")
        return df
    df['half cycle'] = (df['cycle change'] == True).cumsum() # Each time a cycle change occurs, increment half cycle
    # For each cycle, determine the state from change in Q
    # Most of the time this is the same as 'state' calculated at each point from current, except in the case of transients
    cycle_state = df.groupby('half cycle')['Q'].agg(lambda x: int(np.sign(x.iloc[-1] - x.iloc[0])))
    df['cycle state'] = df['half cycle'].map(cycle_state) # Assign a value of cycle state to each point in df
    # Sanity check: the half cycle 0 should have a change in Q of 0 (only rest)
    # and each subsequent half cycle should alternate in sign (charge then discharge, etc.)
    alternation_ok = (cycle_state.values[2:] * cycle_state.values[1:-1] == -1).all() # True if every half cycle starting with 1 alternates cycle state sign
    if cycle_state[0] != 0:
        print("Unexpected state in half cycle 0.")
        return df
    if not alternation_ok:
        print("Invalid half cycle definitions.")
        return df
    # Adding a full cycle column
    # 1 full cycle is charge then discharge; code considers which the test begins with
    if cycle_state[1] == -1: # Cell starts in discharge
        df['full cycle'] = (df['half cycle']/2).apply(np.floor).astype(int)
    elif cycle_state[1] == 1: # Cell starts in charge
        df['full cycle'] = (df['half cycle']/2).apply(np.ceil).astype(int)
    else:
        print("Unexpected state in half cycle 1.")
        return df

    # Calculate Capacity (reset each half cycle)
    df["Capacity"] = df.groupby('half cycle', group_keys=False).apply(
        lambda g: (g['dQ'] * g['cycle state']).cumsum()
    ) # Uses cycle state instead of abs to deal with current transients

    # Add a column for power
    df['Power'] = df['Current']*df['Voltage']

    return df


# HELPER FUNCTIONS
# Deciding on charge, discharge, and rest based on sign of current
def state_from_current(x):
    try:
        if np.isnan(x):
            return 0 # Prevent issues with NaN current
        else:
            return int(np.sign(x)) # 1 if current is positive (charge), 0 if 0, -1 if negative (discharge)
    except:
        print(f"Unexpected value in current: {x}")
        raise ValueError('Unexpected value in current')


def halfcycles_from_cycle(df, cycle):
    # Determines which half cycles correspond to a given full cycle.
    # Relies on the fact that charge always precedes discharge in a given cycle.
    # Returns half cycle number of charge and discharge as a tuple
    try:
        mask = ((df['full cycle'] == cycle) & (df['half cycle'] > 0)) # Ignore half cycle 0, which is the initial rest
        hc = df.loc[mask]['half cycle'].unique()
        if len(hc) == 2:
            # If both a charge and discharge half cycle are present, charge comes first
            cha_hc = min(hc)
            dis_hc = max(hc)
        elif len(hc) == 1 and cycle == 0:
            # In this case, the test starts with discharge and cycle 0 is being requested
            # Therefore there is no charge half cycle
            cha_hc = None
            dis_hc = hc[0]
        elif len(hc) == 1:
            # If only one half cycle is found and it's not cycle 0
            # then only the charge portion of the cycle is present
            cha_hc = hc[0]
            dis_hc = None
        else:
            print(f"Invalid number of half cycles detected in cycle {cycle}.")
            return None, None
    except TypeError:
        print(f"Invalid cycle number: {cycle}.")
        return None, None
    return cha_hc, dis_hc


def cycle_from_halfcycle(df, halfcycle):
    # Function for determining which cycle corresponds to a given half cycle.
    try:
        mask = (df['half cycle'] == halfcycle)
        return df.loc[mask]['full cycle'].iloc[0]
    except TypeError:
        print(f"Invalid halfcycle number: {halfcycle}.")
        return None


def calculate_cycle_numbers(df):
    """
    Determine when a test is switching between charge and discharge, then calculate half cycle and full cycle numbers for each point in the df.
    """
    rest_mask = (df['state'] == 0)
    df['cycle change'] = False
    df['rest count'] = rest_mask.cumsum() # Helper column
    df['cycle change candidate'] = False # Helper column
    if (~rest_mask).any(): # If there is non-rest data
        # When state changes from 1 to -1 or -1 to 1, it is a candidate for a cycle change
        # But could also be caused by current transients
        # There should be rest between each charge and discharge half cycle, so check for an increment in number of rest points
        df.loc[~rest_mask, 'cycle change candidate'] = df.loc[~rest_mask, 'state'].ne(df.loc[~rest_mask, 'state'].shift())
        df.loc[df['cycle change candidate'], 'cycle change'] = df.loc[df['cycle change candidate'], 'rest count'].ne(df.loc[df['cycle change candidate'], 'rest count'].shift())
        df.drop(columns=['rest count', 'cycle change candidate'], inplace=True) # Clean up helper columns
    else:
        # If nothing is found for not_rest_idx, then all points are at rest
        # Therefore all the following values can be set to 0
        df['half cycle'] = 0
        df['full cycle'] = 0
        df['Q'] = 0
        df['Capacity'] = 0
        df['Power'] = 0
        print("No charge or discharge data found.")
        return df

    df['half cycle'] = (df['cycle change'] == True).cumsum() # Each time a cycle change occurs, increment half cycle
    # Adding a full cycle column
    # 1 full cycle is charge then discharge; code considers which the test begins with
    first_halfcycle = df.loc[df['half cycle'] == 1]
    initial_state = np.sign(first_halfcycle['dQ'].cumsum().iloc[-1] - first_halfcycle['dQ'].cumsum().iloc[0]) # Difference in Q between last and first point is charge passed
    # If charge passed is > 0, then test starts in charge, vice versa
    # Has to do the cumsum because Q has not been generated yet when calculate_cycle_numbers is called
    if initial_state == -1: # Cell starts in discharge
        df['full cycle'] = (df['half cycle']/2).apply(np.floor).astype(int)
    elif initial_state == 1: # Cell starts in charge
        df['full cycle'] = (df['half cycle']/2).apply(np.ceil).astype(int)
    else:
        print("Unexpected state in the first data point of half cycle 1.")
        return None
    return df
        

def df_post_process(df, mass=0, full_mass=0, area=0):
    """
    Adds columns to an imported dataframe for areal/specific versions of capacity, current, and power.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the imported data.
        mass (float): Mass (in mg) of starting cathode material
        full_mass (float): Mass (in mg) of fully discharged (e.g. lithiated) cathode
        area (float): Area (in cm^2) to normalize by for areal values

    Returns:
        pandas.DataFrame: The processed DataFrame with processed columns.
    """
    # Adding mass- and area-normalized columns if mass and area are provided
    # Ignore if defaults of 0.001 are present, since they result in incorrect and absurdly high values
    for label in ['Capacity', 'Q', 'Current', 'Power']:
        if label in df.columns:
            if mass > 0.001:
                df['Specific '+label] = 1000*df[label]/mass
            if full_mass > 0.001 and mass > 0.001:
                df['Specific '+label+' Total AM'] = 1000*df[label]/full_mass
            if area > 0.001:
                df['Areal '+label] = df[label]/area


    # # Adding mass- and area-normalized columns if mass and area are provided
    # # Ignore if defaults of 0.001 are present, since they result in absurdly high values
    # if mass > 0.001:
    #     df['Specific Capacity'] = 1000*df['Capacity']/mass
    #     df['Specific Q'] = 1000*df['Q']/mass
    #     df['Specific Current'] = 1000*df['Current']/mass
    #     df['Specific Power'] = 1000*df['Power']/mass
    # if full_mass > 0.001 and mass > 0.001:
    #     df['Specific Capacity Total AM'] = 1000*df['Capacity']/full_mass
    #     df['Specific Q Total AM'] = 1000*df['Q']/full_mass
    #     df['Specific Current Total AM'] = 1000*df['Current']/full_mass
    #     df['Specific Power Total AM'] = 1000*df['Power']/full_mass        
    # if area > 0.001:
    #     df['Areal Capacity'] = df['Capacity']/area
    #     df['Areal Q'] = df['Q']/area
    #     df['Areal Current'] = df['Current']/area
    #     df['Areal Power'] = df['Power']/area
    return df


# IMPORT FUNCTIONS
def arbin_res(df):
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the galvani res2sqlite for Arbin .res files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.
    """
    df.set_index('Data_Point', inplace=True)
    df.sort_index(inplace=True)

    df['state'] = df['Current'].map(lambda x: state_from_current(x))
    not_rest_idx = df[df['state'] != 0].index
    df['cycle change'] = False
    # If the state changes, then it's a half cycle change
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()

    # Calculating the capacity and changing to mAh
    if 'Discharge_Capacity' in df.columns:
        df['Capacity'] = df['Discharge_Capacity'] + df['Charge_Capacity']
    elif 'Discharge_Capacity(Ah)' in df.columns:
        df['Capacity'] = df['Discharge_Capacity(Ah)'] + df['Charge_Capacity(Ah)'] * 1000
    else:
        raise KeyError('Unable to find capacity columns, do not match Charge_Capacity or Charge_Capacity(Ah)')

    # Subtracting the initial capacity from each half cycle so it begins at zero
    for cycle in df['half cycle'].unique():
        idx = df[(df['half cycle'] == cycle) & (df['state'] != 0)].index
        if len(idx) > 0:
            cycle_idx = df[df['half cycle'] == cycle].index
            initial_capacity = df.loc[idx[0], 'Capacity']
            df.loc[cycle_idx, 'Capacity'] = df.loc[cycle_idx, 'Capacity'] - initial_capacity
        else:
            pass

    return df

def biologic_processing(raw):
    """
    Process the given DataFrame to generate a standardized output DataFrame. 
    Works for dataframes from the galvani MPRfile for Biologic .mpr files.

    Args:
        raw (pandas.DataFrame): The input DataFrame containing the raw data.

    Returns:
        pandas.DataFrame: The processed DataFrame with standardized columns: Time, dt, Voltage, Current, dQ, and state
        (See echem_file_loader for definitions and units of these columns)
    """
    df = pd.DataFrame()
    df['Time'] = raw['time/s']
    df['dt'] = np.diff(df['Time'], prepend=0)
    df['Voltage'] = raw['Ewe/V']

    # Determine incremental charge passed between each data point
    # +ve if charging, -ve if discharging
    # First see if raw data file contains dQ
    if ('dQ/mA.h' in raw.columns or 'dq/mA.h' in raw.columns):
        if 'dQ/mA.h' not in raw.columns:
            raw.rename(columns={'dq/mA.h': 'dQ/mA.h'}, inplace=True)
        df['dQ'] = raw['dQ/mA.h']
    # Else compute it from the integrated capacity Q-Qo
    elif ('(Q-Qo)/C' in raw.columns):
        df['dQ'] = np.diff(raw['(Q-Qo)/C'], prepend=0)
    # Otherwise dQ can be computed from current later
    if ('I/mA' in raw.columns or '<I>/mA' in raw.columns):
        if 'I/mA' not in raw.columns:
            raw.rename(columns={'<I>/mA': 'I/mA'}, inplace=True)
        df['Current'] = raw['I/mA']
        # Compute dQ from current if it wasn't present in the data file
        if 'dQ' not in df.columns:
            df['dQ'] = df['Current']*df['dt']
    elif 'dQ' in df.columns:
        # If current wasn't correctly exported, calculate it from dQ
        # Also convert infinities caused by dt of 0 to current of 0 to prevent issues down the line
        df['Current'] = (df['dQ']/df['dt']).replace([np.inf, -np.inf], 0)
    else:
        # If you've made it to this point, the raw data file didn't have dQ or current
        # There's not much you can do without this data
        print("Lacking necessary columns to determine current and capacity.")
        return None
    
    df['state'] = df['Current'].map(lambda x: state_from_current(x))
    
    return df

def bt_export_processing(raw):
    """
    Process the given DataFrame to generate a standardized output DataFrame. 
    Works for dataframes from the BT-Export .csv files.

    Args:
        raw (pandas.DataFrame): The input DataFrame containing the raw data.

    Returns:
        pandas.DataFrame: The processed DataFrame with standardized columns: Time, dt, Voltage, Current, dQ, and state
        (See echem_file_loader for definitions and units of these columns)
    """
    df = pd.DataFrame()
    df['Time'] = raw['Time / s']
    df['dt'] = np.diff(df['Time'], prepend=0)

    # Convert units in the current columns
    if('I / A' in raw.columns) and ('U / V' in raw.columns):
        df['Current'] = raw['I / A']*1000 # Convert into mA
        df['Voltage'] = raw['U / V']
        df['state'] = df['Current'].map(lambda x: state_from_current(x))
    else:
        print("Lacking necessary columns to determine state.")
        return None

    if 'Q / C' in raw.columns:
        df['dQ'] = np.diff(raw['Q / C']/3.6, prepend=0) # Convert from C to mAh
    else:
        df['dQ'] = df['Current']*df['dt']

    return df

def ivium_processing(df):
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Ivium .txt files.
    For Ivium files the cycler records the bare minimum (Current, Voltage, Time) and everything else is calculated from that.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.
    """

    df['dq'] = np.diff(df['time /s'], prepend=0)*df['I /mA']
    df['Capacity'] = df['dq'].cumsum()/3600

    df['state'] = df['I /mA'].map(lambda x: state_from_current(x))
    df['half cycle'] = df['state'].ne(df['state'].shift()).cumsum()
    for cycle in df['half cycle'].unique():
        mask = (df['half cycle'] == cycle)
        idx = df.index[mask]
        df.loc[idx, 'Capacity'] = abs(df.loc[idx, 'dq']).cumsum()/3600
    df['Voltage'] = df['E /V']
    df['Time'] = df['time /s']
    return df

def new_landt_processing(df):
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Landt .xlsx files.
    Landt has many different ways of exporting the data - so this is for one specific way of exporting the data.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.
    """
    # Remove half cycle == 0 for initial resting
    if 'Voltage/V' not in df.columns:
        column_to_search = df.columns[df.isin(['Index']).any()][0]
        df.columns = df[df[column_to_search] == 'Index'].iloc[0]
    df = df[df['Current/mA'].apply(type) != str]
    df = df[pd.notna(df['Current/mA'])]

    df['state'] = df['Current/mA'].map(lambda x: state_from_current(x))

    not_rest_idx = df[df['state'] != 0].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    df['Voltage'] = df['Voltage/V']
    df['Capacity'] = df['Capacity/mAh']
    df['Time'] = df['time /s']
    return df

def old_landt_processing(df):
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Landt .xlsx files.
    Landt has many different ways of exporting the data - so this is for one specific way of exporting the data.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.
    """
    df = df[df['Current/mA'].apply(type) != str]
    df = df[pd.notna(df['Current/mA'])]

    df['state'] = df['Current/mA'].map(lambda x: state_from_current(x))
    not_rest_idx = df[df['state'] != 0].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    df['Voltage'] = df['Voltage/V']
    df['Capacity'] = df['Capacity/mAh']
    return df

def arbin_excel(df):
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Arbin .xlsx files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.
    """

    df.reset_index(inplace=True)

    df['state'] = df['Current(A)'].map(lambda x: state_from_current(x))

    not_rest_idx = df[df['state'] != 0].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    # Calculating the capacity and changing to mAh
    df['Capacity'] = (df['Discharge_Capacity(Ah)'] + df['Charge_Capacity(Ah)']) * 1000

    for cycle in df['half cycle'].unique():
        idx = df[(df['half cycle'] == cycle) & (df['state'] != 0)].index  
        if len(idx) > 0:
            cycle_idx = df[df['half cycle'] == cycle].index
            initial_capacity = df.loc[idx[0], 'Capacity']
            df.loc[cycle_idx, 'Capacity'] = df.loc[cycle_idx, 'Capacity'] - initial_capacity
        else:
            pass

    df['Voltage'] = df['Voltage(V)']
    df['Current'] = df['Current(A)']
    if "Test_Time(s)" in df.columns:
        df["Time"] = df["Test_Time(s)"]

    return df

def neware_reader(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for neware .nda and .ndax files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.
    """
    from NewareNDA.NewareNDA import read
    filename = str(filename)
    df = read(filename)

    # remap to expected navani columns and units (mAh, V, mA) Our Neware machine reports mAh in column name but is in fact Ah...
    df.set_index('Index', inplace=True)
    df.index.rename('index', inplace=True)
    df['Capacity'] = 1000 * (df['Discharge_Capacity(mAh)'] + df['Charge_Capacity(mAh)'])
    df['Current'] = 1000 * df['Current(mA)']
    # Convert Neware state values into state column
    neware_state_dict = {'Rest': 0, 'CC_Chg': 1, 'CC_DChg': -1}
    df['state'] = df['Status'].map(lambda x: neware_state_dict[x])
    df['half cycle'] = df['Cycle']
    df['cycle change'] = False
    not_rest_idx = df[df['state'] != 0].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    return df


# Processing values by cycle number
def cycle_summary(df, mass=None, full_mass=None, area=None):
    """
    Computes summary statistics for each full cycle returning a new dataframe
    with the following columns:
    - 'Current': The average current for the cycle
    - 'UCV': The upper cut-off voltage for the cycle
    - 'LCV': The lower cut-off voltage for the cycle
    - 'Discharge Capacity': The maximum discharge capacity for the cycle
    - 'Charge Capacity': The maximum charge capacity for the cycle
    - 'CE': The charge efficiency for the cycle (Discharge Capacity/Charge Capacity)
    - 'Energy Efficiency': The energy efficiency for the cycle (Discharge Energy/Charge Energy)
    - 'Specific Discharge Capacity': The maximum specific discharge capacity for the cycle
    - 'Specific Charge Capacity': The maximum specific charge capacity for the cycle
    - 'Areal Discharge Capacity': The maximum specific discharge capacity for the cycle
    - 'Areal Charge Capacity': The maximum specific charge capacity for the cycle
    - 'Average Discharge Voltage': The average discharge voltage for the cycle
    - 'Average Charge Voltage': The average charge voltage for the cycle
    - 'Discharge Energy':  The integral energy on discharge for the cycle
    - 'Charge Energy': The integral energy of charge for the cycle
    - 'Discharge Overpotential': Overpotential as calculated by the relaxation after discharge
    - 'Charge Overpotential': Overpotential as calculated by the relaxation after charge
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        mass (float): Mass (in mg) of starting cathode material
        full_mass (float): Mass (in mg) of fully discharged (e.g. lithiated) cathode
        area (float): Area (in cm^2) to normalize by for areal values

    Returns:
        pandas.DataFrame: The summary DataFrame with the calculated values.
    """
    not_rest_mask = (df['state'] != 0)
    # Initialize empty dicts. cha_ and dis_records get added to summary_records which is then turned into the summary_df.
    cha_records = {}
    dis_records = {}
    summary_records = {}
    for cycle in df['full cycle'].unique():
        cha_hc, dis_hc = halfcycles_from_cycle(df, cycle)
        summary_records[cycle] = {} # Initialize empty entry to add keys later
        if cha_hc: # Skip if not present for this cycle
            mask = (df['half cycle'] == cha_hc)
            cha_df = df.loc[mask & not_rest_mask] # For averaged values, we want to exclude data points from rest
            if (cha_df['Time'].max() != cha_df['Time'].min()): # Check that non-zero time has elapsed in the halfcycle
                wts = cha_df['dt']
            else:
                wts = None # If no time has elapsed, this will unweight the time-based averages later to prevent a DivByZero error
            cha_cap = df.loc[mask]['Capacity'].max() # Charge capacity
            cha_energy = np.trapz(df.loc[mask]['Voltage'], df.loc[mask]['Capacity']) # Charge energy
            cha_current = np.average(cha_df['Current'], weights=wts) # Average charge current
            cha_power = np.average(cha_df['Power'], weights=wts) # Average charge power
            cha_voltage = np.average(cha_df['Voltage'], weights=wts) # Average charge voltage
            cha_UCV = cha_df['Voltage'].max() # Maximum voltage on charge is upper cutoff voltage
            cha_overpot = cha_UCV - df.loc[mask]['Voltage'].iloc[-1] # Charge overpotential
            # Subtract final data point (end of rest) from maximum voltage during charge

            cha_records[cycle] = {
                'Charge Capacity': cha_cap,
                'Charge Energy': cha_energy,
                'Average Charge Current': cha_current,
                'Average Charge Power': cha_power,
                'Average Charge Voltage': cha_voltage,
                'UCV': cha_UCV,
                'Charge Overpotential': cha_overpot
            }
            summary_records[cycle] |= cha_records[cycle] # Add entries to summary_records
        
        if dis_hc: # Skip if not present for this cycle
            mask = (df['half cycle'] == dis_hc)
            dis_df = df.loc[mask & not_rest_mask] # For averaged values, we want to exclude data points from rest
            if (dis_df['Time'].max() != dis_df['Time'].min()): # Check that non-zero time has elapsed in the halfcycle
                wts = dis_df['dt']
            else:
                wts = None # If no time has elapsed, this will unweight the time-based averages later to prevent a DivByZero error
            dis_cap = df.loc[mask]['Capacity'].max() # Discharge capacity
            dis_energy = np.trapz(df.loc[mask]['Voltage'], df.loc[mask]['Capacity']) # Discharge energy
            dis_current = np.average(dis_df['Current'], weights=wts) # Average discharge current
            dis_power = np.average(dis_df['Power'], weights=wts) # Average discharge power
            dis_voltage = np.average(dis_df['Voltage'], weights=wts) # Average discharge voltage
            dis_LCV = dis_df['Voltage'].min() # Minimum voltage on discharge is lower cutoff voltage
            dis_overpot =  df.loc[mask]['Voltage'].iloc[-1] - dis_LCV # Discharge overpotential
            # Subtract minimum voltage during discharge from final rest point

            dis_records[cycle] = {
                'Discharge Capacity': dis_cap,
                'Discharge Energy': dis_energy,
                'Average Discharge Current': dis_current,
                'Average Discharge Power': dis_power,
                'Average Discharge Voltage': dis_voltage,
                'LCV': dis_LCV,
                'Discharge Overpotential': dis_overpot
            }
            summary_records[cycle] |= dis_records[cycle] # Add entries to summary_records
    summary_df = pd.DataFrame.from_dict(summary_records, orient='index') # Convert the summary_records dict into a df
    summary_df.index.name = 'cycle'
    
    # Adding mass- and area-normalized columns if mass and area are provided
    # Ignore if defaults of 0.001 are present, since they result in incorrect and absurdly high values
    for label in ['Discharge Capacity', 'Discharge Energy', 'Average Discharge Current', 'Average Discharge Power', 
                  'Charge Capacity', 'Charge Energy', 'Average Charge Current', 'Average Charge Power']:
        if label in summary_df.columns:
            if mass > 0.001:
                summary_df['Specific '+label] = 1000*summary_df[label]/mass
            if full_mass > 0.001 and mass > 0.001:
                summary_df['Specific '+label+' Total AM'] = 1000*summary_df[label]/full_mass
            if area > 0.001:
                summary_df['Areal '+label] = summary_df[label]/area

    # Discharge/charge metrics
    if all(col in summary_df.columns for col in ['Discharge Capacity', 'Charge Capacity']):
        summary_df['CE'] = summary_df['Discharge Capacity']/summary_df['Charge Capacity']
    if all(col in summary_df.columns for col in ['Discharge Energy', 'Charge Energy']):
        summary_df['Energy Efficiency'] = summary_df['Discharge Energy']/summary_df['Charge Energy']

    return summary_df


def dqdv_single_cycle(capacity, voltage, 
                    polynomial_spline=3, s_spline=1e-5,
                    polyorder_1 = 5, window_size_1=101,
                    polyorder_2 = 5, window_size_2=1001,
                    final_smooth=True):
    """
    Calculate the derivative of capacity with respect to voltage (dq/dv) for a single cycle. Data is initially smoothed by a Savitzky-Golay filter and then interpolated and differentiated using a spline.
    Optionally the dq/dv curve can be smoothed again by another Savitzky-Golay filter.

    Args:
        capacity (array-like): Array of capacity values.
        voltage (array-like): Array of voltage values.
        polynomial_spline (int, optional): Order of the spline interpolation for the capacity-voltage curve. Defaults to 3. Best results use odd numbers.
        s_spline (float, optional): Smoothing factor for the spline interpolation. Defaults to 1e-5.
        polyorder_1 (int, optional): Order of the polynomial for the first smoothing filter (Before spline fitting). Defaults to 5. Best results use odd numbers.
        window_size_1 (int, optional): Size of the window for the first smoothing filter. (Before spline fitting). Defaults to 101. Must be odd.
        polyorder_2 (int, optional): Order of the polynomial for the second optional smoothing filter. Defaults to 5. (After spline fitting and differentiation). Best results use odd numbers.
        window_size_2 (int, optional): Size of the window for the second optional smoothing filter. Defaults to 1001. (After spline fitting and differentiation). Must be odd.
        final_smooth (bool, optional): Whether to apply final smoothing to the dq/dv curve. Defaults to True.

    Returns:
        tuple: A tuple containing three arrays: x_volt (array of voltage values), dqdv (array of dq/dv values), smooth_cap (array of smoothed capacity values).
    """
    
    import pandas as pd
    import numpy as np
    from scipy.interpolate import splrep, splev

    df = pd.DataFrame({'Capacity': capacity, 'Voltage': voltage})
    unique_v = df.astype(float).groupby('Voltage').mean().index
    unique_v_cap = df.astype(float).groupby('Voltage').mean()['Capacity']

    x_volt = np.linspace(min(voltage), max(voltage), num=int(1e4))
    f_lit = splrep(unique_v, unique_v_cap, k=1, s=0.0)
    y_cap = splev(x_volt, f_lit)
    smooth_cap = savgol_filter(y_cap, window_size_1, polyorder_1)

    f_smooth = splrep(x_volt, smooth_cap, k=polynomial_spline, s=s_spline)
    dqdv = splev(x_volt, f_smooth, der=1)
    smooth_dqdv = savgol_filter(dqdv, window_size_2, polyorder_2)
    if final_smooth:
        return x_volt, smooth_dqdv, smooth_cap
    else:
        return x_volt, dqdv, smooth_cap
    

# PLOTTING
# TODO move these to plot.py
def multi_cycle_plot(df, cycles, colormap='viridis', norm=None):
    """
    Function for plotting continuously coloured cycles (useful for large numbers).

    Parameters:
        df (DataFrame): The input DataFrame containing the data to be plotted.
        cycles (list or array-like): A list of full cycle numbers to be plotted.
        colormap (str, optional): The name of the colormap to be used for coloring the cycles. Default is 'viridis'.
        norm (str, optional): Normalization by 'area' or 'mass', if desired.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure object.
        ax (matplotlib.axes.Axes): The generated axes object.
    """

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import numpy as np

    # Determine type of capacity to plot, based on value of arg norm
    if norm is None:
        capacity_col = 'Capacity'
        capacity_label = 'Capacity / mAh'
    elif norm == 'area' and 'Areal Capacity' in df.columns:
        capacity_col = 'Areal Capacity'
        capacity_label = 'Areal Capacity / mAh/cm$^2$'
    elif norm == 'mass' and 'Specific Capacity' in df.columns:
        capacity_col = 'Specific Capacity'
        capacity_label = 'Specific Capacity / mAh/g'
    else:
        print("Invalid argument for norm or required column not present in df.")

    fig, ax = plt.subplots()
    cm = plt.get_cmap(colormap)
    norm = Normalize(vmin=int(min(cycles)), vmax=int(max(cycles)))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    for cycle in cycles:
        halfcycles = halfcycles_from_cycle(df, cycle)
        for halfcycle in halfcycles:
            mask = (df['half cycle'] == halfcycle)
            ax.plot(df.loc[mask][capacity_col], df.loc[mask]['Voltage'], color=cm(norm(cycle)))

    cbar = fig.colorbar(sm, ax=plt.gca())
    cbar.set_label('Cycle', rotation=270, labelpad=10)
    ax.set_ylabel('Voltage / V')
    ax.set_xlabel(capacity_label)
    return fig, ax

def multi_dqdv_plot(df, cycles, colormap='viridis', 
    capacity_label='Capacity', 
    voltage_label='Voltage',
    polynomial_spline=3, s_spline=1e-5,
    polyorder_1 = 5, window_size_1=101,
    polyorder_2 = 5, window_size_2=1001,
    final_smooth=True):
    """
    Plot multiple dQ/dV cycles on the same plot with a colormap.
    Uses the internal dqdv_single_cycle function to calculate the dQ/dV curves.

    Parameters:
        df: DataFrame containing the data.
        cycles: List or array-like object of cycle numbers to plot.
        colormap: Name of the colormap to use (default: 'viridis').
        capacity_label: Label of the capacity column in the DataFrame (default: 'Capacity').
        voltage_label: Label of the voltage column in the DataFrame (default: 'Voltage').
        polynomial_spline (int, optional): Order of the spline interpolation for the capacity-voltage curve. Defaults to 3. Best results use odd numbers.
        s_spline (float, optional): Smoothing factor for the spline interpolation. Defaults to 1e-5.
        polyorder_1 (int, optional): Order of the polynomial for the first smoothing filter (Before spline fitting). Defaults to 5. Best results use odd numbers.
        window_size_1 (int, optional): Size of the window for the first smoothing filter. (Before spline fitting). Defaults to 101. Must be odd.
        polyorder_2 (int, optional): Order of the polynomial for the second optional smoothing filter. Defaults to 5. (After spline fitting and differentiation). Best results use odd numbers.
        window_size_2 (int, optional): Size of the window for the second optional smoothing filter. Defaults to 1001. (After spline fitting and differentiation). Must be odd.
        final_smooth (bool, optional): Whether to apply final smoothing to the dq/dv curve. Defaults to True.

    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.

    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots() 
    cm = plt.get_cmap(colormap)
    norm = Normalize(vmin=int(min(cycles)), vmax=int(max(cycles)))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    for cycle in cycles:
        halfcycles = halfcycles_from_cycle(df, cycle)
        for halfcycle in halfcycles:
            df_cycle = df[df['half cycle'] == halfcycle]
            voltage, dqdv, _ = dqdv_single_cycle(df_cycle[capacity_label],
                                        df_cycle[voltage_label], 
                                        window_size_1=window_size_1,
                                        polyorder_1=polyorder_1,
                                        polynomial_spline=polynomial_spline,
                                        s_spline=s_spline,
                                        window_size_2=window_size_2,
                                        polyorder_2=polyorder_2,
                                        final_smooth=final_smooth)
            ax.plot(voltage, dqdv, color=cm(norm(cycle)))

    cbar = fig.colorbar(sm, ax=plt.gca())
    cbar.set_label('Cycle', rotation=270, labelpad=10)
    ax.set_xlabel('Voltage / V')
    ax.set_ylabel('dQ/dV / mAhV$^{-1}$')
    return fig, ax