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

# Different cyclers name their columns slightly differently 
# These dictionaries are guides for the main things you want to plot and what they are called
res_col_dict = {'Voltage': 'Voltage',
                'Capacity': 'Capacity'}

mpr_col_dict = {'Voltage': 'Ewe/V',
                'Capacity': 'Capacity'}

# Deciding on charge, discharge, and rest based on sign of current
def state_from_current(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        # print(x)
        # raise ValueError('Unexpected value in current - not a number')
        return 0

def echem_file_loader(filepath, 
                      df_to_append_to=None, 
                      time_offset=0.0,
                      calc_cycles_and_cap=True):
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

    # Stop here if the cycles and capacity don't need to be calculated yet
    if not calc_cycles_and_cap:
        return df

    # Sort by time before calculating cycle numbers, etc.
    df.sort_values(by='Time', inplace=True)

    df['cycle change'] = False
    not_rest_idx = df[df['state'] != 0].index
    if len(not_rest_idx) > 0:
        df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    else:
        # If nothing is found for not_rest_idx, then all points are at rest
        # Therefore all the following values can be set to 0
        df['cycle change'] = 0
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
    # Find the first state that is not rest
    initial_state = df.loc[not_rest_idx, 'state'].iloc[0]
    if initial_state == -1: # Cell starts in discharge
        df['full cycle'] = (df['half cycle']/2).apply(np.floor)
    elif initial_state == 1: # Cell starts in charge
        df['full cycle'] = (df['half cycle']/2).apply(np.ceil)
    else:
        print("Unexpected state in the first data point of half cycle 1.")
        return None

    # Calculate Q (running total capacity) and Capacity (capacity, reset each half cycle)
    df['Q'] = df['dQ'].cumsum()
    for cycle in df['half cycle'].unique():
        mask = (df['half cycle'] == cycle)
        cycle_idx = df[mask].index
        df.loc[cycle_idx, 'Capacity'] = df.loc[cycle_idx, 'dQ'].abs().cumsum()

    # Add a column for power
    df['Power'] = df['Current']*df['Voltage']

    return df

def standardize_time_label(df):
    # Makes it easier to concatenate tests of different types.
    time_labels = ['time/s', 'time /s', 'Time / s']
    for label in time_labels:
        if label in df.columns:
            df.rename(columns={label: 'Time'}, inplace=True)
            break # Only rename the first match
    if 'Time' in df.columns:
        df['dt'] = np.diff(df['Time'], prepend=0)
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
    # Ignore if defaults of 0.001 are present, since they result in absurdly high values
    if mass > 0.001:
        df['Specific Capacity'] = 1000*df['Capacity']/mass
        df['Specific Q'] = 1000*df['Q']/mass
        df['Specific Current'] = 1000*df['Current']/mass
        df['Specific Power'] = 1000*df['Power']/mass
    if full_mass > 0.001 and mass > 0.001:
        df['Specific Capacity Total AM'] = 1000*df['Capacity']/full_mass
        df['Specific Q Total AM'] = 1000*df['Q']/full_mass
        df['Specific Current Total AM'] = 1000*df['Current']/full_mass
        df['Specific Power Total AM'] = 1000*df['Power']/full_mass        
    if area > 0.001:
        df['Areal Capacity'] = df['Capacity']/area
        df['Areal Q'] = df['Q']/area
        df['Areal Current'] = df['Current']/area
        df['Areal Power'] = df['Power']/area

    return df

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
        df['dQ'] = np.diff(raw['Q / C'], prepend=0)
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
        mask = df['half cycle'] == cycle
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

"""
Processing values by cycle number
"""
def cycle_summary(df, current_label=None, mass=None, full_mass=None, area=None):
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
        current_label (str, optional): The label of the current column. Defaults to None and compares to a list of known current labels.
        mass (float): Mass (in mg) of starting cathode material
        full_mass (float): Mass (in mg) of fully discharged (e.g. lithiated) cathode
        area (float): Area (in cm^2) to normalize by for areal values

    Returns:
        pandas.DataFrame: The summary DataFrame with the calculated values.
    """
    current_labels = ['Current', 'Current(A)', 'I /mA', 'Current/mA', 'I/mA', '<I>/mA']
    # Figuring out which column is current
    if current_label is not None:
        df[current_label] = df[current_label].astype(float)
        summary_df = df.groupby('full cycle')[current_label].mean().to_frame()
    else:
        intersection = set(current_labels) & set(df.columns)
        if len(intersection) > 0:
            # Choose the first available label from current labels
            for label in current_labels:
                if label in intersection:
                    current_label = label
                    break
            df[current_label] = df[current_label].astype(float)
            summary_df = df.groupby('full cycle')[current_label].mean().to_frame()
        else:
            print('Could not find Current column label. Please supply label to function: current_label=label')
            summary_df = pd.DataFrame(index=df['full cycle'].unique())

    summary_df['UCV'] = df.groupby('full cycle')['Voltage'].max()
    summary_df['LCV'] = df.groupby('full cycle')['Voltage'].min()

    dis_mask = df['state'] == -1
    dis_index = df[dis_mask]['full cycle'].unique()
    if len(dis_index) > 0:
        summary_df.loc[dis_index, 'Discharge Capacity'] = df[dis_mask].groupby('full cycle')['Capacity'].max()
        dis_cycles = df.loc[df.index[dis_mask]]['half cycle'].unique()
        for halfcycle in dis_cycles:
            mask = df['half cycle'] == halfcycle
            cycle = df['full cycle'][mask].iloc[0] # Full cycle corresponding with this half cycle
            energy = np.trapz(df[mask]['Voltage'], df[mask]['Capacity'])
            # Add an entry to the summary for each full cycle
            summary_df.loc[cycle, 'Discharge Energy'] = energy
            
            # Amount of relaxation at end of discharge
            # Only accurate for tests with a single rest in the discharge halfcycle
            summary_df.loc[cycle, 'Discharge Overpotential'] = df.loc[mask & (df['state'] == 0)]['Voltage'].max() - df.loc[mask & (df['state'] == -1)]['Voltage'].min()
            
            # Time-weighted average of current/power only among the points in the given half cycle where the cell is not resting
            summary_df.loc[cycle, 'Average Discharge Current'] = np.average(df.loc[mask & dis_mask]['Current'], weights=df.loc[mask & dis_mask]['dt'])
            summary_df.loc[cycle, 'Average Discharge Power'] = np.average(df.loc[mask & dis_mask]['Power'], weights=df.loc[mask & dis_mask]['dt'])
        summary_df['Average Discharge Voltage'] = summary_df['Discharge Energy']/summary_df['Discharge Capacity']
        # Normalized metrics
        if mass > 0:
            summary_df['Specific Discharge Capacity'] = 1000*summary_df['Discharge Capacity']/mass
            summary_df['Specific Discharge Energy'] = 1000*summary_df['Discharge Energy']/mass
            summary_df['Specific Average Discharge Current'] = 1000*summary_df['Average Discharge Current']/mass
            summary_df['Specific Average Discharge Power'] = 1000*summary_df['Average Discharge Power']/mass
        if full_mass > 0:
            summary_df['Specific Discharge Capacity Total AM'] = 1000*summary_df['Discharge Capacity']/full_mass
            summary_df['Specific Discharge Energy Total AM'] = 1000*summary_df['Discharge Energy']/full_mass
            summary_df['Specific Average Discharge Current Total AM'] = 1000*summary_df['Average Discharge Current']/full_mass
            summary_df['Specific Average Discharge Power Total AM'] = 1000*summary_df['Average Discharge Power']/full_mass
        if area > 0:
            summary_df['Areal Discharge Capacity'] = summary_df['Discharge Capacity']/area
            summary_df['Areal Discharge Energy'] = summary_df['Discharge Energy']/area
            summary_df['Areal Average Discharge Current'] = summary_df['Average Discharge Current']/area
            summary_df['Areal Average Discharge Power'] = summary_df['Average Discharge Power']/area

    cha_mask = df['state'] == 1
    cha_index = df[cha_mask]['full cycle'].unique()
    if len(cha_index) > 0:
        summary_df.loc[cha_index, 'Charge Capacity'] = df[cha_mask].groupby('full cycle')['Capacity'].max()
        cha_cycles = df.loc[df.index[cha_mask]]['half cycle'].unique()
        for halfcycle in cha_cycles:
            mask = df['half cycle'] == halfcycle
            cycle = df['full cycle'][mask].iloc[0] # Full cycle corresponding with this half cycle
            energy = np.trapz(df[mask]['Voltage'], df[mask]['Capacity'])
            # Add an entry to the summary for each full cycle
            summary_df.loc[cycle, 'Charge Energy'] = energy
            
            # Amount of relaxation at end of charge
            # Only accurate for tests with a single rest at the end of charge
            summary_df.loc[cycle, 'Charge Overpotential'] = df.loc[mask & (df['state'] == 1)]['Voltage'].max() - df.loc[mask & (df['state'] == 0)]['Voltage'].min()

            # Time-weighted average of current only among the points in the given half cycle where the cell is not resting
            summary_df.loc[cycle, 'Average Charge Current'] = np.average(df.loc[mask & cha_mask]['Current'], weights=df.loc[mask & cha_mask]['dt'])
            summary_df.loc[cycle, 'Average Charge Power'] = np.average(df.loc[mask & cha_mask]['Power'], weights=df.loc[mask & cha_mask]['dt'])
        summary_df['Average Charge Voltage'] = summary_df['Charge Energy']/summary_df['Charge Capacity']
        # Discharge/charge metrics
        summary_df['CE'] = summary_df['Discharge Capacity']/summary_df['Charge Capacity']
        summary_df['Energy Efficiency'] = summary_df['Discharge Energy']/summary_df['Charge Energy']
        # Normalized metrics
        if mass > 0:
            summary_df['Specific Charge Capacity'] = 1000*summary_df['Charge Capacity']/mass
            summary_df['Specific Charge Energy'] = 1000*summary_df['Charge Energy']/mass
            summary_df['Specific Average Charge Current'] = 1000*summary_df['Average Charge Current']/mass
            summary_df['Specific Average Charge Power'] = 1000*summary_df['Average Charge Power']/mass    
        if full_mass > 0:
            summary_df['Specific Charge Capacity Total AM'] = 1000*summary_df['Charge Capacity']/full_mass
            summary_df['Specific Charge Energy Total AM'] = 1000*summary_df['Charge Energy']/full_mass
            summary_df['Specific Average Charge Current Total AM'] = 1000*summary_df['Average Charge Current']/full_mass
            summary_df['Specific Average Charge Power Total AM'] = 1000*summary_df['Average Charge Power']/full_mass
        if area > 0:
            summary_df['Areal Charge Capacity'] = summary_df['Charge Capacity']/area
            summary_df['Areal Charge Energy'] = summary_df['Charge Energy']/area
            summary_df['Areal Average Charge Current'] = summary_df['Average Charge Current']/area
            summary_df['Areal Average Charge Power'] = summary_df['Average Charge Power']/area

    return summary_df

def halfcycles_from_cycle(df, cycle):
    # Determines which half cycles correspond to a given full cycle.
    try:
        mask = df['full cycle'] == cycle
        hc = list(df['half cycle'][mask].unique())
        # The 0th half cycle is rest at the beginning of the test
        # and should not be returned for the 0th full cycle
        if 0 in hc:
            hc.remove(0)
        return hc
    except TypeError:
        print("Invalid type for cycle number.")

def cycle_from_halfcycle(df, halfcycle):
    # Function for determining which cycle corresponds to a given half cycle.
    try:
        mask = df['half cycle'] == halfcycle
        return df['full cycle'][mask].iloc[0]
    except TypeError:
        print("Invalid type for halfcycle number.")

"""
PLOTTING
"""

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
            mask = df['half cycle'] == halfcycle
            ax.plot(df[capacity_col][mask], df['Voltage'][mask], color=cm(norm(cycle)))

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