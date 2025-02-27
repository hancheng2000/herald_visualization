from galvani import MPRfile
from galvani import res2sqlite as r2s

import pandas as pd
import numpy as np
import warnings
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

current_labels = ['Current', 'Current(A)', 'I /mA', 'Current/mA', 'I/mA', '<I>/mA']

# Deciding on charge, discharge, and rest based on current direction
def state_from_current(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        print(x)
        raise ValueError('Unexpected value in current - not a number')

def multi_file_biologic(filepath, time_offset=0, capacity_offset=0):
    """
    
    """
    gal_file = MPRfile(os.path.join(filepath))
    df = pd.DataFrame(data=gal_file.data)

    # Offset time and/or capacity to make stitched files continuous
    if time_offset != 0 or capacity_offset != 0:
        df['time/s'] += time_offset
        df['Q charge/discharge/mA.h'] += capacity_offset

    df = biologic_processing(df)

    return df

def echem_file_loader(filepath):
    """
    Loads a variety of electrochemical filetypes and tries to construct the most useful measurements in a
    consistent way, with consistent column labels. Outputs a dataframe with the original columns, and these constructed columns:
    
    - "state": 0 for rest, -1 for discharge, 1 for charge (defined by the current direction +ve or -ve)
    - "half cycle": Counts the half cycles, rests are not included as a half cycle
    - "full cycle": Counts the full cycles, rests are not included as a full cycle
    - "cycle change": Boolean column that is True when the state changes
    - "Capacity": The capacity of the cell, each half cycle it resets to 0 - In general this will be in mAh - however it depends what unit the original file is in - Arbin Ah automatically converted to mAh
    - "Voltage": The voltage of the cell
    - "Current": The current of the cell - In general this will be in mA - however it depends what unit the original file is in
    
    From these measurements, everything you want to know about the electrochemistry can be calculated.
    
    Parameters:
        filepath (str): The path to the electrochemical file.
    
    Returns:
        pandas.DataFrame: A dataframe with the original columns and the constructed columns.
    """
    extension = os.path.splitext(filepath)[-1].lower()
    # Biologic file
    if extension == '.mpr':
        gal_file = MPRfile(os.path.join(filepath))
        df = pd.DataFrame(data=gal_file.data)
        df = biologic_processing(df)

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
        df = pd.read_csv(os.path.join(filepath), sep='\t')
        # Checking columns are an exact match
        if set(['time /s', 'I /mA', 'E /V']) - set(df.columns) == set([]):
            df = ivium_processing(df)
        else:
            raise ValueError('Columns do not match expected columns for an ivium .txt file')

    # Landdt and Arbin can output .xlsx and .xls files
    elif extension in ['.xlsx', '.xls']:
        if extension == '.xlsx':
            xlsx = pd.ExcelFile(os.path.join(filepath), engine='openpyxl')
        else:
            xlsx = pd.ExcelFile(os.path.join(filepath))

        names = xlsx.sheet_names
        # Use different land processing if all exported as one sheet (different versions of landdt software)
        if len(names) == 1:
            df = xlsx.parse(0)
            df = new_land_processing(df)

        # If Record is a sheet name, then it is a landdt file
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
            df = pd.concat(df_list)
            df.set_index('Index', inplace=True)
            df = old_land_processing(df)

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
                df = pd.concat(df_list)
                df = arbin_excel(df)
            else:
                raise ValueError('Names of sheets not recognised')
            
    # Neware files are .nda or .ndax
    elif extension in (".nda", ".ndax"):
        df = neware_reader(filepath)

    # If the file is a csv previously processed by navani
    # Check for the columns that are expected (Capacity, Voltage, Current, Cycle numbers, state)
    elif extension == '.csv':
        df = pd.read_csv(filepath, 
                         index_col=0,
                         dtype={'Time': np.float64, 'Capacity': np.float64, 'Voltage': np.float64, 'Current': np.float64,
                                'full cycle': np.int32, 'half cycle': np.int32, 'state': np.int16})
        expected_columns = ['Capacity', 'Voltage', 'half cycle', 'full cycle', 'Current', 'state']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError('Columns do not match expected columns for navani processed csv')
        
    # If it's a filetype not seen before raise an error
    else:
        print(extension)
        raise RuntimeError("Filetype {extension} not recognised.")

    df_post_process(df)

    return df

def df_post_process(df, mass=0, full_mass=0, area=0):
    """
    Adds columns to an imported dataframe for full cycles, power, and areal/specific versions of capacity, current, and power.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the imported data.
        mass (float): Mass (in mg) of starting cathode material
        full_mass (float): Mass (in mg) of fully discharged (e.g. lithiated) cathode
        area (float): Area (in cm^2) to normalize by for areal values

    Returns:
        pandas.DataFrame: The processed DataFrame with processed columns.
    """
    # Adding a full cycle column
    # 1 full cycle is charge then discharge; code considers which the test begins with
    if 'half cycle' in df.columns:
        # Find the 'state' of the first data point in half cycle 1
        initial_state = df[df['half cycle'] == 1].iloc[0].state
        
        if initial_state == -1: # Cell starts in discharge
            df['full cycle'] = (df['half cycle']/2).apply(np.floor)
        elif initial_state == 1: # Cell starts in charge
            df['full cycle'] = (df['half cycle']/2).apply(np.ceil)
        else:
            raise Exception("Unexpected state in the first data point of half cycle 1.")

    # Adding a power column
    if 'Current' and 'Voltage' in df.columns:
        df['Power'] = df['Current']*df['Voltage']

    # Adding mass- and area-normalized columns if mass and area are provided
    if mass > 0:
        df['Specific Capacity'] = 1000*df['Capacity']/mass
        df['Specific Current'] = 1000*df['Current']/mass
        df['Specific Power'] = 1000*df['Power']/mass
    if full_mass > 0:
        df['Specific Capacity Total AM'] = 1000*df['Capacity']/full_mass
        df['Specific Current Total AM'] = 1000*df['Current']/full_mass
        df['Specific Power Total AM'] = 1000*df['Power']/full_mass        
    if area > 0:
        df['Areal Capacity'] = df['Capacity']/area
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

def biologic_processing(df):
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the galvani MPRfile for Biologic .mpr files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.
    """

    if 'time/s' in df.columns:
        df['Time'] = df['time/s']

    # If current has been correctly exported then we can use that
    if('I/mA' in df.columns) and ('Q charge/discharge/mA.h' not in df.columns) and ('dQ/mA.h' not in df.columns) and ('Ewe/V' in df.columns):
        df['Current'] = df['I/mA']
        df['dV'] = np.diff(df['Ewe/V'], prepend=df['Ewe/V'][0])
        df['state'] = df['dV'].map(lambda x: state_from_current(x))
        # TODO why the fuck does this use dV instead of current? Fix this!

    elif('<I>/mA' in df.columns) and ('Q charge/discharge/mA.h' not in df.columns) and ('dQ/mA.h' not in df.columns) and ('Ewe/V' in df.columns):
        df['Current'] = df['<I>/mA']
        df['dV'] = np.diff(df['Ewe/V'], prepend=df['Ewe/V'][0])
        df['state'] = df['dV'].map(lambda x: state_from_current(x))
        # TODO ibid

    # Otherwise, add the current column that galvani can't (sometimes) export for some reason
    elif ('time/s' in df.columns) and ('dQ/mA.h' in df.columns or 'dq/mA.h' in df.columns):
        df['dt'] = np.diff(df['time/s'], prepend=0)
        if 'dQ/mA.h' not in df.columns:
            df.rename(columns={'dq/mA.h': 'dQ/mA.h'}, inplace=True)
        df['Current'] = df['dQ/mA.h']/(df['dt']/3600)

        if np.isnan(df['Current'].iloc[0]):
            df.loc[df.index[0], 'Current'] = 0

        df['state'] = df['Current'].map(lambda x: state_from_current(x))

    elif ('time/s' in df.columns) and ('Q charge/discharge/mA.h' in df.columns):
        df['dQ/mA.h'] = np.diff(df['Q charge/discharge/mA.h'], prepend=0)
        df['dt'] = np.diff(df['time/s'], prepend=0)
        df['Current'] = df['dQ/mA.h']/(df['dt']/3600)

        if np.isnan(df['Current'].iloc[0]):
            df.loc[df.index[0], 'Current'] = 0

        df['state'] = df['Current'].map(lambda x: state_from_current(x))

    df['cycle change'] = False
    if 'state' in df.columns:
        not_rest_idx = df[df['state'] != 0].index
        df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())

    df['half cycle'] = (df['cycle change'] == True).cumsum()

    if ('Q charge/discharge/mA.h' in df.columns) and ('half cycle') in df.columns:
        df['Capacity'] = abs(df['Q charge/discharge/mA.h'])
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df

    elif ('dQ/mA.h' in df.columns) and ('half cycle') in df.columns:
        df['Half cycle cap'] = abs(df['dQ/mA.h'])
        for cycle in df['half cycle'].unique():
            mask = df['half cycle'] == cycle
            cycle_idx = df.index[mask]
            df.loc[cycle_idx, 'Half cycle cap'] = df.loc[cycle_idx, 'Half cycle cap'].cumsum()
        df.rename(columns = {'Half cycle cap':'Capacity'}, inplace = True)
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df
    elif ('(Q-Qo)/C' in df.columns) and ('half cycle') in df.columns:
        for cycle in df['half cycle'].unique():
            mask = df['half cycle'] == cycle
            cycle_idx = df.index[mask]
            df.loc[cycle_idx, 'Capacity'] = df.loc[cycle_idx, '(Q-Qo)/C'] - df.loc[cycle_idx[0], '(Q-Qo)/C']
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df
    else:
        print('Warning: unhandled column layout. No capacity or charge columns found.')
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
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

def new_land_processing(df):
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Landdt .xlsx files.
    Landdt has many different ways of exporting the data - so this is for one specific way of exporting the data.

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

def old_land_processing(df):
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Landdt .xlsx files.
    Landdt has many different ways of exporting the data - so this is for one specific way of exporting the data.

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

    df = pd.DataFrame({'Capacity': capacity, 'Voltage':voltage})
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
    - 'Specific Discharge Capacity': The maximum specific discharge capacity for the cycle
    - 'Specific Charge Capacity': The maximum specific charge capacity for the cycle
    - 'Areal Discharge Capacity': The maximum specific discharge capacity for the cycle
    - 'Areal Charge Capacity': The maximum specific charge capacity for the cycle
    - 'Average Discharge Voltage': The average discharge voltage for the cycle
    - 'Average Charge Voltage': The average charge voltage for the cycle
    - 'Discharge Energy': The integral energy on discharge for the cycle
    - 'Charge Energy': The integral energy of charge for the cycle
    - 
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        current_label (str, optional): The label of the current column. Defaults to None and compares to a list of known current labels.
        mass (float): Mass (in mg) of starting cathode material
        full_mass (float): Mass (in mg) of fully discharged (e.g. lithiated) cathode
        area (float): Area (in cm^2) to normalize by for areal values

    Returns:
        pandas.DataFrame: The summary DataFrame with the calculated values.
    """
    
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
        if mass > 0:
            summary_df['Specific Discharge Capacity'] = 1000*summary_df['Discharge Capacity']/mass
            summary_df['Specific Discharge Energy'] = 1000*summary_df['Discharge Energy']/mass
        if full_mass > 0:
            summary_df['Specific Discharge Capacity Total AM'] = 1000*summary_df['Discharge Capacity']/full_mass
            summary_df['Specific Discharge Energy Total AM'] = 1000*summary_df['Discharge Energy']/full_mass
        if area > 0:
            summary_df['Areal Discharge Capacity'] = summary_df['Discharge Capacity']/area
            summary_df['Areal Discharge Energy'] = summary_df['Discharge Energy']/area

    cha_mask = df['state'] == 1
    cha_index = df[cha_mask]['full cycle'].unique()
    if len(cha_index) > 0:
        summary_df.loc[cha_index, 'Charge Capacity'] = df[cha_mask].groupby('full cycle')['Capacity'].max()
        summary_df['CE'] = summary_df['Discharge Capacity']/summary_df['Charge Capacity']
        cha_cycles = df.loc[df.index[cha_mask]]['half cycle'].unique()
        for halfcycle in cha_cycles:
            mask = df['half cycle'] == halfcycle
            cycle = df['full cycle'][mask].iloc[0] # Full cycle corresponding with this half cycle
            energy = np.trapz(df[mask]['Voltage'], df[mask]['Capacity'])
            # Add an entry to the summary for each full cycle
            summary_df.loc[cycle, 'Charge Energy'] = energy
        if mass > 0:
            summary_df['Specific Charge Capacity'] = 1000*summary_df['Charge Capacity']/mass
            summary_df['Specific Charge Energy'] = 1000*summary_df['Charge Energy']/mass
        if full_mass > 0:
            summary_df['Specific Charge Capacity Total AM'] = 1000*summary_df['Charge Capacity']/full_mass
            summary_df['Specific Charge Energy Total AM'] = 1000*summary_df['Charge Energy']/full_mass
        if area > 0:
            summary_df['Areal Charge Capacity'] = summary_df['Charge Capacity']/area
            summary_df['Areal Charge Energy'] = summary_df['Charge Energy']/area
    
    if 'Discharge Energy' in summary_df.columns and 'Discharge Capacity' in summary_df.columns:
        summary_df['Average Discharge Voltage'] = summary_df['Discharge Energy']/summary_df['Discharge Capacity']
    if 'Charge Energy' in summary_df.columns and 'Charge Capacity' in summary_df.columns:
        summary_df['Average Charge Voltage'] = summary_df['Charge Energy']/summary_df['Charge Capacity']

    return summary_df

def halfcycles_from_cycle(df, cycle):
    # Determines which half cycles correspond to a given full cycle.
    try:
        mask = df['full cycle'] == cycle
        return list(df['half cycle'][mask].unique())
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

def charge_discharge_plot(df, cycles, colormap=None, norm=None, fig=None, ax=None, plot_kwargs={}):
    """
    Function for plotting individual or multi but discrete charge discharge cycles

    Args:
        df (DataFrame): The input dataframe containing the data for plotting.
        full_cycle (int or list of ints): The full cycle number(s) to plot. If an integer is provided, a single cycle will be plotted (charge and discharge). If a list is provided, multiple cycles will be plotted.
        colormap (str, optional): The colormap to use for coloring the cycles. If not provided, a default colormap will be used based on the number of cycles.
        norm (str, optional): Normalization by 'mass', 'full_mass', or 'area', if desired. 'full_mass' refers to the mass of the cathode AM material and stoichiometric anode material, i.e. fully lithiated cathode.

    Returns:
        fig (Figure): The matplotlib Figure object.
        ax (Axes): The matplotlib Axes object.

    Raises:
        ValueError: If there are too many cycles for the default colormaps. (20)

    """
    if fig == None and ax == None:
        fig, ax = plt.subplots()

    # Determine type of capacity to plot, based on value of arg norm
    if norm is None:
        capacity_col = 'Capacity'
        capacity_label = 'Capacity / mAh'
    elif norm == 'mass' and 'Specific Capacity' in df.columns:
        capacity_col = 'Specific Capacity'
        capacity_label = 'Specific Capacity / mAh/g cathode'
    elif norm == 'full_mass' and 'Specific Capacity Total AM' in df.columns:
        capacity_col = 'Specific Capacity Total AM'
        capacity_label = 'Specific Capacity / mAh/g AM'
    elif norm == 'area' and 'Areal Capacity' in df.columns:
        capacity_col = 'Areal Capacity'
        capacity_label = 'Areal Capacity / mAh/cm$^2$'
    else:
        print("Invalid argument for norm or required column not present in df.")

    try:
        iter(cycles)

    except TypeError: # Happens if full_cycle is an int
        halfcycles = halfcycles_from_cycle(df, cycles)
        for halfcycle in halfcycles:
            mask = df['half cycle'] == halfcycle
            df1 = df[mask]
            df1 = df1[df1['Specific Capacity']!=0]
            # Making sure half cycle exists within the data
            if sum(mask) > 0:
                ax.plot(df1[capacity_col], df1['Voltage'], **plot_kwargs)

        ax.set_xlabel(capacity_label)
        ax.set_ylabel('Voltage / V')
        return fig, ax

    if not colormap:
        if len(cycles) < 11:
            colormap = 'tab10'
        elif len(cycles) < 21:
            colormap = 'tab20'
        else:
            raise ValueError("Too many cycles for default colormaps. Use multi_cycle_plot instead")

    cm = plt.get_cmap(colormap)
    for count, cycle in enumerate(cycles):
        halfcycles = halfcycles_from_cycle(df, cycle)
        for halfcycle in halfcycles:
            mask = df['half cycle'] == halfcycle
            df1 = df[mask]
            df1 = df1[df1['Specific Capacity']!=0]
            # Making sure half cycle exists within the data
            if sum(mask) > 0:
                ax.plot(df1[capacity_col], df1['Voltage'], color=cm(count), **plot_kwargs)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=cm(count), lw=2) for count, _ in enumerate(cycles)]

    ax.legend(custom_lines, [f'Cycle {i}' for i in cycles])
    ax.set_xlabel(capacity_label)
    ax.set_ylabel('Voltage / V')
    return fig, ax

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