from galvani import MPRfile
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import simpson as simp
import os
from typing import Union
from pathlib import Path

# MAIN FUNCTION
# TODO: correct output for non-BioLogic files
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
    
    df['Current'] = df['Current'].fillna(0.0) # Replace nan values to prevent later problems
    if df.at[0, 'state'] != 0: # If df does not begin with rest, prepend a rest point
        placeholder_voltage = df['Voltage'].loc[0]
        placeholder = pd.DataFrame({'Time':[0], 'dt':[0], 'Current':[0], 'Voltage':[placeholder_voltage], 'state':[0], 'dQ':[0]})
        df = pd.concat([placeholder, df], ignore_index=True)
        df.index.name = 'id' # Restore index name after concat

    # Sort by time before calculating cycle numbers, etc.
    # In case multiple points have the same time value, sort by id
    df.sort_values(by=['Time','id'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate Q (running total capacity) to use as a check for half cycle alternation
    df['Q'] = df['dQ'].cumsum()

    # Calculate cycle numbers
    rest_mask = (df['state'] == 0)
    df['cycle change candidate'] = False # Helper column
    if (~rest_mask).any(): # If there is non-rest data
        # When state changes from rest to charge or discharge, this is the start of a new block
        df['cycle change candidate'] = (~rest_mask) & rest_mask.shift(fill_value=True)
        df['block'] = df['cycle change candidate'].cumsum()
        block_deltaQ = df.groupby('block')['Q'].agg(lambda x: x.iloc[-1] - x.iloc[0])
        block_avg_current = df.groupby('block')['Current'].mean()
        block_state = np.sign(block_deltaQ).fillna(0).astype(int)
        block_state[block_state == 0] = np.sign(block_avg_current[block_state == 0]).fillna(0).astype(int) # If deltaQ is 0, use current within block instead
        half_cycle_id = (block_state != block_state.shift(fill_value=block_state.iloc[0])).cumsum() # Increment half cycle when consecutive blocks have different states
        # e.g. charge followed by discharge
        # This will keep GITT pulses grouped into the same half cycle, correctly
        # Fill value is used to make sure the initial rest is numbered half cycle 0
        df['half cycle'] = df['block'].map(half_cycle_id) # Assign half cycles to correct column
        halfcycle_deltaQ = df.groupby('half cycle')['Q'].agg(lambda x: x.iloc[-1] - x.iloc[0])
        halfcycle_avg_current = df.groupby('half cycle')['Current'].mean()
        cycle_state = np.sign(halfcycle_deltaQ).fillna(0).astype(int)
        cycle_state[cycle_state == 0] = np.sign(halfcycle_avg_current[cycle_state == 0]).fillna(0).astype(int)
        df['cycle state'] = df['half cycle'].map(cycle_state)
        df.drop(columns=['block', 'cycle change candidate'], inplace=True) # Clean up helper columns
        initial_state = half_cycle_id[1] # State of first non-rest half cycle determines full cycle behavior
        # Adding a full cycle column
        # 1 full cycle is charge then discharge; code considers which the test begins with
        if initial_state == -1: # Cell starts in discharge
            df['full cycle'] = (df['half cycle']/2).apply(np.floor).astype(int)
        elif initial_state == 1: # Cell starts in charge
            df['full cycle'] = (df['half cycle']/2).apply(np.ceil).astype(int)
        else:
            print("Invalid state in half cycle 1.")
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


# Calculate whether a block (e.g. halfcycle) is overall discharge, charge, or rest
def compute_block_state(group):
    deltaQ = group['Q'].iloc[-1] - group['Q'].iloc[0] # Difference in Q from beginning to end of block
    s = np.sign(deltaQ)
    if np.isnan(s):
        s = 0
    if s == 0: # If no deltaQ recorded, fall back to using current
        s = np.sign(group['Current'].mean())
    return int(s)


def halfcycles_from_cycle(df, cycle):
    # Determines which half cycles correspond to a given full cycle.
    # Returns half cycle number of charge and discharge as a tuple
    try:
        mask = (df['full cycle'] == cycle)
    except:
        print(f"Invalid value for cycle: {cycle}.")
        return None, None
    cha_mask = (df['cycle state'] == 1) # Charging half cycle
    dis_mask = (df['cycle state'] == -1) # Discharging half cycle
    if (mask & cha_mask).any():
        cha_hc = df.loc[mask & cha_mask]['half cycle'].iloc[0]
    else:
        cha_hc = None
    if (mask & dis_mask).any():
        dis_hc = df.loc[mask & dis_mask]['half cycle'].iloc[0]
    else:
        dis_hc = None

    return cha_hc, dis_hc


def cycle_from_halfcycle(df, halfcycle):
    # Function for determining which cycle corresponds to a given half cycle.
    try:
        mask = (df['half cycle'] == halfcycle)
        return df.loc[mask]['full cycle'].iloc[0]
    except TypeError:
        print(f"Invalid halfcycle number: {halfcycle}.")
        return None


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
    return df


# IMPORT FUNCTIONS
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
def cycle_summary(df, mass=0, full_mass=0, area=0):
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
            if (cha_df['dt'].sum()): # Make sure the weights are valid
                wts = cha_df['dt']
            else:
                wts = None # If no time has elapsed, this will unweight the time-based averages later to prevent a DivByZero error
            cha_cap = df.loc[mask]['Capacity'].max() # Charge capacity
            cha_energy = simp(df.loc[mask]['Voltage'], x=df.loc[mask]['Capacity']) # Charge energy (Simpson's rule)
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
            if (dis_df['dt'].sum()): # Check that non-zero time has elapsed in the halfcycle
                wts = dis_df['dt']
            else:
                wts = None # If no time has elapsed, this will unweight the time-based averages later to prevent a DivByZero error
            dis_cap = df.loc[mask]['Capacity'].max() # Discharge capacity
            dis_energy = simp(df.loc[mask]['Voltage'], x=df.loc[mask]['Capacity']) # Discharge energy (Simpson's rule)
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