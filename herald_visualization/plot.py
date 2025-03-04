import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, glob, re
import numpy as np
import herald_visualization.echem as ec
from herald_visualization.echem import halfcycles_from_cycle, cycle_from_halfcycle

default_params = {
              'font.family': 'Helvetica',
              'axes.labelsize': 20,
              'axes.labelweight': 'bold',  # Make axes labels bold
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'xtick.major.size': 7,
              'ytick.major.size': 7,
              'xtick.major.width': 2.0,
              'ytick.major.width': 2.0,
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              'font.size': 24,
              'axes.linewidth': 2.0,
              'lines.dashed_pattern': (5, 2.5),
              'lines.markersize': 10,
              'lines.linewidth': 3,
              'lines.markeredgewidth': 1,
              'lines.markeredgecolor': 'k',
              'legend.fontsize': 16,  # Adjust the font size of the legend
              'legend.title_fontsize': 24,  # Increase legend title size if needed
              'legend.frameon': True
    }

capacity_col_to_label = {
    'Capacity': 'Capacity (mAh)',
    'Specific Capacity': 'Specific Capacity (mAh/g cathode)',
    'Specific Capacity Total AM': 'Specific Capacity (mAh/g AM)',
    'Areal Capacity': 'Areal Capacity (mAh/cm$^2$)'
}

dqdv_capacity_col_to_label = {
    'Capacity': 'dQ/dV (mAh/V)',
    'Specific Capacity': 'dQ/dV (mAh/V/g cathode)',
    'Specific Capacity Total AM': 'dQ/dV (mAh/V/g AM)',
    'Areal Capacity': 'dQ/dV (mAh/V/cm$^2$)'
}

def plot_gitt(
        dir_name,
        norm=None,
        fig=None,
        ax=None,
        full_cycles = None,
        half_cycles = None,
        save_png = False,
        png_filename = None,
        plt_params = None
    ):

    # plotting params
    if plt_params != None:
        plt.rcParams.update(plt_params)
    else:
        plt.rcParams.update(default_params)
    df, df_sum, _ = parse_cycle_csv(dir_name)

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

    if fig == None and ax == None:
        fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0,1.0,len(full_cycles)*2))
    print(half_cycles)
    for i,cycle in enumerate(half_cycles[1:]):
        df1 = df[df['half cycle']==cycle]
        # round the whole df to 4 decimal places
        df1 = df1.round(4)
        # remove 0 specific capacity
        # df1 = df1[df1[capacity_col]!=0]
        df1[capacity_col] = df1[capacity_col] - df1[capacity_col].min()
        # remove decreasing specific capacity
        df1 = df1[df1[capacity_col].cummax() == df1[capacity_col]]
        if cycle == 1:
            plt.plot(df1[capacity_col],df1['Voltage'],color=colors[int(cycle)],label='Cycle 0',linestyle='-')
        else:
            if cycle % 2 == 1:
                plt.plot(df1[capacity_col],df1['Voltage'],color=colors[int(cycle-1)],label='Cycle '+str(int((cycle-1)/2)),linestyle='-')
            else:
                plt.plot(df1[capacity_col],df1['Voltage'],color=colors[int(cycle)],linestyle='-')
    plt.xlabel(capacity_label)
    plt.ylabel('Voltage vs Li/Li+ (V)')
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_png:
        if png_filename == None:
            png_filename = os.path.join(dir_name,'outputs','cycle.png')
        plt.savefig(png_filename,dpi=300)
    return fig, ax

def plot_ocv(
        dir_name,
        norm=None,
        fig=None,
        ax=None,
        full_cycles = None,
        half_cycles = None,
        save_png = False,
        png_filename = None,
        plt_params = None,
    ):

    # plotting params
    if plt_params != None:
        plt.rcParams.update(plt_params)
    else:
        plt.rcParams.update(default_params)
    df, df_sum, full_cycles, half_cycles = parse_csv(dir_name,full_cycles,half_cycles)

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

    if fig == None and ax == None:
        fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0,1.0,len(full_cycles)*2))
    for i,cycle in enumerate(half_cycles[1:]):
        df1 = df[df['half cycle'] == cycle]
        # round the whole df to 4 decimal places
        df1 = df1.round(4)
        df1[capacity_col] = df1[capacity_col] - df1[capacity_col].min()
        # remove decreasing specific capacity
        df1 = df1[df1[capacity_col].cummax() == df1[capacity_col]]
        # get the ocv through 'state' column, 0 for OCV
        df1 = df1[df1['state'] == 0]
        df1 = df1.groupby(capacity_col).max().reset_index()
        if cycle % 2 == 1: 
            plt.plot(df1[capacity_col],df1['Voltage'],color=colors[int(cycle)],label='Cycle '+str(int((cycle-1)/2)) + 'OCV',linestyle='-')
        else:
            plt.plot(df1[capacity_col],df1['Voltage'],color=colors[int(cycle-1)],linestyle='--')
    plt.xlabel(capacity_label)
    plt.ylabel('Voltage vs Li/Li+ (V)')
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_png:
        if png_filename == None:
            png_filename = os.path.join(dir_name,'outputs','cycle.png')
        plt.savefig(png_filename,dpi=300)
    return fig, ax    

def plot_eis(
    dir_name,
    fig=None,
    ax=None,
    cycles = 'all',
    save_png = False,
    png_filename = None,
    plt_params = None,
    seperate_im = False,
    ):
    if plt_params != None:
        plt.rcParams.update(plt_params)
    else:
        plt.rcParams.update(default_params)    
    csv_file = glob.glob(os.path.join(dir_name,'*EIS*.csv'))[0]
    df = pd.read_csv(csv_file)
    if cycles == 'all':
        cycles = df['cycle number'].unique().tolist()
    max_imaginary_impedance = df['-Im(Z)/Ohm'].max()
    if fig == None and ax == None:
        fig, ax = plt.subplots()
    for i in range(1,int(df['cycle number'].max()+1)):
        df1 = df[df['cycle number']==i]
        if not seperate_im:
            plt.plot(df1['Re(Z)/Ohm'],df1['-Im(Z)/Ohm'],label=f'{i}')
        else:
            offset = max_imaginary_impedance * (i-1) * 0.5
            plt.plot(df1['Re(Z)/Ohm'],df1['-Im(Z)/Ohm']+offset,label=f'{i}')
    plt.xlabel('Re(Z)/Ohm')
    plt.ylabel('-Im(Z)/Ohm')
    plt.tight_layout()
    if save_png:
        plt.savefig(png_filename,dpi=300)
    return fig, ax

# Utility functions

def parse_cycle_csv(dir_name,
                    summary_filename='cycle_summary'):
    csv_files = glob.glob(os.path.join(dir_name,'*.csv'))
    summary_file = os.path.join(dir_name,f'{summary_filename}.csv')
    data_file = [file for file in csv_files if summary_filename not in file and 'PEIS' not in file][0]
    df = ec.echem_file_loader(data_file)
    try:
        df_sum = pd.read_csv(summary_file)
    except:
        df_sum = None
    return df, df_sum

def subscript_formula(text):
    pattern = r'([A-Z][a-z]*)(\d+\.?\d*)'
    replacement = r'\1$_{\2}$'
    return re.sub(pattern, replacement, text)

def retention_over_cycles(df_sum, data_col, start_cycle=0, end_cycle='last', per_cycle=True):
    if end_cycle == 'last':
        end_cycle = df_sum[df_sum[data_col].notna()].index.max()
    if end_cycle <= start_cycle:
        print("end_cycle should be greater than start_cycle")
    ratio = df_sum.loc[end_cycle, data_col]/df_sum.loc[start_cycle, data_col]
    if per_cycle:
        return ratio**(1/(end_cycle-start_cycle))
    else:
        return ratio

# Plotting functions

def plot_cycling(
        dfs,
        cycles='all',
        labels=None,
        colormap=None,
        capacity_col='Specific Capacity Total AM',
        voltage_col='Voltage',
        fig=None,
        ax=None,
        subplots_kwargs={},
        plot_kwargs={}
    ):
    """
    Plot voltage vs. capacity cycling data from one or more cells.

    Arguments:
    - dfs (pandas.DataFrame or dict of pandas.DataFrame): dataframe(s) to plot, e.g. from parse_cycle_csv
    - halfcycles (list, opt): halfcycles to plot, overrides cycles
    - cycles (list): full cycles to plot, 'all' plots all cycles in df
    - colormap: colormap to use for plot
    - capacity_col: column in dfs to plot as x-axis
    - voltage_col: column in dfs to plot as y-axis
    """

    # TODO add functionality to select specific half cycles

    if fig == None and ax == None:
        fig, ax = plt.subplots(**subplots_kwargs)
    
    def colormap_picker(number_of_plots):
        if not colormap:
            if number_of_plots <= 10:
                return plt.get_cmap('tab10')
            elif number_of_plots <= 20:
                return plt.get_cmap('tab20')
            else:
                raise ValueError("Too many plots for default colormaps.")
        else:
            return plt.get_cmap(colormap)

    # Function can handle either multiple dfs or multiple cycles
    multi_df_bool = isinstance(dfs, dict) and len(dfs) > 1
    multi_cycle_bool = (isinstance(cycles, list) and len(cycles) > 1) or cycles == 'all'
    if multi_df_bool and multi_cycle_bool:
        raise NotImplementedError("Either multiple cells or multiple cycles may be plotted.")
    
    # Plot multiple cycles for one df
    elif multi_cycle_bool:
        df = dfs.values[0] if isinstance(dfs, dict) else dfs
        if cycles == 'all':
            cycles = df['full cycle'].unique()
        else:
            cycles = list(cycles) # Ensure that cycles is a list, even if an int is passed
        cm = colormap_picker(len(cycles))
        custom_lines = [Line2D([0], [0], color=cm(i), lw=2) for i, _ in enumerate(cycles)]
        if not labels: # Automatically set labels based on cycle numbers if none are provided
            labels = [f'Cycle {n}' for n in cycles]
        for i, cycle in enumerate(cycles):
            halfcycles = halfcycles_from_cycle(df, cycle)
            for halfcycle in halfcycles:
                mask = df['half cycle'] == halfcycle
                df1 = df.loc[mask]
                # Remove points from OCV at the beginning of the half cycle
                df1 = df1.loc[df1['Capacity']!=0]
                # Make sure half cycle exists within the data
                if sum(mask) > 0:
                    ax.plot(df1[capacity_col], df1[voltage_col], color=cm(i), label=labels[i], **plot_kwargs)

    # Plot one cycle for one or multiple dfs
    else:
        cycle = cycles[0] if isinstance(cycles, list) else cycles
        dfs = dfs if isinstance(dfs, dict) else {'_': dfs} # Ensure that dfs is a dict, even if a dataframe is passed
        cm = colormap_picker(len(dfs))
        custom_lines = [Line2D([0], [0], color=cm(i), lw=2) for i, _ in enumerate(dfs)]
        if not labels: # Automatically set labels based on dict keys if none are provided
            labels = list(dfs.keys())
        for i, df in enumerate(dfs.values()):
            halfcycles = halfcycles_from_cycle(df, cycle)
            for halfcycle in halfcycles:
                mask = df['half cycle'] == halfcycle
                df1 = df.loc[mask]
                # Remove points from OCV at the beginning of the half cycle
                df1 = df1.loc[df1['Capacity'] != 0]
                # Make sure half cycle exists within the data
                if sum(mask) > 0:
                    ax.plot(df1[capacity_col], df1[voltage_col], color=cm(i), label=labels[i], **plot_kwargs)

    try:
        capacity_label = capacity_col_to_label[capacity_col]
    except KeyError:
        capacity_label = capacity_col
    ax.set_xlabel(capacity_label)
    
    if voltage_col == 'Voltage':
        voltage_label = 'Voltage (V)'
    else:
        voltage_label = f'{voltage_col} (V)'
    ax.set_ylabel(voltage_label)
    
    if labels:
        ax.legend(custom_lines, labels)
    return fig, ax

def plot_dqdv(
        dfs,
        halfcycles=None,
        cycles=1,
        labels=None,
        colormap=None, 
        capacity_col='Specific Capacity Total AM', 
        voltage_col='Voltage',
        fig=None,
        ax=None,
        subplots_kwargs={},
        plot_kwargs={},
        polynomial_spline=3, s_spline=1e-5,
        polyorder_1 = 5, window_size_1=101,
        polyorder_2 = 5, window_size_2=1001,
        final_smooth=True,
        v_margin = 0.01
    ):
    """
    Plot multiple dQ/dV cycles on the same plot with a colormap.
    Uses the internal dqdv_single_cycle function to calculate the dQ/dV curves.

    Parameters:
        df: DataFrame containing the data.
        labels: List of labels to make legend.
        halfcycles (list): Half cycles to plot. Will override cycle.
        cycle (int): Cycle number to plot.
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
        v_margin (float, opt): Voltage margin from maximum and minimum voltage in a cycle for which points are removed from analysis.

    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.

    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    if fig == None and ax == None:
        fig, ax = plt.subplots(**subplots_kwargs)
    def colormap_picker(number_of_plots):
        if not colormap:
            if number_of_plots <= 10:
                return plt.get_cmap('tab10')
            elif number_of_plots <= 20:
                return plt.get_cmap('tab20')
            else:
                raise ValueError("Too many plots for default colormaps.")
        else:
            return plt.get_cmap(colormap)

    # Function can handle either multiple dfs or multiple cycles
    multi_df_bool = isinstance(dfs, dict) and len(dfs) > 1
    multi_cycle_bool = (isinstance(cycles, list) and len(cycles) > 1) or cycles == 'all'
    if multi_df_bool and multi_cycle_bool:
        raise NotImplementedError("Either multiple cells or multiple cycles may be plotted.")
    
    # Plot multiple cycles for one df
    elif multi_cycle_bool:
        df = dfs.values[0] if isinstance(dfs, dict) else dfs
        if cycles == 'all':
            cycles = df['full cycle'].unique()
        else:
            cycles = list(cycles) # Ensure that cycles is a list, even if an int is passed
        cm = colormap_picker(len(cycles))
        custom_lines = [Line2D([0], [0], color=cm(i), lw=2) for i, _ in enumerate(cycles)]
        if not labels: # Automatically set labels based on cycle numbers if none are provided
            labels = [f'Cycle {n}' for n in cycles]
        for i, cycle in enumerate(cycles):
            halfcycles = halfcycles_from_cycle(df, cycle)
            for halfcycle in halfcycles:
                mask = df['half cycle'] == halfcycle
                # Make sure half cycle exists within the data
                if sum(mask) > 0:
                    df1 = df.loc[mask]
                    # Remove points from OCV
                    df1 = df1.loc[df1['state'] != 0]
                    # Remove points from constant voltage portion
                    v_max = df1[voltage_col].max()
                    v_min = df1[voltage_col].min()
                    df1 = df1.loc[(df1[voltage_col] < v_max - v_margin) & (df1[voltage_col] > v_min + v_margin)]
                    voltage, dqdv, _ = ec.dqdv_single_cycle(df1[capacity_col],
                                df1[voltage_col], 
                                window_size_1=window_size_1,
                                polyorder_1=polyorder_1,
                                polynomial_spline=polynomial_spline,
                                s_spline=s_spline,
                                window_size_2=window_size_2,
                                polyorder_2=polyorder_2,
                                final_smooth=final_smooth)
                    ax.plot(voltage, dqdv, color=cm(i), label=labels[i], **plot_kwargs)

    # Plot one cycle for one or multiple dfs
    else:
        cycle = cycles[0] if isinstance(cycles, list) else cycles
        dfs = dfs if isinstance(dfs, dict) else {'_': dfs} # Ensure that dfs is a dict, even if a dataframe is passed
        cm = colormap_picker(len(dfs))
        custom_lines = [Line2D([0], [0], color=cm(i), lw=2) for i, _ in enumerate(dfs)]
        if not labels: # Automatically set labels based on dict keys if none are provided
            labels = list(dfs.keys())
        for i, df in enumerate(dfs.values()):
            halfcycles = halfcycles_from_cycle(df, cycle)
            for halfcycle in halfcycles:
                mask = df['half cycle'] == halfcycle
                # Make sure half cycle exists within the data
                if sum(mask) > 0:
                    df1 = df.loc[mask]
                    # Remove points from OCV
                    df1 = df1.loc[df1['state'] != 0]
                    # Remove points from constant voltage portion
                    v_max = df1[voltage_col].max()
                    v_min = df1[voltage_col].min()
                    df1 = df1.loc[(df1[voltage_col] < v_max - v_margin) & (df1[voltage_col] > v_min + v_margin)]
                    voltage, dqdv, _ = ec.dqdv_single_cycle(df1[capacity_col],
                                df1[voltage_col], 
                                window_size_1=window_size_1,
                                polyorder_1=polyorder_1,
                                polynomial_spline=polynomial_spline,
                                s_spline=s_spline,
                                window_size_2=window_size_2,
                                polyorder_2=polyorder_2,
                                final_smooth=final_smooth)
                    ax.plot(voltage, dqdv, color=cm(i), label=labels[i], **plot_kwargs)
    
    # Add line at 0 for visual clarity
    ax.axhline(0, lw=2, color='k')
    
    try:
        capacity_label = dqdv_capacity_col_to_label[capacity_col]
    except KeyError:
        capacity_label = capacity_col
    ax.set_ylabel(capacity_label)
    
    if voltage_col == 'Voltage':
        voltage_label = 'Voltage (V)'
    else:
        voltage_label = f'{voltage_col} (V)'
    ax.set_xlabel(voltage_label)
    
    if labels:
        ax.legend(custom_lines, labels)
    return fig, ax