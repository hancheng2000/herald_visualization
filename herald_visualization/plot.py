import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import numpy as np
import herald_visualization.echem as ec

default_params = {# 'axes.labelsize': 'x-large',
#               'axes.titlesize': 'x-large',
#               'xtick.labelsize': 'x-large',
#               'ytick.labelsize': 'x-large',
              'font.family': 'serif',
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

def plot_gitt(
        dir_name,
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
    if fig == None and ax == None:
        fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0,1.0,len(full_cycles)*2))
    print(half_cycles)
    for i,cycle in enumerate(half_cycles[1:]):
        df1 = df[df['half cycle']==cycle]
        # round the whole df to 4 decimal places
        df1 = df1.round(4)
        # remove 0 specific capacity
        # df1 = df1[df1['Specific Capacity']!=0]
        df1['Specific Capacity'] = df1['Specific Capacity'] - df1['Specific Capacity'].min()
        # remove decreasing specific capacity
        df1 = df1[df1['Specific Capacity'].cummax() == df1['Specific Capacity']]
        if cycle == 1:
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle)],label='Cycle 0',linestyle='-')
        else:
            if cycle % 2 == 1:
                plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle-1)],label='Cycle '+str(int((cycle-1)/2)),linestyle='-')
            else:
                plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle)],linestyle='-')
    plt.xlabel('Specific Capacity (mAh/g)-Cathode AM')
    plt.ylabel('Voltage vs Li/Li+ (V)')
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_png:
        if png_filename == None:
            png_filename = os.path.join(dir_name,'outputs','cycle.png')
        plt.savefig(png_filename,dpi=300)
    return fig, ax

def plot_cycle(
        dir_name,
        fig=None,
        ax=None,
        full_cycles = None,
        save_png = False,
        png_filename = None,
        plt_params = None,
    ):
    # plotting params
    if plt_params != None:
        plt.rcParams.update(plt_params)
    else:
        plt.rcParams.update(default_params)
    csv_files = glob.glob(os.path.join(dir_name,'*.csv'))
    print(csv_files)
    summary_file = os.path.join(dir_name,'cycle_summary.csv')
    data_file = [file for file in csv_files if 'cycle_summary' not in file][0]
    df = pd.read_csv(data_file)
    df_sum = pd.read_csv(summary_file)
    # if full cycle is not specified, use all cycles
    if full_cycles == None:
        full_cycles = df_sum['full cycle'].tolist()
    if fig == None and ax == None:
        fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0,1.0,len(full_cycles)))
    for i,cycle in enumerate(full_cycles[:]):
        df1 = df[df['full cycle']==cycle]
        # round the whole df to 4 decimal places
        df1 = df1.round(4)
        # remove 0 specific capacity
        # df1 = df1[df1['Specific Capacity']!=0]
        df1['Specific Capacity'] = df1['Specific Capacity'] - df1['Specific Capacity'].min()
        # remove decreasing specific capacity
        df1 = df1[df1['Specific Capacity'].cummax() == df1['Specific Capacity']]
        plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle)],label=f'Cycle {cycle}',linestyle='-')
    plt.xlabel('Specific Capacity (mAh/g)-Cathode AM')
    plt.ylabel('Voltage vs Li/Li+ (V)')
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_png:
        if png_filename == None:
            png_filename = os.path.join(dir_name,'outputs','cycle.png')
        plt.savefig(png_filename,dpi=300)
    return fig, ax    

def plot_multi_cell(
    file_list,
    fig=None, 
    ax=None,
    cycles = 'all',
    save_png = False,
    png_filename = None,
    plt_params = None
    ):
    """
    plot the cycle of multiple cells

    Args:
    - file_list: list of csv files to plot cycling data
    - fig, ax: matplotlib figure and axis objects
    - cycles: str or list of lists, specify the cycles to plot. Str options can be 'all', 'first' and 'last'. Default is 'all'.
    - save_png: bool, save the plot as png file. Default is False.

    Returns:
    - fig, ax: matplotlib figure and axis objects
    """    
    if fig == None and ax == None:
        fig, ax = plt.subplots()
    dfs = []
    for file in file_list:
        dfs.append(pd.read_csv(file))
    if type(cycles) == list:
        cycles = cycles
    elif cycles == 'all':
        cycles = [df['full cycle'].unique().tolist() for df in dfs]
    elif cycles == 'first':
        cycles = [[1] for df in dfs]
    elif cycles == 'last':
        cycles = [[df['full cycle'].max() for df in dfs]]
    # count all elements in cycles, including all elements in sublists, and assign one color to each cycle
    n_cycles = sum([len(cycle) for cycle in cycles])
    colors = plt.cm.rainbow(np.linspace(0,1.0,n_cycles))
    colors_i = 0
    for i, df in enumerate(dfs):
        cycles_to_plot = cycles[i]
        for cycle in cycles_to_plot:
            df1 = df[df['full cycle']==cycle]
            # round the whole df to 4 decimal places
            df1 = df1.round(4)
            # remove 0 specific capacity
            # df1 = df1[df1['Specific Capacity']!=0]
            df1['Specific Capacity'] = df1['Specific Capacity'] - df1['Specific Capacity'].min()
            # remove decreasing specific capacity
            df1 = df1[df1['Specific Capacity'].cummax() == df1['Specific Capacity']]
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[colors_i],label='Cell '+str(i+1)+' Cycle '+str(int((cycle+1)/2)),linestyle='-')
            colors_i += 1
    plt.xlabel('Specific Capacity (mAh/g cathode AM)')
    plt.ylabel('Voltage (V)')
    plt.tight_layout()
    if save_png:
        plt.savefig(png_filename,dpi=300)
    return fig, ax


def plot_ocv(
        dir_name,
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
    if fig == None and ax == None:
        fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0,1.0,len(full_cycles)*2))
    for i,cycle in enumerate(half_cycles[1:]):
        df1 = df[df['half cycle']==cycle]
        # round the whole df to 4 decimal places
        df1 = df1.round(4)
        df1['Specific Capacity'] = df1['Specific Capacity'] - df1['Specific Capacity'].min()
        # remove decreasing specific capacity
        df1 = df1[df1['Specific Capacity'].cummax() == df1['Specific Capacity']]
        # get the ocv through 'state' column, "R" for OCV
        df1 = df1[df1['state']=='R']
        df1 = df1.groupby('Specific Capacity').max().reset_index()
        if cycle % 2 == 1: 
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle)],label='Cycle '+str(int((cycle-1)/2)) + 'OCV',linestyle='-')
        else:
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle-1)],linestyle='--')
    plt.xlabel('Specific Capacity (mAh/g)-Cathode AM')
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

def multi_df_dqdv_plot(dfs, labels,
    halfcycles=None,
    cycle=1,
    colormap='tab10', 
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

    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.

    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots() 
    cm = plt.get_cmap(colormap)

    for i, df in enumerate(dfs):
        # If halfcycles arg is supplied, use that to determine what to plot
        # Otherwise use cycle to define the halfcycles to plot
        if not isinstance(halfcycles, list): # Will evaluate True if halfcycles is left as None or is not a valid list
            halfcycles = ec.halfcycles_from_cycle(df, cycle)

        for halfcycle in halfcycles:
            df_cycle = df[df['half cycle'] == halfcycle]
            voltage, dqdv, _ = ec.dqdv_single_cycle(df_cycle[capacity_label],
                                        df_cycle[voltage_label], 
                                        window_size_1=window_size_1,
                                        polyorder_1=polyorder_1,
                                        polynomial_spline=polynomial_spline,
                                        s_spline=s_spline,
                                        window_size_2=window_size_2,
                                        polyorder_2=polyorder_2,
                                        final_smooth=final_smooth)
            
            # Make sure only one instance of each df appears in the legend
            if halfcycle == halfcycles[0]:
                label = labels[i]
            else:
                label = '_'
            ax.plot(voltage, dqdv, color=cm(i), label=label)

    ax.set_xlabel('Voltage / V')
    y_labels = {'Capacity': 'dQ/dV / mAh/V', 'Specific Capacity': 'dQ/dV / mAh/g/V'}
    ax.set_ylabel(y_labels[capacity_label])
    ax.axhline(0, linewidth=2, color='k')
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig, ax


def parse_csv(
    dir_name,
    full_cycles = None,
    half_cycles = None,   
):
    csv_files = glob.glob(os.path.join(dir_name,'*.csv'))
    print(csv_files)
    summary_file = os.path.join(dir_name,'cycle_summary.csv')
    data_file = [file for file in csv_files if 'cycle_summary' not in file][0]
    df = pd.read_csv(data_file)
    df_sum = pd.read_csv(summary_file)
    # if full cycle is not specified, use all cycles
    # only plot half cycles when half cycle is specified and full cycle is not specified
    if full_cycles == None and half_cycles == None:
        full_cycles = df_sum['full cycle'].tolist()
        half_cycles = df['half cycle'].unique().tolist()
    elif full_cycles != None and half_cycles == None:
        half_cycles = df['half cycle'].unique().tolist()
    return df, df_sum, full_cycles, half_cycles