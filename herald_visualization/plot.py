import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import numpy as np

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
    colors = plt.cm.viridis(np.linspace(0,1,len(full_cycles)*2))
    for i,cycle in enumerate(half_cycles[1:]):
        df1 = df[df['half cycle']==cycle]
        # round the whole df to 4 decimal places
        df1 = df1.round(4)
        # remove 0 specific capacity
        df1 = df1[df1['Specific Capacity']!=0]
        df1['Specific Capacity'] = df1['Specific Capacity'] - df1['Specific Capacity'].min()
        # remove decreasing specific capacity
        df1 = df1[df1['Specific Capacity'].cummax() == df1['Specific Capacity']]
        if cycle % 2 == 1:
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle)],label='Cycle '+str(int((cycle+1)/2)),linestyle='-')
        else:
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle-1)],linestyle='--')
    plt.xlabel('Specific Capacity (mAh/g)-AM')
    plt.ylabel('Voltage (V)')
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
    colors = plt.cm.viridis(np.linspace(0,1,len(full_cycles)))
    for i,cycle in enumerate(full_cycles[:]):
        df1 = df[df['full cycle']==cycle]
        # round the whole df to 4 decimal places
        df1 = df1.round(4)
        # remove 0 specific capacity
        # df1 = df1[df1['Specific Capacity']!=0]
        df1['Specific Capacity'] = df1['Specific Capacity'] - df1['Specific Capacity'].min()
        # remove decreasing specific capacity
        df1 = df1[df1['Specific Capacity'].cummax() == df1['Specific Capacity']]
        plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle)],label='Cycle '+str(int((cycle+1)/2)),linestyle='-')
    plt.xlabel('Specific Capacity (mAh/g)-AM')
    plt.ylabel('Voltage (V)')
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
    plt_params = None,
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
    colors = plt.cm.viridis(np.linspace(0,1,n_cycles))
    colors_i = 0
    for i, df in enumerate(dfs):
        cycles_to_plot = cycles[i]
        for cycle in cycles_to_plot:
            df1 = df[df['full cycle']==cycle]
            # round the whole df to 4 decimal places
            df1 = df1.round(4)
            # remove 0 specific capacity
            df1 = df1[df1['Specific Capacity']!=0]
            df1['Specific Capacity'] = df1['Specific Capacity'] - df1['Specific Capacity'].min()
            # remove decreasing specific capacity
            df1 = df1[df1['Specific Capacity'].cummax() == df1['Specific Capacity']]
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[colors_i],label='Cell '+str(i+1)+' Cycle '+str(int((cycle+1)/2)),linestyle='-')
            colors_i += 1
    plt.xlabel('Specific Capacity (mAh/g)-AM')
    plt.ylabel('Voltage (V)')
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_png:
        if png_filename == None:
            png_filename = os.path.join(dir_name,'outputs','cycle.png')
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
    colors = plt.cm.viridis(np.linspace(0,1,len(full_cycles)*2))
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
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle)],label='Cycle '+str(int((cycle+1)/2)) + 'OCV',linestyle='-')
        else:
            plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle-1)],linestyle='--')
    plt.xlabel('Specific Capacity (mAh/g)-AM')
    plt.ylabel('Voltage (V)')
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_png:
        if png_filename == None:
            png_filename = os.path.join(dir_name,'outputs','cycle.png')
        plt.savefig(png_filename,dpi=300)
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
    half_cycles = list(range(0,len(full_cycles)*2))
    return df, df_sum, full_cycles, half_cycles