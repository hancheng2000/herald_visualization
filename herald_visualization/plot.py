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
    plt.xlabel('Specific Capacity (mAh/g)')
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
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0,1,len(full_cycles)))
    for i,cycle in enumerate(full_cycles[:]):
        df1 = df[df['full cycle']==cycle]
        # round the whole df to 4 decimal places
        df1 = df1.round(4)
        # remove 0 specific capacity
        df1 = df1[df1['Specific Capacity']!=0]
        df1['Specific Capacity'] = df1['Specific Capacity'] - df1['Specific Capacity'].min()
        # remove decreasing specific capacity
        df1 = df1[df1['Specific Capacity'].cummax() == df1['Specific Capacity']]
        plt.plot(df1['Specific Capacity'],df1['Voltage'],color=colors[int(cycle)],label='Cycle '+str(int((cycle+1)/2)),linestyle='-')
    plt.xlabel('Specific Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_png:
        if png_filename == None:
            png_filename = os.path.join(dir_name,'outputs','cycle.png')
        plt.savefig(png_filename,dpi=300)
    return fig, ax    