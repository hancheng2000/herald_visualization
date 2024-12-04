import pandas as pd
import matplotlib.pyplot as plt
import os, glob

plt_params = {# 'axes.labelsize': 'x-large',
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
plt.rcParams.update(plt_params)

def plot_discharge(
        dir_name,
        full_cycles = None,
        half_cycles = None,
    ):
    csv_files = glob.glob(os.path.join(dir_name,'outputs','*.csv'))
    summary_file = os.path.join(dir_name,'outputs','cycle_summary.csv')
    data_file = [file for file in csv_files if 'cycle_summary' not in file][0]
    df = pd.read_csv(data_file)
    df_sum = pd.read_csv(summary_file)
    # if full cycle is not specified, use all cycles
    # only plot half cycles when half cycle is specified and full cycle is not specified
    if full_cycles == None and half_cycles == None:
        full_cycles = df_sum['full cycle'].tolist()
    fig, ax = plt.subplots()
    for cycle in full_cycles:
        df1 = df[df['full cycle']==cycle]
        # remove 0 specific capacity
        df
    plt.plot(df)