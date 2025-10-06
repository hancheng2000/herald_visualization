import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
import argparse
# from google.colab import auth
# import plotly.express as px
# from jupyter_dash import JupyterDash
# from dash import dcc, html, Input, Output, State
# auth.authenticate_user()
# import herald_visualization.echem as ec
# from herald_visualization.mpr2csv import cycle_mpr2csv
# from herald_visualization.plot import plot_cycling, plot_gitt, parse_cycle_csv, plot_cycling_plotly
from herald_visualization.fancy_plot import voltage_vs_capacity_cycling,plot_multiple_voltage_vs_cycling
# if prompted to put in data path, you can put in anything that makes this cell run. The data path will be automatically taken care of later

default_params = {
              # 'font.family': 'Helvetica',
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
plt.rcParams.update(default_params)


# function for handling id to absolute path
data_path = "/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing" # here is where data path is taken care of
def id_to_path(cellid, root_dir=data_path):
    """
    Find the correct directory path to a data folder from the cell ID
    """

    glob_str = os.path.join('**/outputs/*'+cellid+'_*.csv')
    paths = glob.glob(glob_str, root_dir=root_dir, recursive=True)
    if len(paths) == 1:
        return os.path.join(root_dir, paths[0])
    elif len(paths) == 0: 
        glob_str = os.path.join('**/outputs/*'+cellid+'*.csv')
        paths = glob.glob(glob_str, root_dir=root_dir, recursive=True)
        if len(paths) == 1:
            return os.path.join(root_dir, paths[0])
        elif len(paths) == 0:
            print(f"No paths matched for {cellid}")
            return None
        else:
            print(f"Too many paths matched for {cellid}: {paths}")
            return None
    else:
        print(f"Too many paths matched for {cellid}: {paths}")
        return None

if __name__ == '__main__':
    fig, ax = plt.subplots(2,2,figsize=(18,12),dpi=150)
    p = argparse.ArgumentParser(
        description="Plot charge current vs time for a given cell ID")
    p.add_argument("cell_id", help="the ID of the cell to plot")
    args = p.parse_args()
    cell_id_1 = args.cell_id
    df = pd.read_csv(id_to_path(cell_id_1))
    unique_cycles = df['full cycle'].unique().astype(int).tolist()
    print(unique_cycles)
    cell_id_1_cycles = list(range(0,11))#unique_cycles[:-1]
    output_dict = voltage_vs_capacity_cycling(df,cycles=cell_id_1_cycles,plot=False)
    
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    norm = Normalize(vmin=1, vmax=max(cell_id_1_cycles))
    sm = ScalarMappable(cmap='coolwarm_r', norm=norm)
    start_current = output_dict['specific_current_charge_lst'][0][0]
    print(output_dict['specific_capacity_discharge_formation'][-1])
    for i,cycle in enumerate(cell_id_1_cycles[1:]):
        # if i==3:
        #     continue
        color = sm.cmap(norm(cycle))
        charge_time_start = output_dict['time_charge_lst'][i][0]
        charge_time_adjusted_lst = (output_dict['time_charge_lst'][i] - charge_time_start)/3600
        charge_current_lst = output_dict['specific_current_charge_lst'][i]
        charge_capacity_lst = output_dict['specific_capacity_charge_lst'][i]
        #print(output_dict['specific_capacity_discharge_lst'][-1])
        print(output_dict['specific_capacity_discharge_lst'][i][-1])
        print(charge_capacity_lst[-1])
        if cycle in [1,max(cell_id_1_cycles)]:
            ax[0,0].plot(charge_time_adjusted_lst,charge_current_lst,linestyle='-',color=color)
            ax[1,0].plot(charge_capacity_lst,charge_current_lst,linestyle='-',color=color)
        else:
            ax[0,0].plot(charge_time_adjusted_lst,charge_current_lst,linestyle='--',color=color,linewidth=1.0)
            ax[1,0].plot(charge_capacity_lst,charge_current_lst,linestyle='--',color=color,linewidth=1.0)
        round_charge_current= np.round(charge_current_lst,decimals=0)
        if len(np.where(round_charge_current!=round_charge_current[1])[0])==0:
            print('No CV step')
            continue
        cv_time_start_idx = np.where(round_charge_current!=round_charge_current[1])[0][1]
        
        cv_time = charge_time_adjusted_lst[cv_time_start_idx:]-charge_time_adjusted_lst[cv_time_start_idx]
        cv_current = charge_current_lst[cv_time_start_idx:]
        cv_capacity = charge_capacity_lst[cv_time_start_idx:] - charge_capacity_lst[cv_time_start_idx]
        if cycle in [1,max(cell_id_1_cycles)]:
            ax[0,1].plot(cv_time,cv_current,linestyle='-',color=color)
            ax[1,1].plot(cv_capacity,cv_current,linestyle='-',color=color)
        else:
            ax[0,1].plot(cv_time,cv_current,linestyle='--',color=color,linewidth=1.0)
            ax[1,1].plot(cv_capacity,cv_current,linestyle='--',color=color,linewidth=1.0)


    ax[0,0].axhline(y=start_current/10,linestyle='--',color='black',label='Cutoff Current',linewidth=1.5)
    ax[0,1].axhline(y=start_current/10,linestyle='--',color='black',label='Cutoff Current',linewidth=1.5)
    ax[1,0].axhline(y=start_current/10,linestyle='--',color='black',label='Cutoff Current',linewidth=1.5)
    ax[1,1].axhline(y=start_current/10,linestyle='--',color='black',label='Cutoff Current',linewidth=1.5)



    ax[0,0].set_xlabel('Charge Time (h)')
    ax[0,0].set_ylabel('Specific Current (mA/g-AM)')

    ax[0,1].set_xlabel('CV Time (h)')
    ax[0,1].set_ylabel('Specific Current (mA/g-AM)')

    ax[1,0].set_xlabel('Charge Specific Capacity (mAh/g-AM)')
    ax[1,0].set_ylabel('Specific Current (mA/g-AM)')

    ax[1,1].set_xlabel('CV Specific Capacity (mAh/g-AM)')
    ax[1,1].set_ylabel('Specific Current (mA/g-AM)')
    ax[0,1].set_xlim(0,3)

    #cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), location='right', fraction=0.02, pad=0.04)
    #cbar.set_label(f"Cycle Number")
    fig.subplots_adjust(right=0.88, wspace=0.2, hspace=0.2)
    fig.savefig(cell_id_1+'_current_vs_time_and_capacity.png', dpi=300,bbox_inches='tight')