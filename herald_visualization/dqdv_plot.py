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
import herald_visualization.echem as ec
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

# cells_df = pd.read_csv('/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing/In-house cells and syntheses - Coin Cells.csv')
# available_ids = cells_df['Test ID'].tolist()

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

    fig, ax = plt.subplots(1,2,figsize=(18,6),dpi=150)

    p = argparse.ArgumentParser(
        description="Plot charge current vs time for a given cell ID")
    p.add_argument("cell_id", help="the ID of the cell to plot")
    args = p.parse_args()
    cell_id_1 = args.cell_id
    df = pd.read_csv(id_to_path(cell_id_1))
    unique_cycles = df['full cycle'].unique().astype(int).tolist()
    print(unique_cycles)
    cell_id_1_cycles = list(range(0,6))#unique_cycles[:-1]
    output_dict = voltage_vs_capacity_cycling(df,cycles=cell_id_1_cycles,plot=False)
    cycle_0_end_discharge = output_dict['specific_capacity_rest_end_formation'][0]
    cycle_1_charge_capacity = output_dict['specific_capacity_charge_lst'][0][-1]
    charge_capacity_shift = cycle_0_end_discharge - cycle_1_charge_capacity
    fig,ax[0] = plot_multiple_voltage_vs_cycling([output_dict['voltage_discharge_formation']+output_dict['voltage_rest_end_formation']],[output_dict['specific_capacity_discharge_formation']+output_dict['specific_capacity_rest_end_formation']],[0],linestyle='--',color_customize=None,fig=fig,ax=ax[0],colorbar=False)


    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    norm = Normalize(vmin=1, vmax=max(cell_id_1_cycles))
    sm = ScalarMappable(cmap='coolwarm_r', norm=norm)

    for i, cycle in enumerate(output_dict['cycle_lst']):
        voltage_discharge_tot_lst = output_dict['voltage_discharge_tot_lst'][i]
        specific_capacity_discharge_tot_lst = output_dict['specific_capacity_discharge_tot_lst'][i]
        voltage_charge_lst = output_dict['voltage_charge_lst'][i]
        specific_capacity_charge_lst = output_dict['specific_capacity_charge_lst'][i]
        cycle = output_dict['cycle_lst'][i]
        voltage_discharge_lst = output_dict['voltage_discharge_lst'][i]
        specific_capacity_discharge_lst = output_dict['specific_capacity_discharge_lst'][i]
        idx_voltage_discharge_less_than_2 = np.where(np.array(voltage_discharge_tot_lst)<2.5)[0][0]
        specific_capacity_lower_than_v2 = np.array(specific_capacity_discharge_tot_lst)[idx_voltage_discharge_less_than_2:]
        idx_specific_capacity_lower_than_5_in_v2 = np.where(np.array(specific_capacity_lower_than_v2)<5)[0]
        if len(idx_specific_capacity_lower_than_5_in_v2) > 0:
            delete_idx_lst = []
            for j in idx_specific_capacity_lower_than_5_in_v2:
                specific_capacity_j = specific_capacity_lower_than_v2[j]
                tot_idx = np.where(np.array(specific_capacity_discharge_tot_lst) == specific_capacity_j)[0]
                for k in tot_idx:
                    if k < idx_voltage_discharge_less_than_2:
                        continue 
                    else:
                        delete_idx_lst.append(k)
            #delete the idx from the voltage_discharge_tot_lst and specific_capacity_discharge_tot_lst
            voltage_discharge_tot_lst = np.delete(voltage_discharge_tot_lst, delete_idx_lst)
            specific_capacity_discharge_tot_lst = np.delete(specific_capacity_discharge_tot_lst, delete_idx_lst)
        # if cycle in [12,13]:
        #     fig,ax[0]=plot_multiple_voltage_vs_cycling([voltage_discharge_tot_lst],[np.array(specific_capacity_discharge_tot_lst)],[cycle],linestyle='-',color_customize='k',fig=fig,ax=ax[0],colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles),variable_linewidth=False)
        #     fig,ax[0]=plot_multiple_voltage_vs_cycling([voltage_charge_lst],[specific_capacity_discharge_tot_lst[-1]-np.array(specific_capacity_charge_lst)],[cycle],linestyle='-',color_customize='k',fig=fig,ax=ax[0],colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles),variable_linewidth=False)
        # else:
        fig,ax[0]=plot_multiple_voltage_vs_cycling([voltage_charge_lst],[np.array(specific_capacity_charge_lst)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax[0],colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles),variable_linewidth=True)
        fig,ax[0]=plot_multiple_voltage_vs_cycling([voltage_discharge_tot_lst],[specific_capacity_charge_lst[-1]-np.array(specific_capacity_discharge_tot_lst)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax[0],colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles),variable_linewidth=True)
        voltage_discharge,smooth_dqdv_discharge,smooth_cap_discharge = ec.dqdv_single_cycle(np.array(specific_capacity_discharge_lst), np.array(voltage_discharge_lst))
        voltage_charge,smooth_dqdv_charge,smooth_cap_charge = ec.dqdv_single_cycle(np.array(specific_capacity_charge_lst), np.array(voltage_charge_lst))
        color = sm.cmap(norm(cycle))
        if cycle in [1,max(cell_id_1_cycles)]:
            ax[1].plot(voltage_discharge, smooth_dqdv_discharge, linestyle='-',color=color)
            ax[1].plot(voltage_charge, smooth_dqdv_charge, linestyle='-',color=color)
        # elif cycle in [12,13]:
        #     ax[1].plot(voltage_discharge, smooth_dqdv_discharge, linestyle='-',color='k')
        #     ax[1].plot(voltage_charge, smooth_dqdv_charge, linestyle='-',color='k')
        else:
            ax[1].plot(voltage_discharge, smooth_dqdv_discharge, linestyle='--',color=color,linewidth=1.0)
            ax[1].plot(voltage_charge, smooth_dqdv_charge, linestyle='--',color=color,linewidth=1.0)
    ax[1].axvline(x=2.66, color='k', linestyle='--', linewidth=1.0)
    ax[1].axvline(x=2.9, color='k', linestyle='--', linewidth=1.0)
    #add text to the vline
    #ax[1].text(2.7, 1500, r'Theo. OCV Fe$^{0/2^+}$', rotation=90, verticalalignment='bottom', horizontalalignment='right', fontsize=16)
    #ax[1].text(2.9, 1500, r'Theo. OCV Fe$^{2^+/3^+}$', rotation=90, verticalalignment='bottom', horizontalalignment='right', fontsize=16)

    cbar = fig.colorbar(sm, ax=ax[1])
    cbar.set_label(f"Cycle Number")
    ax[1].set_xlabel('Voltage (V)')
    ax[1].set_ylabel('dQ/dV (mAh/gV)')
    #ax[1].set_ylim([-2500,2500])
    fig.savefig(cell_id_1+'_dqdv.png', dpi=300,bbox_inches='tight')


