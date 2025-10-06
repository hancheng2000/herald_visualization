import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
import argparse
import json
# from google.colab import auth
# import plotly.express as px
# from jupyter_dash import JupyterDash
# from dash import dcc, html, Input, Output, State
# auth.authenticate_user()
# import herald_visualization.echem as ec
# from herald_visualization.mpr2csv import cycle_mpr2csv
# from herald_visualization.plot import plot_cycling, plot_gitt, parse_cycle_csv, plot_cycling_plotly
from herald_visualization.fancy_plot import voltage_vs_capacity_cycling,plot_multiple_voltage_vs_cycling,voltage_vs_capacity_GITT
from herald_visualization.mpr2csv import cycle_mpr2csv 
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
    from scipy.integrate import simpson
    
    fig, ax = plt.subplots(1,1,figsize=(9,6),dpi=150)
    p = argparse.ArgumentParser(
        description="Plot voltage vs capacity for a given cell ID")
    p.add_argument("cell_id", help="the ID of the cell to plot")
    args = p.parse_args()
    cell_id_1 = args.cell_id
    df = cycle_mpr2csv("/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing/Li-FeF3-25CB-IL3-GFA/250325-CC088L-C20_1V_4V_CV-GITT_10SoC_3h")
    #df = pd.read_csv(id_to_path(cell_id_1))
    unique_cycles = df['full cycle'].unique().astype(int).tolist()
    print(unique_cycles)
    cell_id_1_cycles = unique_cycles#[:-1]
    print(cell_id_1_cycles)
    output_dict = voltage_vs_capacity_GITT(df,cycles=[0],plot=True)#1,2,3,4,6,7,8,9,10,11,12,13,14
    # save to df 
    df = pd.DataFrame({'capacity':output_dict['full_discharge_specific_capacity_formation_cycle'],'voltage':output_dict['full_discharge_voltage_formation_cycle']})
    df.to_csv(f'{cell_id_1}_voltage_vs_capacity_GITT_formation_cycle.csv',index=False)
    # pd.DataFrame(output_dict).to_csv(f'{cell_id_1}_voltage_vs_capacity_GITT_formation_cycle.csv',index=False)
    for key in output_dict.keys():
        print(key, len(output_dict[key]))
    # with open(f"{cell_id_1}_voltage_vs_capacity_GITT_formation_cycle.json", "w") as f:
    #     json.dump(output_dict, f)
    # fig,ax = plot_multiple_voltage_vs_cycling([output_dict['voltage_discharge_formation']+output_dict['voltage_rest_end_formation']],[output_dict['specific_capacity_discharge_formation']+output_dict['specific_capacity_rest_end_formation']],[0],linestyle='--',color_customize=None,fig=fig,ax=ax,colorbar=False)
    # fig,ax = plot_multiple_voltage_vs_cycling(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'],output_dict['cycle_lst'],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=False)
    # fig,ax = plot_multiple_voltage_vs_cycling(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'],output_dict['cycle_lst'],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=True)

    formation_cycle_se = simpson(x=output_dict['specific_capacity_discharge_formation'],y=output_dict['voltage_discharge_formation'])
    #ax.set_title(f'Cell ID: {cell_id_1}')
    ax.set_xlim([0,601*1.2])
    ax.set_ylim([0,4.5])
    plt.tight_layout()
    fig.savefig(save_folder+cell_id_1+'_voltage_vs_capacity.png', dpi=300)

    discharge_se_lst_cell_id_1= []
    charge_se_lst_cell_id_1 = []

    for i, (v,c) in enumerate(zip(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'])):
        se = simpson(x=c,y=v)  
        charge_se_lst_cell_id_1.append(se)


    for i, (v,c) in enumerate(zip(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'])):
        se = simpson(x=c,y=v)
        discharge_se_lst_cell_id_1.append(se)
    

    fig, ax = plt.subplots(1,1,figsize=(7.25,6),dpi=150)
    ax.scatter(cell_id_1_cycles[1:],discharge_se_lst_cell_id_1,label='Discharge',color="tab:green",edgecolors='k',marker='o')

    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Specific Energy (Wh/kg-AM)',color="tab:green")
    ax.set_xlim([0,11])
    ax2 = ax.twinx()
    er_lst_cell_id_1 = []
    er_per_cycle_lst_cell_id_1 = []
    for i in range(len(discharge_se_lst_cell_id_1)):
        if i == 0:
            er_lst_cell_id_1.append(100)
        else:
            er_lst_cell_id_1.append(discharge_se_lst_cell_id_1[i]/discharge_se_lst_cell_id_1[0]*100)
            er_per_cycle_lst_cell_id_1.append(discharge_se_lst_cell_id_1[i]/discharge_se_lst_cell_id_1[i-1])
        
    print(er_lst_cell_id_1[-1])
    print(discharge_se_lst_cell_id_1)
    print(np.mean(er_per_cycle_lst_cell_id_1))

    ax2.scatter(cell_id_1_cycles[1:],er_lst_cell_id_1,label='Energy Retention',color="tab:red",edgecolors='k',marker='o')
    ax2.set_ylabel('Energy Retention (%)',color='tab:red')
    ax2.set_ylim([60,105])   
    ax2.spines['right'].set_color('tab:red')
    ax2.spines['left'].set_color('tab:green')
    ax2.tick_params(axis='y', colors='tab:red')
    ax.tick_params(axis='y', colors='tab:green')
    ax.set_ylim([700,1500/(82.5/105)]) #2000
    ax2.hlines(xmin=0,xmax=12,y=90,linestyle='--',color='k',linewidth=2,alpha=1)

    ax2.set_yticks([60,70,80,90,100])
    ax2.set_yticklabels(['60','70','80','90','100'])
    #ax.set_title(f'Cell ID: {cell_id_1}')
    plt.tight_layout()
    fig.savefig(save_folder+cell_id_1+'_energy_retention.png', dpi=300)
