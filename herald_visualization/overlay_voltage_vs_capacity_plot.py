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
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.integrate import simpson
    fig_vc, ax_vc = plt.subplots(1,1,figsize=(9,6),dpi=150)
    fig_er, ax_er = plt.subplots(1,1,figsize=(9,6),dpi=150)
    ax2_er = ax_er.twinx()
    cell_id_1 = '117A'
    cell_id_1_linestyle = '-'
    cell_id_1_marker = 'o'
    cell_id_2 = '115I'
    cell_id_2_linestyle = ':'
    cell_id_2_marker = 's'

    df_cell_id_1 = pd.read_csv(id_to_path(cell_id_1))
    #df_cell_id_1 = df_cell_id_1[df_cell_id_1['full cycle'] <= 10].copy()
    unique_cycles_cell_id_1 = df_cell_id_1['full cycle'].unique().astype(int).tolist()

    df_cell_id_2 = pd.read_csv(id_to_path(cell_id_2))
    #df_cell_id_2 = df_cell_id_2[df_cell_id_2['full cycle'] <= 10].copy()
    unique_cycles_cell_id_2 = df_cell_id_2['full cycle'].unique().astype(int).tolist()
    
    max_cycle = max(max(unique_cycles_cell_id_1),max(unique_cycles_cell_id_2))
    same_cycle_lst = range(1,max_cycle+1)

    
    cell_id_1_cycles = unique_cycles_cell_id_1[:-1]
    # cell_id_1_cycles = [0,1,2,3,4,5,6]
    output_dict = voltage_vs_capacity_cycling(df_cell_id_1,cycles=cell_id_1_cycles,plot=False)#1,2,3,4,6,7,8,9,10,11,12,13,14
    fig_vc,ax_vc = plot_multiple_voltage_vs_cycling([output_dict['voltage_discharge_formation']+output_dict['voltage_rest_end_formation']],[output_dict['specific_capacity_discharge_formation']+output_dict['specific_capacity_rest_end_formation']],[0],linestyle='--',color_customize=None,fig=fig_vc,ax=ax_vc,colorbar=False)
    fig_vc,ax_vc = plot_multiple_voltage_vs_cycling(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'],same_cycle_lst,linestyle=cell_id_1_linestyle,color_customize=None,fig=fig_vc,ax=ax_vc,colorbar=False)
    fig_vc,ax_vc = plot_multiple_voltage_vs_cycling(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'],same_cycle_lst,linestyle=cell_id_1_linestyle,color_customize=None,fig=fig_vc,ax=ax_vc,colorbar=False)
    
    formation_cycle_se = simpson(x=output_dict['specific_capacity_discharge_formation'],y=output_dict['voltage_discharge_formation'])
    ax_vc.set_xlim([0,601*1.2])
    ax_vc.set_ylim([0,4.5])
    norm = Normalize(vmin=output_dict['cycle_lst'][0], vmax=same_cycle_lst[-1])
    sm = ScalarMappable(cmap='coolwarm_r', norm=norm)
    cbar = fig_vc.colorbar(sm, ax=ax_vc)
    cbar.set_label(f"{cell_id_1} Cycle Number")
        
    #fig_vc.savefig(save_folder+cell_id_1+'_voltage_vs_capacity.png', dpi=300)

    discharge_se_lst_cell_id_1= []
    charge_se_lst_cell_id_1 = []

    for i, (v,c) in enumerate(zip(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'])):
        se = simpson(x=c,y=v)  
        charge_se_lst_cell_id_1.append(se)
    for i, (v,c) in enumerate(zip(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'])):
        se = np.round(simpson(x=c,y=v),decimals=4)
        discharge_se_lst_cell_id_1.append(se)   
    fig_current, ax_current = plt.subplots(1,1,figsize=(7.25,6),dpi=150)
    ax_current_2 = ax_current.twinx()
    last_current_lst = []
    last_charge_capacity_lst = []
    for i,current in enumerate(output_dict['specific_current_charge_lst']):
        last_current_lst.append(current[-1])
        last_charge_capacity_lst.append(output_dict['specific_capacity_charge_lst'][i][-1])
    
    ax_current.scatter(cell_id_1_cycles[1:],last_current_lst,color="tab:blue",edgecolors='k',marker=cell_id_1_marker,label=f'{cell_id_1}')
    ax_current_2.scatter(cell_id_1_cycles[1:],last_charge_capacity_lst,color="tab:orange",edgecolors='k',marker=cell_id_1_marker)


    
    er_lst_cell_id_1 = []
    er_per_cycle_lst_cell_id_1 = []
    for i in range(len(discharge_se_lst_cell_id_1)):
        if i == 0:
            er_lst_cell_id_1.append(100)
        else:
            er_lst_cell_id_1.append(np.round(discharge_se_lst_cell_id_1[i]/discharge_se_lst_cell_id_1[0]*100,decimals=2))
            er_per_cycle_lst_cell_id_1.append(np.round(discharge_se_lst_cell_id_1[i]/discharge_se_lst_cell_id_1[i-1],decimals=2))

    print(np.array(er_lst_cell_id_1))
    print(np.array(discharge_se_lst_cell_id_1))
    #print(np.mean(er_per_cycle_lst_cell_id_1))
    
    
    ax_er.scatter(cell_id_1_cycles[1:],discharge_se_lst_cell_id_1,color="tab:green",edgecolors='k',marker=cell_id_1_marker)
    ax2_er.scatter(cell_id_1_cycles[1:],er_lst_cell_id_1,color="tab:red",edgecolors='k',marker=cell_id_1_marker)


    cell_id_2_cycles = unique_cycles_cell_id_2[:-1]
    output_dict = voltage_vs_capacity_cycling(df_cell_id_2,cycles=cell_id_2_cycles,plot=False)#1,2,3,4,6,7,8,9,10,11,12,13,14
    fig_vc,ax_vc = plot_multiple_voltage_vs_cycling([output_dict['voltage_discharge_formation']+output_dict['voltage_rest_end_formation']],[output_dict['specific_capacity_discharge_formation']+output_dict['specific_capacity_rest_end_formation']],[0],linestyle='--',color_customize=None,color_gradient_customize='coolwarm',fig=fig_vc,ax=ax_vc,colorbar=False)
    fig_vc,ax_vc = plot_multiple_voltage_vs_cycling(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'],same_cycle_lst,linestyle=cell_id_2_linestyle,color_customize=None,fig=fig_vc,ax=ax_vc,color_gradient_customize='coolwarm',colorbar=False)
    fig_vc,ax_vc = plot_multiple_voltage_vs_cycling(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'],same_cycle_lst,linestyle=cell_id_2_linestyle,color_customize=None,fig=fig_vc,ax=ax_vc,color_gradient_customize='coolwarm',colorbar=False)

    formation_cycle_se = simpson(x=output_dict['specific_capacity_discharge_formation'],y=output_dict['voltage_discharge_formation'])
    ax_vc.set_xlim([0,601*1.2])
    ax_vc.set_ylim([0,4.5])
    plt.tight_layout()
    legend_elements = [plt.Line2D([0], [0], color='k', label=f'{cell_id_1}',linestyle=cell_id_1_linestyle),
                       plt.Line2D([0], [0], color='k', label=f'{cell_id_2}',linestyle=cell_id_2_linestyle)]

    norm = Normalize(vmin=output_dict['cycle_lst'][0], vmax=same_cycle_lst[-1])
    sm = ScalarMappable(cmap='coolwarm', norm=norm)
    cbar = fig_vc.colorbar(sm, ax=ax_vc)
    cbar.set_label(f"{cell_id_2} Cycle Number")

    ax_vc.legend(handles=legend_elements, loc='lower left', fontsize=16)
    fig_vc.savefig('/scratch/venkvis_root/venkvis/shared_data/herald/voltage-overlay/'+cell_id_1+'_'+cell_id_2+'_voltage_vs_capacity.png', dpi=300, bbox_inches='tight')

    discharge_se_lst_cell_id_2= []
    charge_se_lst_cell_id_2 = []

    for i, (v,c) in enumerate(zip(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'])):
        se = simpson(x=c,y=v)  
        charge_se_lst_cell_id_2.append(se)
    for i, (v,c) in enumerate(zip(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'])):
        se = np.round(simpson(x=c,y=v),decimals=4)
        discharge_se_lst_cell_id_2.append(se)

    last_current_lst = []
    last_charge_capacity_lst = []
    for i,current in enumerate(output_dict['specific_current_charge_lst']):
        last_current_lst.append(current[-1])
        last_charge_capacity_lst.append(output_dict['specific_capacity_charge_lst'][i][-1])

    ax_current.scatter(cell_id_2_cycles[1:],last_current_lst,color="tab:blue",edgecolors='k',marker=cell_id_2_marker,label=f'{cell_id_2}')
    ax_current_2.scatter(cell_id_2_cycles[1:],last_charge_capacity_lst,color="tab:orange",edgecolors='k',marker=cell_id_2_marker)

    ax_current.set_xlabel('Cycle Number')
    ax_current.set_ylabel('Last Current (mA/g-AM)',color="tab:blue")
    ax_current_2.set_ylabel('Last Charge Capacity (mAh/g-AM)',color="tab:orange")
    ax_current_2.spines['right'].set_color('tab:orange')
    ax_current_2.spines['left'].set_color('tab:blue')
    ax_current.set_xlim([0,11])
    ax_current.legend(loc='upper right',frameon=True,borderpad=0)
    plt.tight_layout()
    fig_current.savefig('/scratch/venkvis_root/venkvis/shared_data/herald/voltage-overlay/'+cell_id_1+'_'+cell_id_2+'_last_current.png', dpi=300, bbox_inches='tight')



    er_lst_cell_id_2 = []
    er_per_cycle_lst_cell_id_2 = []
    for i in range(len(discharge_se_lst_cell_id_2)):
        if i == 0:
            er_lst_cell_id_2.append(100)
        else:
            er_lst_cell_id_2.append(np.round(discharge_se_lst_cell_id_2[i]/discharge_se_lst_cell_id_2[0]*100,decimals=2))
            er_per_cycle_lst_cell_id_2.append(np.round(discharge_se_lst_cell_id_2[i]/discharge_se_lst_cell_id_2[i-1],decimals=2))



    print(np.array(er_lst_cell_id_2))
    print(np.array(discharge_se_lst_cell_id_2))
    #print(np.mean(er_per_cycle_lst_cell_id_2))

    ax_er.scatter(cell_id_2_cycles[1:],discharge_se_lst_cell_id_2,color="tab:green",edgecolors='k',marker=cell_id_2_marker)
    ax2_er.scatter(cell_id_2_cycles[1:],er_lst_cell_id_2,color="tab:red",edgecolors='k',marker=cell_id_2_marker)

    ax_er.set_xlabel('Cycle Number')
    ax_er.set_ylabel('Specific Energy (Wh/kg-AM)',color="tab:green")
    #ax_er.set_xlim([0,11])

    ax2_er.set_ylabel('Energy Retention (%)',color='tab:red')
    ax2_er.set_ylim([60,110])   
    ax2_er.spines['right'].set_color('tab:red')
    ax2_er.spines['left'].set_color('tab:green')
    ax2_er.tick_params(axis='y', colors='tab:red')
    ax_er.tick_params(axis='y', colors='tab:green')
    ax_er.set_ylim([700,1500/(82.5/105)]) #2000
    #ax2_er.hlines(xmin=0,xmax=12,y=90,linestyle='--',color='k',linewidth=2,alpha=1)

    ax2_er.set_yticks([60,70,80,90,100,110])
    ax2_er.set_yticklabels(['60','70','80','90','100','110'])

    legend_elements = [
        plt.Line2D([0], [0], linestyle='-', color='w', marker=cell_id_1_marker,markerfacecolor='k',label=f'{cell_id_1}'),
        plt.Line2D([0], [0], linestyle='-', color='w',marker=cell_id_2_marker,markerfacecolor='k',label=f'{cell_id_2}'),
    ]

    ax_er.legend(handles=legend_elements,loc='lower left',frameon=False,borderpad=0)
    plt.tight_layout()
    fig_er.savefig('/scratch/venkvis_root/venkvis/shared_data/herald/voltage-overlay/'+cell_id_1+'_'+cell_id_2+'_energy_retention.png', dpi=300, bbox_inches='tight')
    #ax.set_title(f'Cell ID: {cell_id_1}')
    #plt.tight_layout()
    #fig.savefig(save_folder+cell_id_1+'_energy_retention.png', dpi=300)

    