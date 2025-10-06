import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
import argparse
# import herald_visualization.echem as ec
# from herald_visualization.mpr2csv import cycle_mpr2csv
# from herald_visualization.plot import plot_cycling, plot_gitt, parse_cycle_csv, plot_cycling_plotly
from herald_visualization.fancy_plot import voltage_vs_capacity_cycling,plot_multiple_voltage_vs_cycling
from herald_visualization.mpr2csv import cycle_mpr2csv

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
    save_folder = '/scratch/venkvis_root/venkvis/shared_data/herald/cycling_plots/'
    p = argparse.ArgumentParser(
        description="Plot voltage vs capacity for a given cell ID")
    p.add_argument("cell_id", help="the ID of the cell to plot")
    args = p.parse_args()
    cell_id_1 = args.cell_id
    from scipy.integrate import simpson
    save_folder = '/scratch/venkvis_root/venkvis/shared_data/herald/cycling_plots/'
    fig, ax = plt.subplots(1,1,figsize=(11,6),dpi=300)    
    print(id_to_path(cell_id_1))
    from herald_visualization.mpr2csv import cycle_mpr2csv
    cell_id = cell_id_1
    full_path = id_to_path(cell_id).split('/outputs')[0]
    df=cycle_mpr2csv(full_path)
    #df = pd.read_csv(id_to_path(cell_id_1))
    unique_cycles = df['full cycle'].unique().astype(int).tolist()
    cell_id_1_cycles = unique_cycles[:-1]
    print(cell_id_1_cycles)
    output_dict = voltage_vs_capacity_cycling(df,cycles=cell_id_1_cycles,plot=False)
    if len(output_dict['cycle_lst']) == 0:
        print(f"No cycles found for {cell_id_1}")
        exit()
    # fig,ax = plot_multiple_voltage_vs_cycling([output_dict['voltage_discharge_formation']+output_dict['voltage_rest_end_formation']],[output_dict['specific_capacity_discharge_formation']+output_dict['specific_capacity_rest_end_formation']],[0],linestyle='--',color_customize=None,fig=fig,ax=ax,colorbar=False)        

    for i, cycle in enumerate(output_dict['cycle_lst']):#[-2:-1]
        specific_capacity_discharge = output_dict['specific_capacity_discharge_lst'][i]
        # where specific_capacity_discharge is decreases, add to the previous value
        specific_capacity_discharge = np.array(specific_capacity_discharge)
        voltage_discharge = np.array(output_dict['voltage_discharge_lst'][i])
        stop_power_indices = np.where(np.diff(specific_capacity_discharge) < -5)[0]
        # remove index in specific_capacity_discharge and voltage_discharge where index is in stop_power_indices+1
        while len(stop_power_indices) > 3 or (np.any(np.diff(stop_power_indices) <= 4) and np.all(np.diff(stop_power_indices) > 0)):
            specific_capacity_discharge = specific_capacity_discharge[[j for j in range(len(specific_capacity_discharge)) if j not in stop_power_indices+1]]
            voltage_discharge = voltage_discharge[[j for j in range(len(voltage_discharge)) if j not in stop_power_indices+1]]
            stop_power_indices = np.where(np.diff(specific_capacity_discharge) < -5)[0]
        x_translation_values = specific_capacity_discharge[stop_power_indices]
        for ii, index in enumerate(stop_power_indices):
            specific_capacity_discharge[index+1:] += x_translation_values[ii]
        specific_capacity_charge = output_dict['specific_capacity_charge_lst'][i]
        voltage_charge = output_dict['voltage_charge_lst'][i]
        if cycle == max(cell_id_1_cycles):
            #fig,ax = plot_multiple_voltage_vs_cycling([voltage_charge],[specific_capacity_discharge[-1]-np.array(specific_capacity_charge)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=True,min_cycle=1,max_cycle=max(cell_id_1_cycles))
            print('Plotting charge curve for last cycle')
            fig,ax = plot_multiple_voltage_vs_cycling([voltage_charge],[np.array(specific_capacity_charge)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles))
            fig,ax = plot_multiple_voltage_vs_cycling([voltage_discharge],[np.array(specific_capacity_discharge)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles))
        else:
            fig,ax = plot_multiple_voltage_vs_cycling([voltage_discharge],[specific_capacity_discharge],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles))
            fig,ax = plot_multiple_voltage_vs_cycling([voltage_charge],[specific_capacity_charge],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles))
        
        output_dict['voltage_discharge_lst'][i] = voltage_discharge
        output_dict['specific_capacity_discharge_lst'][i] = specific_capacity_discharge

    # add ax2 for power profile
    specific_capacity_discharge = np.array(output_dict['specific_capacity_discharge_lst'][-1])
    voltage_discharge = np.array(output_dict['voltage_discharge_lst'][-1])
    stop_power_indices = np.where(np.diff(voltage_discharge) > 0.02)[0]
    print(stop_power_indices)
    stop_power_capacity = specific_capacity_discharge[stop_power_indices].max()
    capacities = [0, stop_power_capacity, np.max(specific_capacity_discharge[-1])]
    powers = [1.5, 1.5, 0.5]
    ax2 = ax.twinx()
    ax2.step(capacities, powers, linestyle='--', color='k')
    ax2.set_ylabel('Power (kW/kg-AM)', color='k')
    ax2.set_xlabel('Specific Capacity (mAh/g-AM)')
    ax2.set_yticks([0, 0.5,1.0,1.5])
    ax2.tick_params(axis='y', colors='k')

    # formation_cycle_se = simpson(x=output_dict['specific_capacity_discharge_formation'],y=output_dict['voltage_discharge_formation'])
    #ax.set_title(f'Cell ID: {cell_id_1}')
    # ax.set_xlim([0,601*1.2])
    # ax.set_xlim([0,40])
    ax.set_ylim([0,4.5])
    # ax.set_xlim([0,50])
    ax.set_ylabel('Voltage (V)',color='k')

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.ticker as ticker
    norm = Normalize(vmin=1, vmax=max(cell_id_1_cycles))
    sm = ScalarMappable(cmap='coolwarm_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2,pad=0.15)
    cbar.set_label("Cycle Number")
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) 


    # ax.set_ylim([3.0,4.0])
    plt.tight_layout()
    # add title
    #ax.set_title(f'Cell ID: {cell_id_1}')

    fig.savefig(save_folder+cell_id_1+'_voltage_vs_capacity.png', dpi=100, bbox_inches='tight')

    discharge_se_lst_cell_id_1= []
    charge_se_lst_cell_id_1 = []
    discharge_sc_lst_cell_id_1 = []
    charge_sc_lst_cell_id_1 = []

    for i, (v,c) in enumerate(zip(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'])):
        se = simpson(x=c,y=v)  
        charge_se_lst_cell_id_1.append(se)

    
    for i, (v,c) in enumerate(zip(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'])):
        se = simpson(x=c,y=v)
        discharge_se_lst_cell_id_1.append(se)
        discharge_sc_lst_cell_id_1.append(max(c))
    discharge_se_lst_cell_id_1 = np.array(discharge_se_lst_cell_id_1)
    discharge_sc_lst_cell_id_1 = np.array(discharge_sc_lst_cell_id_1)
    print(len(discharge_se_lst_cell_id_1))
    print(len(discharge_sc_lst_cell_id_1))
    print(cell_id_1_cycles)
    fig, ax = plt.subplots(1,1,figsize=(7.25,6),dpi=100)
    ax.scatter(output_dict['cycle_lst'],discharge_se_lst_cell_id_1,label='Discharge',color="tab:blue",edgecolors='k',marker='o')

    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Specific Energy (Wh/kg-AM)',color="tab:blue")
    # ax.set_xlim([0,11])
    # ax2 = ax.twinx()
    er_lst_cell_id_1 = []
    er_per_cycle_lst_cell_id_1 = []
    for i in range(len(discharge_se_lst_cell_id_1)):
        if i == 0:
            er_lst_cell_id_1.append(100)
            er_per_cycle_lst_cell_id_1.append(100)
        else:
            er_lst_cell_id_1.append(discharge_se_lst_cell_id_1[i]/discharge_se_lst_cell_id_1[0]*100)
            er_per_cycle_lst_cell_id_1.append(discharge_se_lst_cell_id_1[i]/discharge_se_lst_cell_id_1[i-1]*100)
        
    print(er_lst_cell_id_1[-1])
    print('discharge_se',discharge_se_lst_cell_id_1)
    print(np.mean(er_per_cycle_lst_cell_id_1))

    # ax2.scatter(output_dict['cycle_lst'],er_lst_cell_id_1,label='Energy Retention',color="tab:orange",edgecolors='k',marker='o')
    # ax2.scatter(output_dict['cycle_lst'],er_per_cycle_lst_cell_id_1,label='Energy Retention',color="tab:orange",edgecolors='k',marker='o')
    # ax.scatter(output_dict['cycle_lst'],er_per_cycle_lst_cell_id_1,label='Energy Retention',color="tab:blue",edgecolors='k',marker='o')
    # ax2.set_ylabel('Energy Retention (%)',color="tab:orange")
    # ax2.set_xlim([0, int(max(cell_id_1_cycles))+1])
    # only tick the integer x and tick no more than 5
    print(max(cell_id_1_cycles))
    # n_ticks = 5 if max(cell_id_1_cycles) > 10 else max(cell_id_1_cycles)
    # ax2.set_xticks(np.linspace(1, (int(max(cell_id_1_cycles))+1)//5*5+1, 6).astype(int))
    # ax.set_ylim([60,105])   
    # ax2.spines['right'].set_color('tab:orange')
    # ax2.spines['left'].set_color('tab:blue')
    # ax2.tick_params(axis='y', colors='tab:orange')
    # ax.tick_params(axis='y', colors='tab:blue')
    # # ax.set_ylim([400,1500/(82.5/105)]) #2000
    # ax.set_ylim([750, 1850])
    # ax2.set_ylim([80,None])
    # ax2.axhline(y=80,linestyle='--',color='orange',linewidth=2,alpha=1)

    plt.tight_layout()
    # add title
    ax.set_title(f'Cell ID: {cell_id_1}')
    fig.savefig(save_folder+cell_id_1+'_energy_retention.png', dpi=100, bbox_inches='tight')

    if len(discharge_se_lst_cell_id_1) > 10:
        max_cycle = 10
        overall_er = discharge_se_lst_cell_id_1[9]/discharge_se_lst_cell_id_1[0]*100
    else:
        max_cycle = len(discharge_se_lst_cell_id_1)
        overall_er = discharge_se_lst_cell_id_1[-1]/discharge_se_lst_cell_id_1[0]*100
    # if overall_er < 80:
    #     print(f"Energy retention for {cell_id_1} @ {max_cycle} cycles is below 80%: {overall_er:.2f}%")
    #     os.remove(save_folder+cell_id_1+'_energy_retention.png')
    #     os.remove(save_folder+cell_id_1+'_voltage_vs_capacity.png')


    # fig, ax = plt.subplots(1,1,figsize=(7.25,6),dpi=100)
    # ax.scatter(output_dict['cycle_lst'],discharge_sc_lst_cell_id_1,label='Discharge',color="tab:blue",edgecolors='k',marker='o')

    # ax.set_xlabel('Cycle Number')
    # ax.set_ylabel('Specific Capacity (mAh/kg-AM)',color="tab:blue")
    # ax.set_xlim([0,11])
    # ax2 = ax.twinx()
    # cr_lst_cell_id_1 = []
    # cr_per_cycle_lst_cell_id_1 = []
    # for i in range(len(discharge_sc_lst_cell_id_1)):
    #     if i == 0:
    #         cr_lst_cell_id_1.append(100)
    #         cr_per_cycle_lst_cell_id_1.append(100)
    #     else:
    #         cr_lst_cell_id_1.append(discharge_sc_lst_cell_id_1[i]/discharge_sc_lst_cell_id_1[0]*100)
    #         cr_per_cycle_lst_cell_id_1.append(discharge_sc_lst_cell_id_1[i]/discharge_sc_lst_cell_id_1[i-1]*100)
        
    # print(cr_lst_cell_id_1[-1])
    # print(discharge_sc_lst_cell_id_1)
    # print(np.mean(cr_per_cycle_lst_cell_id_1))

    # # ax2.scatter(output_dict['cycle_lst'],er_lst_cell_id_1,label='Energy Retention',color="tab:orange",edgecolors='k',marker='o')
    # ax2.scatter(output_dict['cycle_lst'],cr_per_cycle_lst_cell_id_1,label='Energy Retention',color="tab:orange",edgecolors='k',marker='o')
    # ax2.set_ylabel('Capacity Retention (%)',color="tab:orange")
    # # ax2.set_xlim([0, 50])
    # # ax2.set_ylim([60,105])   
    # ax2.spines['right'].set_color('tab:orange')
    # ax2.spines['left'].set_color('tab:blue')
    # ax2.tick_params(axis='y', colors='tab:orange')
    # ax.tick_params(axis='y', colors='tab:blue')
    # # ax.set_ylim([400,1500/(82.5/105)]) #2000
    # # ax.set_ylim([150, 550])
    # ax2.axhline(y=0.998*100,linestyle='--',color='r',linewidth=2,alpha=1)

    # # ax2.set_yticks([60,70,80,90,100])
    # # ax2.set_yticklabels(['60','70','80','90','100'])
    # #ax.set_title(f'Cell ID: {cell_id_1}')
    # plt.tight_layout()
    # fig.savefig(save_folder+cell_id_1+'_energy_capacity_retention.png', dpi=100)

    # discharge_sc_lst_cell_id_1 = np.array(discharge_sc_lst_cell_id_1)
    # discharge_se_lst_cell_id_1 = np.array(discharge_se_lst_cell_id_1)
    # avg_voltage = discharge_se_lst_cell_id_1/discharge_sc_lst_cell_id_1
    # print(avg_voltage)
    # print(np.diff(avg_voltage))