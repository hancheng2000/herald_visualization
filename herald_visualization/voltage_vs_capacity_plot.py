import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
import argparse
import matplotlib.ticker as ticker
from herald_visualization.fancy_plot import voltage_vs_capacity_cycling,plot_multiple_voltage_vs_cycling
from herald_visualization.id_to_path import id_to_path

import json 
with open("plot_setting.json", "r") as f:
    param = json.load(f)
plt.rcParams.update(params)



if __name__ == '__main__':
    from scipy.integrate import simpson
    save_folder = ''#'/scratch/venkvis_root/venkvis/shared_data/herald/cycling_plots/'
    fig, ax = plt.subplots(1,1,figsize=(8,6),dpi=150)
    p = argparse.ArgumentParser(
        description="Plot voltage vs capacity for a given cell ID")
    p.add_argument("cell_id", help="the ID of the cell to plot")
    args = p.parse_args()
    cell_id_1 = args.cell_id
    df = pd.read_csv(id_to_path(cell_id_1))
    print(df.keys())
    unique_cycles = df['full cycle'].unique().astype(int).tolist()
    print(unique_cycles)
    cell_id_1_cycles = list(range(0,21))#unique_cycles[:-1] #[0,1,2,3,5,6,7,8,9,10]#
    
    output_dict = voltage_vs_capacity_cycling(df,cycles=cell_id_1_cycles,plot=False)#1,2,3,4,6,7,8,9,10,11,12,13,14
    fig,ax = plot_multiple_voltage_vs_cycling([output_dict['voltage_discharge_formation']+output_dict['voltage_rest_end_formation']],[output_dict['specific_capacity_discharge_formation']+output_dict['specific_capacity_rest_end_formation']],[0],linestyle='--',color_customize=None,fig=fig,ax=ax,colorbar=False)

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
        fig,ax = plot_multiple_voltage_vs_cycling([voltage_discharge_tot_lst],[np.array(specific_capacity_discharge_tot_lst)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=True,min_cycle=1,max_cycle=max(cell_id_1_cycles),variable_linewidth=True)
        fig,ax=plot_multiple_voltage_vs_cycling([voltage_charge_lst],[np.array(specific_capacity_charge_lst)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=True,min_cycle=1,max_cycle=max(cell_id_1_cycles),variable_linewidth=True)
    #fig,ax = plot_multiple_voltage_vs_cycling(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'],output_dict['cycle_lst'],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=False)
    #fig,ax = plot_multiple_voltage_vs_cycling(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'],output_dict['cycle_lst'],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=True)

    ax.set_xlim([0,601*1.2])
    #ax.set_ylim([0,4.5])
    #ax.legend(fontsize=16,frameon=False,loc='lower left')
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    cbar.set_label(f"Cycle Number")
    plt.tight_layout()
    fig.savefig(save_folder+cell_id_1+'_voltage_vs_capacity.png', dpi=300)

    #df = pd.read_csv("CFx_chunshen_cycle_10_discharge.csv")
    # sc_d = 1/(1/df['Specific capacity (mAh/g-AM)']+1/(0.7*3860))
    # ax.plot(sc_d,df['Voltage (V)'],linestyle='-',color='tab:blue',label='Chunsheng et al. Cycle 10')
    # se_cycle_10_chunsheng = simpson(x=sc_d,y=df['Voltage (V)'])

    # df = pd.read_csv("CFx_chunshen_cycle_10_charge.csv")
    # sc_c = 1/(1/df['Specific capacity (mAh/g-AM)']+1/(0.7*3860))
    # ax.plot(sc_c,df['Voltage (V)'],linestyle='-',color='tab:blue')
    # ce_chunshen_10 = sc_d.iloc[-1]/sc_c.iloc[-1]

   
    # df = pd.read_csv("CFx_chunshen_cycle_8_discharge.csv")
    # sc_d = 1/(1/df['Specific capacity (mAh/g-AM)']+1/(0.7*3860))
    # ax.plot(sc_d,df['Voltage (V)'],linestyle='-',color='tab:red',label='Chunsheng et al. Cycle 8')
    # se_cycle_8_chunsheng = simpson(x=sc_d,y=df['Voltage (V)'])

    # df = pd.read_csv("CFx_chunshen_cycle_8_charge.csv")
    # sc_c = 1/(1/df['Specific capacity (mAh/g-AM)']+1/(0.7*3860))
    # ax.plot(sc_c,df['Voltage (V)'],linestyle='-',color='tab:red')

    df = pd.read_csv("CFx_chunsheng_graphite_LiX_LiF.csv")
    sc_d = df.loc[df['state']==-1]['specific capacity (mAh/g-AM)']
    sc_c = df.loc[df['state']==1]['specific capacity (mAh/g-AM)']
    v_d = df.loc[df['state']==-1]['voltage (V)']
    v_c = df.loc[df['state']==1]['voltage (V)']
    ax.plot(sc_d,v_d,linestyle='-',color='tab:red',label='Chunsheng et al.')
    ax.plot(sc_c,v_c,linestyle='-',color='tab:red')
    chunshen_se = simpson(x=sc_d,y=v_d)
    print('Discharge SE Chunsheng:', chunshen_se)
    #se_cycle_8_chunsheng = simpson(x=sc_d,y=df['Voltage (V)'])
    #ce_chunshen_8 =  sc_d.iloc[-1]/sc_c.iloc[-1]


    #print('Avg ER Chunsheng:', np.sqrt(se_cycle_10_chunsheng/se_cycle_8_chunsheng))
    formation_cycle_se = simpson(x=output_dict['specific_capacity_discharge_formation'],y=output_dict['voltage_discharge_formation'])
    #ax.set_title(f'Cell ID: {cell_id_1}')
    ax.set_xlim([0,601*1.2])
    #ax.set_ylim([0,4.5])
    ax.legend(fontsize=16,frameon=False,loc='lower left')
    plt.tight_layout()
    fig.savefig(save_folder+cell_id_1+'_voltage_vs_capacity.png', dpi=300)

    discharge_se_lst_cell_id_1= []
    charge_se_lst_cell_id_1 = []
    discharge_c_lst = []
    charge_c_lst = []
    
    for i, (v,c) in enumerate(zip(output_dict['voltage_charge_lst'],output_dict['specific_capacity_charge_lst'])):
        se = simpson(x=c,y=v)  
        charge_se_lst_cell_id_1.append(se)
        charge_c_lst.append(c[-1])



    for i, (v,c) in enumerate(zip(output_dict['voltage_discharge_lst'],output_dict['specific_capacity_discharge_lst'])):
        se = simpson(x=c,y=v)
        discharge_se_lst_cell_id_1.append(se)
        discharge_c_lst.append(c[-1])




    fig, ax = plt.subplots(1,1,figsize=(7.25,6),dpi=150)
    ax.scatter(cell_id_1_cycles[1:],discharge_se_lst_cell_id_1,label=r'In-house',color="tab:green",edgecolors='k',marker='o')
    print('Chunsheng SE:',se)
    ax.axhline(y=chunshen_se,linestyle='--',color='tab:red',label=r'Chunsheng et al.')
    #ax.scatter([8,10],[se_cycle_8_chunsheng,se_cycle_10_chunsheng],label=r'Chunsheng et al. CF$_x$',color="tab:green",edgecolors='k',marker='^')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Specific Energy (Wh/kg-AM)')
    ax.set_xlim([0,11])

    ax.set_ylim([700,1600])
    ax.legend()

    #ax2 = ax.twinx()
    er_lst_cell_id_1 = []
    er_per_cycle_lst_cell_id_1 = []
    ce_per_cycle_lst_cell_id_1 = []
    for i in range(len(discharge_se_lst_cell_id_1)):
        ce_per_cycle_lst_cell_id_1.append(discharge_c_lst[i]/charge_c_lst[i])
        if i == 0:
            er_lst_cell_id_1.append(100)
        else:
            er_lst_cell_id_1.append(discharge_se_lst_cell_id_1[i]/discharge_se_lst_cell_id_1[0]*100)
            er_per_cycle_lst_cell_id_1.append(discharge_se_lst_cell_id_1[i]/discharge_se_lst_cell_id_1[i-1])
    
    print(er_lst_cell_id_1[-1])
    print(discharge_se_lst_cell_id_1)
    print('Avg. ER cycle over cycle: ',np.mean(er_per_cycle_lst_cell_id_1))

   #save ce_per_cycle_lst_cell_id_1 to a csv file
    ce_per_cycle_lst_cell_id_1 = np.array(ce_per_cycle_lst_cell_id_1)*100
    ce_dict = {'cycle':cell_id_1_cycles[1:],'CE':ce_per_cycle_lst_cell_id_1}
    ce_df = pd.DataFrame(ce_dict)
    ce_df.to_csv(save_folder+cell_id_1+'_ce_per_cycle.csv',index=False)

    # ax2.scatter(cell_id_1_cycles[1:],np.array(ce_per_cycle_lst_cell_id_1)*100,color="tab:red",edgecolors='k',marker='o')
    # ax2.scatter([8,10],np.array([ce_chunshen_8,ce_chunshen_10])*100,color="tab:red",edgecolors='k',marker='^')
    # ax2.set_ylabel('Coulombic Efficiency (%)',color='tab:red')
    # ax2.set_ylim([60,105])   
    # ax2.spines['right'].set_color('tab:red')
    # ax2.spines['left'].set_color('tab:green')
    # ax2.tick_params(axis='y', colors='tab:red')
    # ax.tick_params(axis='y', colors='tab:green')
    # ax.legend(fontsize=16,frameon=False,loc='lower left')
    # ax.set_ylim([700,1500/(82.5/105)]) #2000
    # ax2.hlines(xmin=0,xmax=12,y=90,linestyle='--',color='k',linewidth=2,alpha=1)

    # ax2.set_yticks([60,70,80,90,100])
    # ax2.set_yticklabels(['60','70','80','90','100'])
    # #ax.set_title(f'Cell ID: {cell_id_1}')
    plt.tight_layout()
    fig.savefig(save_folder+cell_id_1+'_se_ce.png', dpi=300)

    fig, ax = plt.subplots(1,1,figsize=(8,6),dpi=150)
    ax.scatter(cell_id_1_cycles[1:],discharge_c_lst,marker='o',label='Discharge',facecolors='white',edgecolors="tab:red")
    ax.scatter(cell_id_1_cycles[1:],charge_c_lst,marker='o',label='Charge',color="tab:red")

    ax2 = ax.twinx()
    ax2.scatter(cell_id_1_cycles[1:],np.array(ce_per_cycle_lst_cell_id_1),color="tab:green",edgecolors='k',marker='^')
    ax2.set_ylabel('Coulombic Efficiency (%)',color='tab:green')
    ax2.set_ylim([60,105])
    ax2.spines['right'].set_color('tab:green')
    ax2.spines['left'].set_color('tab:red')
    ax2.tick_params(axis='y', colors='tab:green')
    ax.tick_params(axis='y', colors='tab:red')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Specific Capacity (mAh/g-AM)',color='tab:red')
    ax.legend(loc='lower left',fontsize=16,frameon=True)
    plt.tight_layout()
    fig.savefig(save_folder+cell_id_1+'_ce.png', dpi=300)

