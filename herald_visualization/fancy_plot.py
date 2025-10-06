import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re

def voltage_vs_capacity_cycling(df,cycles=None,plot=False):
    if plot == True:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6))

    voltage_discharge_lst=[]
    voltage_discharge_rest_lst=[]
    voltage_charge_lst = []
    voltage_charge_rest_lst = []
    specific_capacity_discharge_lst=[]
    specific_capacity_discharge_rest_lst=[]
    specific_capacity_discharge_with_rest_lst = []
    specific_capacity_charge_lst = []
    specific_capacity_charge_rest_lst = []
    time_discharge_lst=[]
    time_discharge_rest_lst=[]
    time_discharge_with_rest_lst = []
    time_charge_lst = []
    time_charge_rest_lst = []
    specific_current_discharge_lst=[]
    areal_current_discharge_lst=[]
    specific_current_charge_lst=[]
    areal_current_charge_lst=[]
    cycle_lst=[]
    voltage_discharge_tot_lst=[]
    specific_capacity_discharge_tot_lst =[]
    specific_current_avg_discharge_lst = []
    specific_current_avg_charge_lst = []
    areal_current_avg_discharge_lst = []
    areal_current_avg_charge_lst = []
    specific_power_discharge_lst = []
    specific_power_charge_lst = []

    time_col_name = 'time/s'
    if 'Time' in df.columns:
        time_col_name = 'Time'

    df_unique_cycle = np.unique(df['full cycle'])
    if cycles == None:
        formation_cycle = 0
        start_cycle = df_unique_cycle[0]
        end_cycle = df_unique_cycle[-1]
        if end_cycle >= 10:
            end_cycle = 10
        cycles = [start_cycle,end_cycle]
    elif cycles[0] == -1:
        end_cycle = df_unique_cycle[-1]
        if end_cycle >= 10:
            end_cycle = 10
        cycles = [end_cycle]
        formation_cycle = None
    elif cycles[0] == 0:
        formation_cycle = cycles[0]
        cycles = cycles[0:]
    elif cycles[0] != 0:
        formation_cycle = None
    if formation_cycle != None:
        df0 = df[df['full cycle']==formation_cycle]
        df_discharge_running = df0[df0['state']==-1]
        df_discharge_resting = df0[df0['state']==0]
        df_discharge_start_formation_resting_idx = df0.index[0]
        if len(df_discharge_running) == 0:
            formation_cycle= None
        else:
            df_discharge_start_formation_running_idx = df_discharge_running.index[0]
            df_discharge_end_formation_running_idx = df_discharge_running.index[-1]
            df_discharge_end_formation_resting_idx = df0.index[-1]
            specific_capacity_discharge_formation = df_discharge_running['Specific Capacity Total AM']
            df_discharge_resting_formation_end = df0.loc[df_discharge_end_formation_running_idx:df_discharge_end_formation_resting_idx]#df_discharge_resting[df_discharge_resting['Specific Capacity Total AM']!=0]['Specific Capacity Total AM']
            df_discharge_resting_formation_start = df0.loc[df_discharge_start_formation_resting_idx:df_discharge_start_formation_running_idx]
            # print(specific_capacity_rest_end_formation.index)

            voltage_discharge_formation = df_discharge_running['Voltage']
            voltage_rest_start_formation = df_discharge_resting_formation_start['Voltage']
            voltage_rest_end_formation = df_discharge_resting_formation_end['Voltage']

            time_discharge_formation = df_discharge_running[time_col_name]
            time_rest_formation = df_discharge_resting[time_col_name]

            specific_current_discharge_formation = df_discharge_running['Specific Current Total AM']
            areal_current_discharge_formation = df_discharge_running['Areal Current']
            specific_capacity_rest_end_formation = df_discharge_resting_formation_end['Specific Capacity Total AM'].copy()
            
            if len(specific_capacity_rest_end_formation) >= 2:
                if specific_capacity_rest_end_formation.iloc[1] == 0:
                    specific_capacity_rest_end_formation.iloc[:]=specific_capacity_discharge_formation.iloc[-1]

            if plot == True:
                ax.plot(specific_capacity_discharge_formation,voltage_discharge_formation,label=f'Cycle {formation_cycle} Discharge')
                ax.plot(specific_capacity_rest_end_formation,voltage_rest_end_formation,label=f'Cycle {formation_cycle} Rest')
    for i, cycle_i in enumerate(cycles):
        print(f'Processing cycle {cycle_i}...')
        # time= df['time/s'][df['full cycle'] == cycle_i]
        # time_0 = time.iloc[0]
        df_i = df[df['full cycle'] == cycle_i]
        df_i_half_cycle = df_i['half cycle'].unique()
        df_i_charge = df_i[df_i['half cycle']==df_i_half_cycle[0]]
        df_charge_running = df_i_charge[df_i_charge['state']==1]
        df_charge_resting = df_i_charge[df_i_charge['state']==0]
        # remove the first row if it is much smaller than the second row
        if len(df_charge_resting['Voltage']>2):
            df_charge_resting = df_charge_resting[1:]
        if len(df_i_half_cycle) == 1:
            print(f'Only one half cycle for cycle {cycle_i}, skipping...')
            continue
        df_i_discharge = df_i[df_i['half cycle']==df_i_half_cycle[1]]
        df_discharge_running = df_i_discharge[df_i_discharge['state']==-1]
        df_discharge_resting = df_i_discharge[df_i_discharge['state']==0]
        if len(df_discharge_resting) == 0:
            print(f'No resting data for cycle {cycle_i}, skipping...')
            continue
        df_discharge_resting_indx = df_discharge_resting.index

        df_discharge_resting_indx_diff = np.diff(df_discharge_resting_indx)
        df_discharge_resting_indx_diff = np.where(df_discharge_resting_indx_diff != 1)[0]

        if len(df_discharge_resting_indx_diff) > 0:
            df_discharge_resting_start_indx = df_discharge_resting_indx[df_discharge_resting_indx_diff[-1]]
            df_discharge_resting_end_indx = df_discharge_resting_indx[-1]
        elif len(df_discharge_resting_indx) != 0:
            df_discharge_resting_start_indx = df_discharge_resting_indx[0]
            df_discharge_resting_end_indx = df_discharge_resting_indx[-1]
        else:
            df_discharge_resting_start_indx = df_discharge_running.index[-1]
            df_discharge_resting_end_indx = df_discharge_running.index[-1]

        #only include continuous index 
        df_discharge_resting = df_discharge_resting.loc[df_discharge_resting_start_indx:df_discharge_resting_end_indx]

        # voltage_discharge = voltage.loc[discharge_start_idx:discharge_end_idx]
        # time_discharge = time.loc[discharge_start_idx:]
        #ax[0].plot(time_discharge-time_0,voltage_discharge,label=f'Cycle {i}')
        # if plot == True:
        #     ax.plot(discharge_specfic_capacity,voltage_discharge,label=f'Cycle {cycle_i}')
        
        voltage_discharge = df_discharge_running['Voltage']
        
        #add the last voltage of the discharge to the rest at the start
        voltage_discharge_rest = df_discharge_resting[['Voltage']].copy()
        if len(voltage_discharge_rest) == 0:
            # create an empty dataframe
            voltage_discharge_rest = pd.DataFrame({})
        else:
            voltage_discharge_rest[:] = voltage_discharge_rest['Voltage'].iloc[-1]
        #voltage_discharge_rest.iloc[:,'Voltage']=voltage_discharge.iloc[-1]
        # try:
        if len(voltage_discharge_rest) == 0:
            voltage_discharge_rest_separate = pd.Series([voltage_discharge.iloc[-1]])
        else:
            voltage_discharge_rest_separate = pd.concat(
                [pd.Series([voltage_discharge.iloc[-1]]), voltage_discharge_rest],
                ignore_index=True
            )
        voltage_charge = df_charge_running['Voltage']
        voltage_charge_rest = df_charge_resting['Voltage']
        time_discharge = df_discharge_running[time_col_name]
        time_discharge_rest = df_discharge_resting[time_col_name]
        time_charge = df_charge_running[time_col_name]
        time_charge_rest = df_charge_resting[time_col_name]
        specific_capacity_discharge = df_discharge_running['Specific Capacity Total AM']
        specific_capacity_discharge_rest = df_discharge_resting['Specific Capacity Total AM'].copy()
        specific_power_discharge = df_discharge_running['Specific Power Total AM']
        specific_power_charge = df_charge_running['Specific Power Total AM']
        if len(specific_capacity_discharge_rest) == 0:
            # create an empty dataframe
            specific_capacity_discharge_rest = pd.DataFrame({})
        elif specific_capacity_discharge_rest.iloc[0] == 0:
            specific_capacity_discharge_rest.iloc[:]=specific_capacity_discharge.iloc[-1]
        if len(specific_capacity_discharge_rest) == 0:
            specific_capacity_discharge_rest_separate = pd.Series([specific_capacity_discharge.iloc[-1]])
        else:
            specific_capacity_discharge_rest_separate = pd.concat(
                [pd.Series([specific_capacity_discharge.iloc[-1]]), specific_capacity_discharge_rest],
                ignore_index=True
            )
        specific_capacity_charge = df_charge_running['Specific Capacity Total AM']
        specific_capacity_charge_rest = df_charge_resting['Specific Capacity Total AM'].copy()
        if len(specific_capacity_charge) == 0:
            # half cycle 0, no charge, only discharge
            specific_capacity_charge = pd.DataFrame({})
            specific_capacity_charge_rest = pd.DataFrame({})
        if len(specific_capacity_charge_rest) == 0:
            # create an empty dataframe
            specific_capacity_charge_rest = pd.DataFrame({})
        elif specific_capacity_charge_rest.iloc[0] == 0:
            specific_capacity_charge_rest.iloc[:]=specific_capacity_charge.iloc[-1]
        specific_current_discharge = df_discharge_running['Specific Current Total AM']
        specific_current_charge = df_charge_running['Specific Current Total AM']
        areal_current_discharge = df_discharge_running['Areal Current']
        areal_current_charge = df_charge_running['Areal Current']
        

        if plot == True:

            # temp
            # sc_totalAM_not_charge = df_i['Specific Capacity Total AM'][df_i['state'] != 1]
            # voltage_not_charge = df_i['Voltage'][df_i['state'] != 1]
            # ax.plot(sc_totalAM_not_charge,voltage_not_charge,label=f'Cycle {cycle_i} Not Charge')            
            #ax.plot(df_i['Specific Capacity Total AM'],df_i['Voltage'],label=f'Cycle {cycle_i} All')
        
            ax.plot(specific_capacity_discharge,voltage_discharge,label=f'Cycle {cycle_i} Discharge')
            ax.plot(specific_capacity_discharge_rest,voltage_discharge_rest,label=f'Cycle {cycle_i} Rest')
            ax.plot(specific_capacity_charge,voltage_charge,label=f'Cycle {cycle_i} Charge')
            
        # Convert to 1D NumPy arrays for consistency
        voltage_discharge_arr = list(voltage_discharge.to_numpy().ravel())
        voltage_discharge_rest_arr = list(voltage_discharge_rest_separate.to_numpy().ravel())
        voltage_discharge_tot_arr = list(voltage_discharge.to_numpy().ravel()) + list(voltage_discharge_rest.to_numpy().ravel())
        voltage_charge_arr = list(voltage_charge.to_numpy().ravel())
        voltage_charge_rest_arr = list(voltage_charge_rest.to_numpy().ravel())
        specific_capacity_discharge_arr = list(specific_capacity_discharge.to_numpy().ravel())
        specific_capacity_discharge_rest_arr = list(specific_capacity_discharge_rest_separate.to_numpy().ravel())
    
        specific_capacity_discharge_tot_arr = list(specific_capacity_discharge.to_numpy().ravel()) + list(specific_capacity_discharge_rest.to_numpy().ravel())
        specific_capacity_charge_arr = list(specific_capacity_charge.to_numpy().ravel())
        specific_capacity_charge_rest_arr = list(specific_capacity_charge_rest.to_numpy().ravel())

        time_discharge_arr = list(time_discharge.to_numpy().ravel())
        time_discharge_rest_arr = list(time_discharge_rest.to_numpy().ravel())
        time_charge_arr = list(time_charge.to_numpy().ravel())
        time_charge_rest_arr = list(time_charge_rest.to_numpy().ravel())

        # Append to lists
        voltage_discharge_lst.append(voltage_discharge_arr)
        voltage_discharge_rest_lst.append(voltage_discharge_rest_arr)
        voltage_discharge_tot_lst.append(voltage_discharge_tot_arr)
        voltage_charge_lst.append(voltage_charge_arr)
        voltage_charge_rest_lst.append(voltage_charge_rest_arr)

        specific_capacity_discharge_lst.append(specific_capacity_discharge_arr)
        specific_capacity_discharge_rest_lst.append(specific_capacity_discharge_rest_arr) 
        specific_capacity_discharge_tot_lst.append(specific_capacity_discharge_tot_arr)
        specific_capacity_charge_lst.append(specific_capacity_charge_arr)
        specific_capacity_charge_rest_lst.append(specific_capacity_charge_rest_arr) 

        time_discharge_lst.append(time_discharge_arr)
        time_discharge_rest_lst.append(time_discharge_rest_arr) 
        time_charge_lst.append(time_charge_arr)
        time_charge_rest_lst.append(time_charge_rest_arr) 

        specific_power_discharge_lst.append(specific_power_discharge.to_numpy())
        specific_power_charge_lst.append(specific_power_charge.to_numpy())

        # Scalars: mean values
        specific_current_avg_discharge_lst.append(np.round(np.mean(specific_current_discharge.to_numpy()), decimals=4))
        specific_current_avg_charge_lst.append(np.round(np.mean(specific_current_charge.to_numpy()), decimals=4))
        areal_current_avg_discharge_lst.append(np.round(np.mean(areal_current_discharge.to_numpy()), decimals=4))
        areal_current_avg_charge_lst.append(np.round(np.mean(areal_current_charge.to_numpy()), decimals=4))
        specific_current_discharge_lst.append(specific_current_discharge.to_numpy())
        specific_current_charge_lst.append(specific_current_charge.to_numpy())
        areal_current_discharge_lst.append(areal_current_discharge.to_numpy())
        areal_current_charge_lst.append(areal_current_charge.to_numpy())
        # Cycle index
        cycle_lst.append(cycle_i)
    if plot == True:
        plt.legend()

    if formation_cycle != None:
        return {
            'voltage_discharge_lst':voltage_discharge_lst,
            'voltage_discharge_rest_lst':voltage_discharge_rest_lst,
            'voltage_discharge_tot_lst':voltage_discharge_tot_lst,
            'voltage_charge_lst':voltage_charge_lst,
            'voltage_charge_rest_lst':voltage_charge_rest_lst,
            'specific_capacity_discharge_lst':specific_capacity_discharge_lst,
            'specific_capacity_discharge_rest_lst':specific_capacity_discharge_rest_lst,
            'specific_capacity_discharge_tot_lst':specific_capacity_discharge_tot_lst,
            'specific_capacity_charge_lst':specific_capacity_charge_lst,
            'specific_capacity_charge_rest_lst':specific_capacity_charge_rest_lst,
            'time_discharge_lst':time_discharge_lst,
            'time_discharge_rest_lst':time_discharge_rest_lst,
            'time_charge_lst':time_charge_lst,
            'time_charge_rest_lst':time_charge_rest_lst,
            'specific_current_discharge_lst':specific_current_discharge_lst,
            'specific_current_charge_lst':specific_current_charge_lst,
            'areal_current_discharge_lst':areal_current_discharge_lst,
            'areal_current_charge_lst':areal_current_charge_lst,
            'specific_power_discharge_lst':specific_power_discharge_lst,
            'specific_power_charge_lst': specific_power_charge_lst,
            'cycle_lst':cycle_lst,
            'voltage_discharge_formation':list(voltage_discharge_formation.to_numpy().ravel()),
            'voltage_rest_end_formation': list(voltage_rest_end_formation.to_numpy().ravel()),
            'time_discharge_formation': list(time_discharge_formation.to_numpy().ravel()),
            'time_rest_formation': list(time_rest_formation.to_numpy().ravel()),
            'specific_capacity_discharge_formation': list(specific_capacity_discharge_formation.to_numpy().ravel()),
            'specific_capacity_rest_end_formation': list(specific_capacity_rest_end_formation.to_numpy().ravel()), 
            'specific_current_discharge_formation': np.round(np.mean(specific_current_discharge_formation.to_numpy()), decimals=4),
            'areal_current_discharge_formation': np.round(np.mean(areal_current_discharge_formation.to_numpy()), decimals=4),
        }
    else:
        return {
            'voltage_discharge_lst':voltage_discharge_lst,
            'voltage_discharge_rest_lst':voltage_discharge_rest_lst,
            'voltage_discharge_tot_lst':voltage_discharge_tot_lst,
            'voltage_charge_lst':voltage_charge_lst,
            'voltage_charge_rest_lst':voltage_charge_rest_lst,
            'specific_capacity_discharge_lst':specific_capacity_discharge_lst,
            'specific_capacity_discharge_rest_lst':specific_capacity_discharge_rest_lst,
            'specific_capacity_discharge_tot_lst':specific_capacity_discharge_tot_lst,
            'specific_capacity_charge_lst':specific_capacity_charge_lst,
            'specific_capacity_charge_rest_lst':specific_capacity_charge_rest_lst,
            'time_discharge_lst':time_discharge_lst,
            'time_discharge_rest_lst':time_discharge_rest_lst,
            'time_charge_lst':time_charge_lst,
            'time_charge_rest_lst':time_charge_rest_lst,
            'specific_current_discharge_lst':specific_current_discharge_lst,
            'specific_current_charge_lst':specific_current_charge_lst,
            'specific_power_discharge_lst':specific_power_discharge_lst,
            'specific_power_charge_lst': specific_power_charge_lst,
            'areal_current_discharge_lst':areal_current_discharge_lst,
            'areal_current_charge_lst':areal_current_charge_lst,
            'cycle_lst':cycle_lst,
        }


def voltage_vs_capacity_GITT(
    df,
    cycles=None,
    plot=False,
    last_relax_index = -1,
    end_of_discharge_stop_correctly=[True],
    state_corrected=[True],

    plot_raw=False
):
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    time_col_name = 'time/s'
    if 'Time' in df.columns:
        time_col_name = 'Time'
    df_unique_cycle = np.unique(df['full cycle'])
    if cycles == None:
        formation_cycle = 0
        start_cycle = df_unique_cycle[0]
        end_cycle = df_unique_cycle[-1]
        if end_cycle >= 10:
            end_cycle = 10
        cycles = [start_cycle,end_cycle]
    elif cycles[0] == -1:
        end_cycle = df_unique_cycle[-1]
        if end_cycle >= 10:
            end_cycle = 10
        cycles = [end_cycle]
        formation_cycle = None
    elif cycles[0] == 0:
        formation_cycle = cycles[0]
        cycles = cycles[1:]
    elif cycles[0] != 0:
        formation_cycle = None

    if formation_cycle != None:
        df0 = df[df['full cycle']==formation_cycle]
        df_discharge_running = df0[df0['state']==-1]
        df_discharge_resting = df0[df0['state']==0]

        specific_capacity_discharge_formation = df_discharge_running['Specific Capacity Total AM']
        voltage_discharge_formation = df_discharge_running['Voltage']
        #rest_idx=np.array(df_discharge_running.index[np.where(np.diff(df_discharge_running.index)>1)[0]][:-1])
        end_discharge_idx=np.where(np.diff(df_discharge_running.index)>1)[0]
        start_discharge_idx = [int(i+1) for i in end_discharge_idx]

        separated_specific_capacity_discharge_lst = []
        separated_voltage_discharge_lst = []
        for i in range(len(end_discharge_idx)):
            if i == 0:
                specific_capacity_discharge = df_discharge_running['Specific Capacity Total AM'].iloc[:end_discharge_idx[i]]
                voltage_discharge = df_discharge_running['Voltage'].iloc[:end_discharge_idx[i]]
            elif i == len(end_discharge_idx)-1:
                specific_capacity_discharge = df_discharge_running['Specific Capacity Total AM'].iloc[start_discharge_idx[i]:]
                voltage_discharge = df_discharge_running['Voltage'].iloc[start_discharge_idx[i]:]
            else:
                specific_capacity_discharge = df_discharge_running['Specific Capacity Total AM'].iloc[start_discharge_idx[i-1]:end_discharge_idx[i]]
                voltage_discharge = df_discharge_running['Voltage'].iloc[start_discharge_idx[i-1]:end_discharge_idx[i]]
            # if plot == True:
            #     ax.plot(specific_capacity_discharge,voltage_discharge,label=f'Cycle {formation_cycle} Discharge {i}')
            #     fig.savefig('temp_test.png',dpi=100)
            
            separated_specific_capacity_discharge_lst.append(specific_capacity_discharge)
            separated_voltage_discharge_lst.append(voltage_discharge)
        # if plot == True:
        #     ax.plot(specific_capacity_discharge_formation,voltage_discharge_formation,label=f'Cycle {formation_cycle} Discharge Formation',linestyle='--')
        #     print(len(separated_specific_capacity_discharge_lst),len(separated_voltage_discharge_lst))
        #     [ax.plot(x, y, linestyle='-', label=f'Cycle {formation_cycle} Rest') for x, y in zip(separated_specific_capacity_discharge_lst, separated_voltage_discharge_lst)]
        #     fig.savefig('temp_test.png',dpi=100)
        # exit()

        specific_capacity_rest = df_discharge_resting['Specific Capacity Total AM'].copy()
        specific_capacity_rest_round = specific_capacity_rest.round(0) 
        unique_specific_capacity_rest = np.unique(specific_capacity_rest_round)
        separated_full_relax_voltage_lst = []
        separated_full_relax_specific_capacity_lst = []
        separated_full_relax_time_lst = []
        last_relax_voltage_lst = []
        last_relax_specific_capacity_lst = []
        last_relax_time_lst = []
        for i, unique_specific_capacity in enumerate(unique_specific_capacity_rest):
            idx_lst = np.where(specific_capacity_rest_round == unique_specific_capacity)[0]
            full_relax_voltage = df_discharge_resting['Voltage'].iloc[idx_lst]
            full_relax_time = df_discharge_resting[time_col_name].iloc[idx_lst]
            full_relax_specific_capacity = df_discharge_resting['Specific Capacity Total AM'].iloc[idx_lst]

            last_relax_voltage = df_discharge_resting['Voltage'].iloc[idx_lst[last_relax_index]]
            last_relax_time = df_discharge_resting[time_col_name].iloc[idx_lst[last_relax_index]]
            last_relax_specific_capacity = df_discharge_resting['Specific Capacity Total AM'].iloc[idx_lst[last_relax_index]]

            separated_full_relax_specific_capacity_lst.append(unique_specific_capacity)
            separated_full_relax_voltage_lst.append(full_relax_voltage)
            separated_full_relax_time_lst.append(full_relax_time)

            last_relax_specific_capacity_lst.append(last_relax_specific_capacity)
            last_relax_voltage_lst.append(last_relax_voltage)
            last_relax_time_lst.append(last_relax_time)
    output_dict = {
        'separated_full_relax_specific_capacity_lst_formation_cycle': [float(x) for x in separated_full_relax_specific_capacity_lst],
        'separated_full_relax_voltage_lst_formation_cycle': [v.tolist() for v in separated_full_relax_voltage_lst],
        'separated_full_relax_time_lst_formation_cycle': [t.tolist() for t in separated_full_relax_time_lst],
        'last_relax_specific_capacity_lst_formation_cycle': [float(x) for x in last_relax_specific_capacity_lst],
        'last_relax_voltage_lst_formation_cycle': [float(x) for x in last_relax_voltage_lst],
        'last_relax_time_lst_formation_cycle': [float(x) for x in last_relax_time_lst],
        'separated_discharge_specific_capacity_formation_cycle': [v.tolist() for v in separated_specific_capacity_discharge_lst],
        'separated_discharge_voltage_formation_cycle': [v.tolist() for v in separated_voltage_discharge_lst],
        'full_discharge_specific_capacity_formation_cycle': specific_capacity_discharge_formation.tolist(),
        'full_discharge_voltage_formation_cycle': voltage_discharge_formation.tolist()
    }
    return output_dict
    #to do 
    # voltage_lst = []
    # specific_capacity_lst = []
    # ocv_lst = []
    # overpotential_lst = []
    # ocv_specific_capacity_lst = []
    # cycle_lst = []
    # c_rate_lst = []
    # specific_current_lst = []

    # for i, cycle_i in enumerate(cycles):

    #     df_i = df[df['full cycle'] == cycle_i]
    #     df_i_half_cycle = df_i['half cycle'].unique()

    #     df_i_charge = df_i[df_i['half cycle']==df_i_half_cycle[0]]
    #     df_charge_running = df_i_charge[df_i_charge['state']==1]
    #     df_charge_resting = df_i_charge[df_i_charge['state']==0]
        

    #     df_i_discharge = df_i[df_i['half cycle']==df_i_half_cycle[1]]
    #     df_discharge_running = df_i_discharge[df_i_discharge['state']==-1]
    #     df_discharge_resting = df_i_discharge[df_i_discharge['state']==0]

    #     specific_capacity = df['Specific Capacity Total AM'][df['full cycle'] == cycle_i]
    #     voltage = df['Voltage'][df['full cycle'] == cycle_i]
    #     specific_current = df['Specific Current Total AM'][df['full cycle'] == cycle_i]

    #     if plot_raw:
    #         plt.plot(specific_capacity, voltage, label=f'Cycle {cycle_i}')
    #         plt.legend()
    #         plt.show()

    #     discharge_start_idx = specific_capacity[np.abs(specific_capacity - 0) < 0.0001].index[-1]
    #     discharge_specific_capacity = specific_capacity.loc[discharge_start_idx:]
    #     discharge_stop_flag = [False] + list(np.diff(discharge_specific_capacity) < 0.0001)

    #     leading_true_idx = []
    #     skip_discharge = False
    #     for df_idx, flag in zip(discharge_specific_capacity.index, discharge_stop_flag):
    #         if flag and not skip_discharge:
    #             leading_true_idx.append(df_idx)
    #             skip_discharge = True
    #         elif not flag:
    #             skip_discharge = False

    #     trailing_true_idx = []
    #     for j, df_idx in enumerate(leading_true_idx):
    #         df_section = df.loc[df_idx:]
    #         diff_arr = np.diff(np.array(df_section['state']))

    #         if j == len(leading_true_idx) - 1:
    #             if end_of_discharge_stop_correctly[i] and state_corrected[i]:
    #                 nonzero_idx = np.where(diff_arr[1:] == 0)[-1]
    #             else:
    #                 nonzero_idx = np.where(diff_arr[1:] > 0)[0]
    #         else:
    #             nonzero_idx = np.where(diff_arr < 0)[0]

    #         stop_idx = np.array(df_section.index)[nonzero_idx[0]]
    #         trailing_true_idx.append(stop_idx)

    #     trailing_true_idx = [discharge_start_idx] + trailing_true_idx
    #     voltage_discharge = voltage.loc[discharge_start_idx:]

    #     if plot:
    #         ax.plot(specific_capacity.loc[discharge_start_idx:], voltage_discharge, label=f'Cycle {cycle_i}')
    #         ax.scatter(
    #             discharge_specific_capacity.loc[leading_true_idx],
    #             voltage_discharge.loc[leading_true_idx],
    #             color='g',
    #             marker='.'
    #         )
    #         ax.scatter(
    #             discharge_specific_capacity.loc[trailing_true_idx],
    #             voltage_discharge.loc[trailing_true_idx],
    #             color='r',
    #             marker='.'
    #         )

    #     ocv = voltage_discharge.loc[trailing_true_idx].to_numpy()
    #     voltage_relaxed_start = voltage_discharge.loc[leading_true_idx].to_numpy()
    #     discharge_ocv_capacity = discharge_specific_capacity.loc[trailing_true_idx].to_numpy()

    #     voltage_lst.append(voltage_discharge.to_numpy())
    #     specific_capacity_lst.append(specific_capacity.loc[discharge_start_idx:].to_numpy())
    #     ocv_lst.append(ocv)
    #     ocv_specific_capacity_lst.append(discharge_ocv_capacity)
    #     overpotential_lst.append(ocv[1:] - voltage_relaxed_start)
    #     cycle_lst.append(cycle_i)

    #     specific_current_discharge = specific_current.loc[discharge_start_idx]
    #     non_zero_current = specific_current_discharge[specific_current_discharge != 0]
    #     specific_current_lst.append(np.round(non_zero_current.mean(), decimals=4))

    # return {
    #     'voltage_lst': voltage_lst,
    #     'specific_capacity_lst': specific_capacity_lst,
    #     'ocv_lst': ocv_lst,
    #     'overpotential_lst': overpotential_lst,
    #     'ocv_specific_capacity_lst': ocv_specific_capacity_lst,
    #     'cycle_lst': cycle_lst,
    #     'specific_current_lst': specific_current_lst
    # }


def plot_multiple_voltage_vs_cycling(voltage,capacity,cycle_lst,linestyle,color_customize=None,color_gradient_customize=None,fig=None,ax=None,dpi=150,colorbar=False,set_colorbar_label=True,min_cycle=None,max_cycle=None,variable_linewidth=False):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.ticker as ticker
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6),dpi=dpi)
    if min_cycle is None and max_cycle is None:
        min_cycle = min(cycle_lst)
        max_cycle = max(cycle_lst)
    norm = Normalize(vmin=min_cycle, vmax=max_cycle)
    if color_gradient_customize is not None:
        sm = ScalarMappable(cmap=color_gradient_customize, norm=norm)
    else:
        from matplotlib.cm import get_cmap
        from matplotlib.colors import LinearSegmentedColormap
        base = get_cmap('Blues_r')
        colors = base(np.linspace(0.0, 0.7, max_cycle - min_cycle + 1))
        custom_cmap = LinearSegmentedColormap.from_list('custom_blues', colors)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    for i, (v, c) in enumerate(zip(voltage,capacity)):        
        if color_customize is None:
            color = sm.cmap(norm(cycle_lst[i]))
        else:
            color = color_customize
        if variable_linewidth == True:
            if cycle_lst[i] not in [min_cycle,max_cycle]:
                ax.plot(c,v,color=color,linestyle='--',linewidth=1.0)
            else:
                ax.plot(c,v,color=color,linestyle=linestyle)
        else:
            ax.plot(c,v,color=color,linestyle=linestyle)

    if colorbar:
        cbar = fig.colorbar(sm, ax=ax)
        if set_colorbar_label == True:
            cbar.set_label("Cycle Number")
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) 

    ax.set_xlabel('Specific Capacity (mAh/g-AM)')
    ax.set_ylabel('Voltage (V)')
    #plt.legend(bbox_to_anchor=(1.35,1))
    return fig, ax 

def plot_ocv_and_overpotential(ocv, overpotential, capacity, cycle_lst, ocv_marker, dpi=150,legend=False,colorbar=False,fig=None,ax=None,overpotential_separate_plot=False):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    if fig is None and ax is None:
        if overpotential_separate_plot:
            fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(18,6),dpi=dpi)
        else:
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6),dpi=dpi)
    norm = Normalize(vmin=0, vmax=max(cycle_lst))
    sm = ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    # ax_right = ax.twinx()
    for i, (ocv_i, overpotential_i, c_i) in enumerate(zip(ocv, overpotential, capacity)):
        color = sm.cmap(norm(cycle_lst[i]))
        # if overpotential_marker is None:
        #     over_marker = 'x'
        # else:
        #     over_marker = overpotential_marker
        if ocv_marker is None:
            o_marker = 'o'
        else:
            o_marker = ocv_marker
        if overpotential_separate_plot == True:
            ax[0].scatter(c_i, ocv_i, marker=o_marker, color=color, edgecolors='k')
        else:
            ax.scatter(c_i, ocv_i, marker=o_marker, color=color, edgecolors='k')
        if overpotential_separate_plot == True:
            ax[1].scatter(c_i[1:], overpotential_i, marker=o_marker, color=color,edgecolors='k')
        else:
            ax.scatter(c_i[1:], overpotential_i, marker=o_marker, color='none',edgecolors=color)

    if legend==True:
        # Custom legend
        legend_elements = [
            plt.Line2D([0], [0], marker=o_marker, color='w', markerfacecolor='black', markersize=8, label='OCV'),
            plt.Line2D([0], [0], marker=over_marker, color='w', markerfacecolor='black', markersize=8, label='Overpotential')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    if colorbar:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Cycle Number")
    if overpotential_separate_plot:
        ax[0].set_xlabel('Specific Capacity (mAh/g-AM)')
        ax[0].set_ylabel('Voltage (V)')
        ax[1].set_xlabel('Specific Capacity (mAh/g-AM)')
        ax[1].set_ylabel('Overpotential (V)')
    else:
        ax.set_xlabel('Specific Capacity (mAh/g-AM)')
        ax.set_ylabel('Voltage (V)')
    #ax_right.set_ylabel('Overpotential (V)')
    return fig,ax
