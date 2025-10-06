import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
import argparse

# import herald_visualization.echem as ec
# from herald_visualization.mpr2csv import cycle_mpr2csv
# from herald_visualization.plot import plot_cycling, plot_gitt, parse_cycle_csv, plot_cycling_plotly
from herald_visualization.fancy_plot import (
    voltage_vs_capacity_cycling,
    plot_multiple_voltage_vs_cycling,
)
from herald_visualization.celldesignroutine import cellmodel_IL_pouch_final
from ruamel.yaml import YAML

default_params = {
    # 'font.family': 'Helvetica',
    "axes.labelsize": 20,
    "axes.labelweight": "bold",  # Make axes labels bold
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.size": 24,
    "axes.linewidth": 2.0,
    "lines.dashed_pattern": (5, 2.5),
    "lines.markersize": 10,
    "lines.linewidth": 3,
    "lines.markeredgewidth": 1,
    "lines.markeredgecolor": "k",
    "legend.fontsize": 20,  # Adjust the font size of the legend
    "legend.title_fontsize": 20,  # Increase legend title size if needed
    "legend.frameon": False,
}
plt.rcParams.update(default_params)

# cells_df = pd.read_csv('/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing/In-house cells and syntheses - Coin Cells.csv')
# available_ids = cells_df['Test ID'].tolist()

# function for handling id to absolute path
data_path = "/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing/BCS905"  # here is where data path is taken care of


def id_to_path(cellid, root_dir=data_path):
    """
    Find the correct directory path to a data folder from the cell ID
    """

    glob_str = os.path.join("**/outputs/*" + cellid + "_*.csv")
    paths = glob.glob(glob_str, root_dir=root_dir, recursive=True)
    if len(paths) == 1:
        return os.path.join(root_dir, paths[0])
    elif len(paths) == 0:
        glob_str = os.path.join("**/outputs/*" + cellid + "*.csv")
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

def calc_se_cell(
    output_dict,
):
    from scipy.integrate import simpson

    discharge_se_lst_cell_id_1 = []
    charge_se_lst_cell_id_1 = []
    discharge_sc_lst_cell_id_1 = []
    charge_sc_lst_cell_id_1 = []

    for i, (v, c) in enumerate(
        zip(
            output_dict["voltage_discharge_lst"],
            output_dict["specific_capacity_discharge_lst"],
        )
    ):
        # se = simpson(x=c,y=v)
        # discharge_se_lst_cell_id_1.append(se)
        discharge_sc_lst_cell_id_1.append(max(c))
    for i, (p, t) in enumerate(
        zip(
            output_dict["specific_power_discharge_lst"],
            output_dict["time_discharge_lst"],
        )
    ):
        t = np.array(t)
        se = simpson(x=t, y=-p) / 3600.0  # Convert to Wh
        discharge_se_lst_cell_id_1.append(se)
    discharge_se_lst_cell_id_1 = np.array(discharge_se_lst_cell_id_1)
    discharge_sc_lst_cell_id_1 = np.array(discharge_sc_lst_cell_id_1)
    charge_se_lst_cell_id_1 = np.array(charge_se_lst_cell_id_1)
    charge_sc_lst_cell_id_1 = np.array(charge_sc_lst_cell_id_1)
    return (
        discharge_se_lst_cell_id_1,
        discharge_sc_lst_cell_id_1,
        charge_se_lst_cell_id_1,
        charge_sc_lst_cell_id_1,
    )


def plot_voltage_vs_capacity_single_cell(
    cell_id,
    fig,
    ax,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
    if not os.path.exists(id_to_path(cell_id)) or id_to_path(cell_id) is None:
        print(f"No data found for cell ID {cell_id}")
        return None
    df = pd.read_csv(id_to_path(cell_id))
    print(id_to_path(cell_id))
    df_in_house_cell_and_synthesis = pd.read_csv(
        "/scratch/venkvis_root/venkvis/shared_data/herald/In-house cells and syntheses - Cells.csv"
    )
    df_in_house_cell_and_synthesis = df_in_house_cell_and_synthesis[
        df_in_house_cell_and_synthesis["Test ID"] == cell_id
    ]
    unique_cycles = df["full cycle"].unique().astype(int).tolist()
    cell_id_1_cycles = unique_cycles[:-1]
    output_dict = voltage_vs_capacity_cycling(df, cycles=cell_id_1_cycles, plot=False)
    if len(output_dict["cycle_lst"]) == 0:
        print(f"No cycles found for {cell_id}")
        return None, None, None
    print('output cycle list: ', output_dict['cycle_lst'])
    for i, cycle in enumerate(output_dict["cycle_lst"][:11]):
        max_cycle = min(max(output_dict["cycle_lst"][:-1]), 10)
        min_cycle = 0
        if cycle > max_cycle:
            continue
        specific_capacity_discharge = output_dict["specific_capacity_discharge_lst"][i]
        # where specific_capacity_discharge is decreases, add to the previous value
        specific_capacity_discharge = np.array(specific_capacity_discharge)
        voltage_discharge = np.array(output_dict["voltage_discharge_lst"][i])
        # in the power profile, make sure the offset when changing power profile is accounted for
        noisy_indices = np.where(np.diff(specific_capacity_discharge) < -10)[0]
        while len(noisy_indices) > 1:
            specific_capacity_discharge = specific_capacity_discharge[
                [
                    j
                    for j in range(len(specific_capacity_discharge))
                    if j not in noisy_indices
                ]
            ]
            voltage_discharge = voltage_discharge[
                [j for j in range(len(voltage_discharge)) if j not in noisy_indices]
            ]
            noisy_indices = np.where(np.diff(specific_capacity_discharge) < -10)[0]
        
        reset_indices = np.where(np.diff(specific_capacity_discharge) < -10)[0]
        # print('reset indices: ', reset_indices, specific_capacity_discharge[reset_indices])
        if len(reset_indices) != 0:
            x_translation_value = specific_capacity_discharge[reset_indices[0]]
            specific_capacity_discharge[reset_indices[0] + 1 :] = (
                specific_capacity_discharge[reset_indices[0] + 1 :] + x_translation_value
            )

        specific_capacity_charge = output_dict["specific_capacity_charge_lst"][i]
        specific_capacity_charge = np.array(specific_capacity_charge)
        # specific_capacity_charge = specific_capacity_charge * m_am / m_chem
        voltage_charge = output_dict["voltage_charge_lst"][i]
        output_dict["voltage_discharge_lst"][i] = voltage_discharge
        output_dict["specific_capacity_discharge_lst"][i] = specific_capacity_discharge
        if cycle == min_cycle:
            fig, ax = plot_multiple_voltage_vs_cycling(
                [np.array(voltage_discharge)],
                [np.array(specific_capacity_discharge)],
                [cycle],
                linestyle="--",
                color_customize=None,
                fig=fig,
                ax=ax,
                colorbar=False,
                min_cycle=0,
                max_cycle=max_cycle,
            )
        else:
            fig, ax = plot_multiple_voltage_vs_cycling(
                [np.array(voltage_discharge)],
                [np.array(specific_capacity_discharge)],
                [cycle],
                linestyle="-",
                color_customize=None,
                fig=fig,
                ax=ax,
                colorbar=False,
                min_cycle=0,
                max_cycle=max_cycle,
            )
        if cycle == max_cycle:
            # fig,ax = plot_multiple_voltage_vs_cycling([voltage_charge],[specific_capacity_discharge[-1]-np.array(specific_capacity_charge)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=True,min_cycle=1,max_cycle=max(cell_id_1_cycles))
            print("Plotting charge curve for last cycle")
            fig, ax = plot_multiple_voltage_vs_cycling(
                [voltage_charge],
                [np.array(specific_capacity_charge)],
                [cycle],
                linestyle="-",
                color_customize=None,
                fig=fig,
                ax=ax,
                colorbar=True,
                min_cycle=0,
                max_cycle=max_cycle,
            )      
        else:
            # fig,ax = plot_multiple_voltage_vs_cycling([voltage_charge],[specific_capacity_discharge[-1]-np.array(specific_capacity_charge)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles))
            fig, ax = plot_multiple_voltage_vs_cycling(
                [voltage_charge],
                [np.array(specific_capacity_charge)],
                [cycle],
                linestyle="-",
                color_customize=None,
                fig=fig,
                ax=ax,
                colorbar=False,
                min_cycle=0,
                max_cycle=max_cycle,
            )
    ax.set_ylim([0, 4.5])
    ax.set_xlabel("Specific Capacity (mAh/g-AM)")
    ax.set_ylabel('Voltage (V)')
    plt.tight_layout()
    # add title
    return output_dict, fig, ax

def plot_voltage_vs_capacity_single_cell_with_overpotential(
    cell_id,
    fig,
    ax,
    max_cycle = 10,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
    print(id_to_path(cell_id))
    if not os.path.exists(id_to_path(cell_id)) or id_to_path(cell_id) is None:
        print(f"No data found for cell ID {cell_id}")
        return None
    df = pd.read_csv(id_to_path(cell_id))
    print(id_to_path(cell_id))
    df_in_house_cell_and_synthesis = pd.read_csv(
        "/scratch/venkvis_root/venkvis/shared_data/herald/In-house cells and syntheses - Cells.csv"
    )
    df_in_house_cell_and_synthesis = df_in_house_cell_and_synthesis[
        df_in_house_cell_and_synthesis["Test ID"] == cell_id
    ]
    unique_cycles = df["full cycle"].unique().astype(int).tolist()
    cell_id_1_cycles = unique_cycles[:-1]
    output_dict = voltage_vs_capacity_cycling(df, cycles=cell_id_1_cycles, plot=False)
    if len(output_dict["cycle_lst"]) == 0:
        print(f"No cycles found for {cell_id}")
        return None, None, None
    print('output cycle list: ', output_dict['cycle_lst'])
    for i, cycle in enumerate(output_dict["cycle_lst"][:max_cycle+1]):
        min_cycle = 0
        if cycle > max_cycle:
            continue
        specific_capacity_discharge = output_dict["specific_capacity_discharge_lst"][i]
        # where specific_capacity_discharge is decreases, add to the previous value
        specific_capacity_discharge = np.array(specific_capacity_discharge)
        voltage_discharge = np.array(output_dict["voltage_discharge_lst"][i])
        areal_current = np.array(output_dict["areal_current_discharge_lst"][i])
        # # power = np.array(output_dict['specific_power_discharge_lst'][i])
        # # in the power profile, make sure the offset when changing power profile is accounted for
        # noisy_indices = np.where(np.diff(specific_capacity_discharge) < -1)[0]
        # while len(noisy_indices) > 1:
        #     specific_capacity_discharge = specific_capacity_discharge[
        #         [
        #             j
        #             for j in range(len(specific_capacity_discharge))
        #             if j not in noisy_indices
        #         ]
        #     ]
        #     voltage_discharge = voltage_discharge[
        #         [j for j in range(len(voltage_discharge)) if j not in noisy_indices]
        #     ]
        #     areal_current = areal_current[
        #         [j for j in range(len(areal_current)) if j not in noisy_indices]
        #     ]
        #     # power = power[
        #     #     [j for j in range(len(power)) if j not in noisy_indices]
        #     # ]
        #     noisy_indices = np.where(np.diff(specific_capacity_discharge) < -1)[0]
        # reset_indices = np.where(np.diff(specific_capacity_discharge) < -1)[0]
        # # print('reset indices: ', reset_indices, specific_capacity_discharge[reset_indices])
        # if len(reset_indices) != 0:
        #     x_translation_value = specific_capacity_discharge[reset_indices[0]]
        #     specific_capacity_discharge[reset_indices[0] + 1 :] = (
        #         specific_capacity_discharge[reset_indices[0] + 1 :] + x_translation_value
        #     )
        output_dict["voltage_discharge_lst"][i] = voltage_discharge
        output_dict["specific_capacity_discharge_lst"][i] = specific_capacity_discharge  
        output_dict["areal_current_discharge_lst"][i] = areal_current
        # output_dict["specific_power_discharge_lst"][i] = power
        specific_capacity_discharge_rest = np.ones_like(output_dict["specific_capacity_discharge_rest_lst"][i]) * max(specific_capacity_discharge)
        voltage_discharge_rest = np.array(output_dict["voltage_discharge_rest_lst"][i])
        voltage_discharge_rest = voltage_discharge_rest[~np.isnan(voltage_discharge_rest)]
        specific_capacity_discharge_rest = specific_capacity_discharge_rest[~np.isnan(specific_capacity_discharge_rest)]
        specific_capacity_discharge = np.concatenate((specific_capacity_discharge, specific_capacity_discharge_rest))
        voltage_discharge = np.concatenate((voltage_discharge, voltage_discharge_rest))                  

        specific_capacity_charge = output_dict["specific_capacity_charge_lst"][i]
        specific_capacity_charge = np.array(specific_capacity_charge)
        # specific_capacity_charge = specific_capacity_charge * m_am / m_chem
        voltage_charge = output_dict["voltage_charge_lst"][i]
        voltage_charge = np.array(voltage_charge)
        specific_capacity_charge_rest = np.array(output_dict["specific_capacity_charge_rest_lst"][i])
        voltage_charge_rest = np.array(output_dict["voltage_charge_rest_lst"][i])
        voltage_charge_rest = voltage_charge_rest[~np.isnan(voltage_charge_rest)]
        specific_capacity_charge_rest = specific_capacity_charge_rest[~np.isnan(specific_capacity_charge_rest)]
        specific_capacity_charge = np.concatenate((specific_capacity_charge, specific_capacity_charge_rest))
        voltage_charge = np.concatenate((voltage_charge, voltage_charge_rest))

        if cycle == min_cycle:
            fig, ax = plot_multiple_voltage_vs_cycling(
                [np.array(voltage_discharge)],
                [np.array(specific_capacity_discharge)],
                [cycle],
                linestyle="--",
                color_customize=None,
                fig=fig,
                ax=ax,
                colorbar=False,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
            )
        else:
            fig, ax = plot_multiple_voltage_vs_cycling(
                [np.array(voltage_discharge)],
                [np.array(specific_capacity_discharge)],
                [cycle],
                linestyle="-",
                color_customize=None,
                fig=fig,
                ax=ax,
                colorbar=False,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
            )
        if cycle == max(output_dict["cycle_lst"][:max_cycle+1]) or cycle== max_cycle:
            # fig,ax = plot_multiple_voltage_vs_cycling([voltage_charge],[specific_capacity_discharge[-1]-np.array(specific_capacity_charge)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=True,min_cycle=1,max_cycle=max(cell_id_1_cycles))
            print("Plotting charge curve for last cycle")
            fig, ax = plot_multiple_voltage_vs_cycling(
                [voltage_charge],
                [np.array(specific_capacity_charge)],
                [cycle],
                linestyle="-",
                color_customize=None,
                fig=fig,
                ax=ax,
                colorbar=True,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
            )
        elif cycle != 0:
            # fig,ax = plot_multiple_voltage_vs_cycling([voltage_charge],[specific_capacity_discharge[-1]-np.array(specific_capacity_charge)],[cycle],linestyle='-',color_customize=None,fig=fig,ax=ax,colorbar=False,min_cycle=1,max_cycle=max(cell_id_1_cycles))
            fig, ax = plot_multiple_voltage_vs_cycling(
                [voltage_charge],
                [np.array(specific_capacity_charge)],
                [cycle],
                linestyle="-",
                color_customize=None,
                fig=fig,
                ax=ax,
                colorbar=False,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
            )      
    ax.set_ylim([0, 4.5])
    ax.set_xlabel("Specific Capacity (mAh/g-AM)")
    ax.set_ylabel('Voltage (V)')
    plt.tight_layout()
    # add title
    return output_dict, fig, ax


def plot_er_cell(cell_id, output_dict, discharge_se_lst_cell_id_1, fig, ax, save_folder="/scratch/venkvis_root/venkvis/shared_data/herald/all_cycling_plots/"):
    if not fig or not ax:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)

    discharge_se_lst_cell_id_1 = np.array(discharge_se_lst_cell_id_1)
    # discard indices where discharge_se_lst_cell_id_1 is less than 0
    cycle_list = np.array(output_dict["cycle_lst"])[discharge_se_lst_cell_id_1 > 0]
    cycle_list = cycle_list[1:]
    discharge_se_lst_cell_id_1 = discharge_se_lst_cell_id_1[
        discharge_se_lst_cell_id_1 > 0
    ]
    discharge_se_lst_cell_id_1 = discharge_se_lst_cell_id_1[1:]
    if len(discharge_se_lst_cell_id_1) == 0:
        return fig, ax
    er_lst_cell_id_1 = discharge_se_lst_cell_id_1 / discharge_se_lst_cell_id_1[0] * 100
    er_per_cycle_lst_cell_id_1 = [
        discharge_se_lst_cell_id_1[i] / discharge_se_lst_cell_id_1[i - 1] * 100
        for i in range(1, len(discharge_se_lst_cell_id_1))
    ]
    er_per_cycle_lst_cell_id_1.insert(0, 100)  # first cycle is always 100%
    er_lst_cell_id_1 = np.array(er_lst_cell_id_1)
    er_per_cycle_lst_cell_id_1 = np.array(er_per_cycle_lst_cell_id_1)
    # # discard the indices where er_lst_cell_id_1 > 200
    # cycle_list = cycle_list[er_lst_cell_id_1 < 200]
    # discharge_se_lst_cell_id_1 = discharge_se_lst_cell_id_1[er_lst_cell_id_1 < 200]
    # er_per_cycle_lst_cell_id_1 = er_per_cycle_lst_cell_id_1[er_lst_cell_id_1 < 200]
    # er_lst_cell_id_1 = er_lst_cell_id_1[er_lst_cell_id_1 < 200]

    er_lst_cell_id_1 = np.array(er_lst_cell_id_1)
    er_per_cycle_lst_cell_id_1 = np.array(er_per_cycle_lst_cell_id_1)

    # print(er_lst_cell_id_1)
    # print(discharge_se_lst_cell_id_1)
    # print(np.mean(er_per_cycle_lst_cell_id_1))

    # ax.scatter(cycle_list,discharge_se_lst_cell_id_1,label='Discharge',color="tab:blue",edgecolors='k',marker='^',s=120,alpha=0.8)
    # ax.scatter(
    #     cycle_list,
    #     er_lst_cell_id_1,
    #     label="Energy Retention",
    #     color="tab:orange",
    #     edgecolors="k",
    #     marker="o",
    #     alpha=0.8,
    # )
    # calculate chemistry level specific energy
    discharge_se_lst_cell_id_1 = np.array(discharge_se_lst_cell_id_1)
    discharge_se_lst_chem = []
    # for i, se in enumerate(discharge_se_lst_cell_id_1):
        # se, mass_dict = cell_design(se / 1000.0, cell_id=cell_id,print_results=False)
        # discharge_se_lst_chem.append(se)
    print(er_lst_cell_id_1, output_dict['cycle_lst'], discharge_se_lst_cell_id_1)
    ax.scatter(cycle_list,er_lst_cell_id_1,label='Energy Retention',color="tab:orange",edgecolors='k',marker='^')
    ax2 = ax.twinx()
    ax2.scatter(cycle_list, discharge_se_lst_cell_id_1, label='AM level', color="tab:blue", edgecolors='k', marker='o', s=120, alpha=0.5)
    ax2.set_ylabel("Specific Energy (Wh/kg-AM)", color="tab:blue")
    ax.set_ylabel("Energy Retention (%)", color="tab:orange")
    cell_id_1_cycles = output_dict["cycle_lst"]
    max_plot_cycles = min(max(cell_id_1_cycles), 10)
    ax.set_xlim([0, min(int(max(cell_id_1_cycles)) + 1, 10)])
    # ax.set_ylim([20,None])
    # only tick the integer x and tick no more than 5
    n_ticks = 5 if max_plot_cycles > 10 else max_plot_cycles
    ax.set_xticks(
        np.linspace(1, max_plot_cycles, n_ticks).astype(int)
    )

    # ax2.spines["right"].set_color("tab:orange")
    # ax2.spines["left"].set_color("tab:blue")
    ax.tick_params(axis="y", colors="tab:orange")
    # ax.tick_params(axis="y", colors="tab:blue")
    # ax2.set_ylim([0, 150])
    # # ax.set_ylim([400,1500/(82.5/105)]) #2000
    # ax.set_ylim([750, 1850])
    # ax2.set_ylim([80,None])
    # ax2.axhline(y=80,linestyle='--',color='orange',linewidth=2,alpha=1)

    plt.tight_layout()
    # add title
    # ax.set_title(f"Cell ID: {cell_id_1}")
    fig.savefig(
        save_folder + cell_id + "_energy_retention.png", dpi=100, bbox_inches="tight"
    )

    # if len(discharge_se_lst_cell_id_1) > 10:
    #     max_cycle = 10
    #     overall_er = discharge_se_lst_cell_id_1[9] / discharge_se_lst_cell_id_1[0] * 100
    # else:
    #     max_cycle = len(discharge_se_lst_cell_id_1)
    #     overall_er = (
    #         discharge_se_lst_cell_id_1[-1] / discharge_se_lst_cell_id_1[0] * 100
    #     )
    return fig, ax

def save_small_df(output_dict):
    capacities = np.array(output_dict["specific_capacity_discharge_lst"][0]).flatten()
    voltages = np.array(output_dict["voltage_discharge_lst"][0]).flatten()
    powers = np.array(output_dict["specific_power_discharge_lst"][0]).flatten()
    df_out = pd.DataFrame(
        {
            "specific_capacity_discharge": capacities,
            "voltage_discharge": voltages,
            "specific_power_discharge": powers,
        }
    )
    df_out.to_csv('123H_discharge.csv',index=False)    

if __name__ == "__main__":
    
    from scipy.integrate import simpson

    save_folder = "/scratch/venkvis_root/venkvis/shared_data/herald/all_cycling_plots/"
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
    yaml = YAML()
    with open('/scratch/venkvis_root/venkvis/shared_data/herald/list_of_cycling_cell.yaml', 'r') as file:
        df_ids = yaml.load(file)
    ids = df_ids['ids']
    se_am_dict = {}
    sc_am_dict = {}    
    avg_voltage_dict = {}
    ids_cleaned = []
    for cell_id in ids:
        print(
            f"Processing Cell ID: {cell_id}"
        )
        fig1, ax1 = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
        try:
            output_dict, fig1, ax1 = plot_voltage_vs_capacity_single_cell_with_overpotential(
                cell_id, fig1, ax1
            )
            ids_cleaned.append(cell_id)
        except: 
            print(f"Error processing Cell ID {cell_id}, skipping...")
            continue
        if fig1 is None or ax1 is None:
            print(f"No data found for cell ID {cell_id}, skipping...")
            continue
        ax1.set_xlim([-20, 650])
        ax1.set_ylim([0.0, 4.5])
        ax1_2 = ax1.twinx()
        color = 'purple' # plt.get_cmap('Blues')(0.1)
        ax1_2.set_ylabel('Areal Current (mA/cm$^2$)', color=color)  
        ax1_2.tick_params(axis='y', colors=color)
        ax1_2.spines['right'].set_color(color)
        areal_current = -output_dict['areal_current_discharge_lst'][1]
        sc = output_dict['specific_capacity_discharge_lst'][1]
        areal_current = np.array(areal_current)
        sc = np.array(sc)
        ax1_2.plot(sc, areal_current, color=color, linestyle='--', linewidth=2, alpha=1.0)
        ax1_2.set_ylim([0, 2.0])

        # Right axis #2: Power (create new twinx and offset its spine)
        power = -output_dict['specific_power_discharge_lst'][1]
        color = 'tab:orange'
        ax1_3 = ax1.twinx()
        ax1_3.spines["right"].set_position(("outward", 100))  # shift 60 pts away
        ax1_3.plot(sc, power, color=color, linestyle="-.", label="Power")
        ax1_3.set_ylabel("Power (W/kg-AM)", color=color)
        ax1_3.tick_params(axis="y", colors=color)
        ax1_3.spines['right'].set_color(color)
        ax1_3.set_ylim([400, 2000])
        # adjust colorbar position
        for a in fig1.axes:
            if a not in [ax1, ax1_2]:     # exclude your main axis
                # you can filter more specifically, e.g., by checking if it's a Colorbar
                cbar_ax = a
                break

        # shift the colorbar to the right
        pos = cbar_ax.get_position()
        cbar_ax.set_position([pos.x0 + 0.3, pos.y0, pos.width, pos.height])        

        fig1.savefig(
            save_folder + f"{cell_id}_voltage_vs_capacity.png",
            dpi=100,
            bbox_inches="tight",
        )
        if output_dict is None:
            continue
        # calculate specific energy
        (
            discharge_se_lst_cell_id_1,
            discharge_sc_lst_cell_id_1,
            charge_se_lst_cell_id_1,
            charge_sc_lst_cell_id_1,
        ) = calc_se_cell(output_dict)

        # calculate energy retention
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
        fig, ax = plot_er_cell(
            cell_id,
            output_dict,
            discharge_se_lst_cell_id_1,
            fig,
            ax,
            save_folder=save_folder,
        )
        # save_small_df(output_dict)

        discharge_sc_lst_cell_id_1 = np.array(discharge_sc_lst_cell_id_1)
        avg_voltage_lst_cell_id_1 = discharge_se_lst_cell_id_1 / discharge_sc_lst_cell_id_1
        se_am_dict[cell_id] = discharge_se_lst_cell_id_1.tolist()
        sc_am_dict[cell_id] = discharge_sc_lst_cell_id_1.tolist()
        avg_voltage_dict[cell_id] = avg_voltage_lst_cell_id_1.tolist()
        print(f"Discharge SE for {cell_id}: {list(discharge_se_lst_cell_id_1)}")
        print(f'Discharge SC for {cell_id}: {list(discharge_sc_lst_cell_id_1)}')
        print(f'Average voltage for {cell_id}: {list(avg_voltage_lst_cell_id_1)}')
        plt.close(fig1)
        plt.close(fig)

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(
        "/scratch/venkvis_root/venkvis/shared_data/herald/hypo_se_am_dict.yaml", "w"
    ) as file:
        yaml.dump(se_am_dict, file)
    with open(
        "/scratch/venkvis_root/venkvis/shared_data/herald/hypo_sc_am_dict.yaml", "w"
    ) as file:
        yaml.dump(sc_am_dict, file)
    with open(
        "/scratch/venkvis_root/venkvis/shared_data/herald/hypo_vavg_am_dict.yaml", "w"
    ) as file:
        yaml.dump(avg_voltage_dict, file)

