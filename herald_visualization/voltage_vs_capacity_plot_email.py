import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
import argparse
import shutil

# import herald_visualization.echem as ec
# from herald_visualization.mpr2csv import cycle_mpr2csv
# from herald_visualization.plot import plot_cycling, plot_gitt, parse_cycle_csv, plot_cycling_plotly
from herald_visualization.fancy_plot import (
    voltage_vs_capacity_cycling,
    plot_multiple_voltage_vs_cycling,
)
from ruamel.yaml import YAML
from scipy.optimize import curve_fit

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

def cell_design(
    E_spec_Wh_per_g,
    cell_id,
    cell_design_input_csv="/scratch/venkvis_root/venkvis/shared_data/herald/herald_visualization/herald_visualization/celldesignroutine/In-house cells and syntheses - cell-design-input.csv",
    print_results = False,
):
    """
    Deprecated.
    Determine the cell design based on the specific energy in Wh/kg-AM
    """
    df = pd.read_csv(cell_design_input_csv)
    df = df[df["cell id"] == cell_id].copy()
    cathode_thickness = df["cathode thickness"].values[0]  # um
    cathode_composite_density = 1 / (
        df["cathode FeF3 mass fraction"].values[0] / df["FeF3 density"].values[0]
        + df["cathode carbon mass fraction"].values[0] / df["conductive additive density"].values[0]
        + df["cathode binder mass fraction"].values[0] / df["binder density"].values[0]
    )
    area = np.pi * (df['cathode diameter'].values[0] / 2)**2 * 1e-4
    if not np.isnan(df['Cathode Mass (mg)'].values[0]):
        cathode_thickness = df['Cathode Mass (mg)'].values[0] * 1e-6 / cathode_composite_density / 1e3 / area / (1.0-df['cathode porosity'].values[0]) * 1e6
    m_cathode_composite = df['Cathode Mass (mg)'].values[0] * 1e-6  # kg
    m_fef3 = m_cathode_composite * df["cathode FeF3 mass fraction"].values[0]  # kg
    if np.isnan(df['Li mass'].values):
        m_li = m_fef3 * 1000 / 112.84 * 3.0 * 6.94 / 1000  # kg, assuming 3 Li per FeF3
    else:
        m_li = df['Li mass'].values[0] * 1e-3  # kg, from the csv file
    m_al = (
        df["Current collector thickness - Al"].values[0] * 1e-6
        * area
        * df['Current collector density - Al'].values[0] * 1000
    )  # kg
    m_cu = (
        df["Current collector thickness - Cu"].values[0] * 1e-6
        * area
        * 8.96e3
    )  # kg
    m_separator = (
        df["separator density"].values[0] * 1e3
        * area
        * df["separator thickness"].values[0] * 1e-6
    )  # kg
    v_electrolyte = (
        df["separator porosity"].values[0]
        * df["separator thickness"].values[0] * 1e-6
        * area
        + area
        * cathode_thickness * 1e-6
        * df["cathode porosity"].values[0]
    )  # m³
    m_electrolyte = v_electrolyte * df["electrolyte density"].values[0] * 1000  # kg
    E_total = E_spec_Wh_per_g * (m_fef3 + m_li) * 1000  # total energy in Wh
    m_all = m_cathode_composite + m_al + m_li + m_cu + m_separator + m_electrolyte  # total mass in kg
    # m_all = m_cathode_composite + m_al + m_li # just cathode side
    grav_Wh_per_kg = np.array(E_total) / m_all  # Wh/kg
    mass_dict = {
        "m_cathode_composite": m_cathode_composite,
        "m_fef3": m_fef3,
        "m_li": m_li,
        "m_al": m_al,
        "m_cu": m_cu,
        "m_separator": m_separator,
        "m_electrolyte": m_electrolyte,
        "m_all": m_all,
        "E_total_Wh": E_total,
        "grav_Wh_per_kg": grav_Wh_per_kg,
        'Cathode Thickness (um)': cathode_thickness
    }
    if print_results is True:
        print('mass breakdown:')
        print(f'Cathode composite mass: {m_cathode_composite*1000:.10f} g, mass percentage : {m_cathode_composite/m_all*100:.10f}%')
        print(f"FeF3 mass: {m_fef3*1000:.10f} g, mass percentage : {m_fef3/m_all*100:.10f}%")
        print(f"Li mass: {m_li*1000:.10f} g, mass percentage : {m_li/m_all*100:.10f}%")
        print(f"Al mass: {m_al*1000:.10f} g, mass percentage : {m_al/m_all*100:.10f}%")
        print(f"Cu mass: {m_cu*1000:.10f} g, mass percentage : {m_cu/m_all*100:.10f}%")
        print(f"Separator mass: {m_separator*1000:.10f} g, mass percentage : {m_separator/m_all*100:.10f}%")
        print(f"Electrolyte mass: {m_electrolyte*1000:.10f} g, mass percentage : {m_electrolyte/m_all*100:.10f}%")
        print(f'AM grav Wh/kg: {E_spec_Wh_per_g*1000:.10f}')
        print(f"Total mass: {m_all*1000:.10f} g, Chem grav Wh/kg: {grav_Wh_per_kg:.10f}")
        print(f'Cathode thickness used: {cathode_thickness:.2f} um')
    return grav_Wh_per_kg, mass_dict


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
    if id_to_path(cell_id) is None or (not os.path.exists(id_to_path(cell_id))):
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
    ax.set_ylabel('Voltage')
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



def plot_er_cell(cell_id, output_dict, discharge_se_lst_cell_id_1, fig, ax, save_folder="/scratch/venkvis_root/venkvis/shared_data/herald/email_cycling_plots/"):
    if not fig or not ax:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)

    discharge_se_lst_cell_id_1 = np.array(discharge_se_lst_cell_id_1)
    # discard indices where discharge_se_lst_cell_id_1 is less than 0
    cycle_list = np.array(output_dict["cycle_lst"])[discharge_se_lst_cell_id_1 > 0]
    discharge_se_lst_cell_id_1 = discharge_se_lst_cell_id_1[
        discharge_se_lst_cell_id_1 > 0
    ]
    cycle_list = cycle_list[1:]
    discharge_se_lst_cell_id_1 = discharge_se_lst_cell_id_1[1:]
    if len(discharge_se_lst_cell_id_1) == 0:
        return fig, ax
    er_lst_cell_id_1 = discharge_se_lst_cell_id_1 / discharge_se_lst_cell_id_1[0] * 100
    er_per_cycle_lst_cell_id_1 = [
        discharge_se_lst_cell_id_1[i] / discharge_se_lst_cell_id_1[i - 1] * 100
        for i in range(2, len(discharge_se_lst_cell_id_1))
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

def fit_2rc(t, V, current):
    def rc_func(t, RC1, V1, RC2, V2, Vocv):
        return Vocv - V1 * np.exp(-t / RC1) - V2 * np.exp(-t / RC2)
    # initial guess for R, C, V0
    V1_guess = 0.8
    RC1_guess = 300
    V2_guess = 0.4
    RC2_guess = 30
    Vocv_guess = 2.0
    t = t - t.min()
    Rint = -(V[1]-V[0]) / current[0] * 1000
    popt, pcov = curve_fit(rc_func, t[1:], V[1:], p0=[RC1_guess, V1_guess, RC2_guess, V2_guess, Vocv_guess], maxfev=10000)
    RC1_fit, V1_fit, RC2_fit, V2_fit, Vocv_fit = popt
    R1 = -V1_fit / current[0] * 1000
    C1 = RC1_fit / R1
    R2 = -V2_fit / current[0] * 1000
    C2 = RC2_fit / R2
    rmse = np.sqrt(np.mean((V - rc_func(t, *popt))**2))
    return Rint, R1, C1, R2, C2, Vocv_fit, rmse


def gen_summary_df(output_dict, cell_id, cycles=None):
    summary_csv = os.path.join('/'.join(id_to_path(cell_id).split('/')[:-1]),'cycle_summary.csv')
    summary_df = pd.read_csv(summary_csv)
    summary_df = summary_df[['cycle','CE','Discharge Overpotential','Charge Overpotential',]]
    if not cycles:
        cycles = output_dict['cycle_lst']
    df = pd.read_csv(id_to_path(cell_id))
    summary_df['Discharge Rint (Ohm)'] = np.zeros(len(summary_df))
    summary_df['Discharge R1 (Ohm)'] = np.zeros(len(summary_df))
    summary_df['Discharge C1 (F)'] = np.zeros(len(summary_df))
    summary_df['Discharge R2 (Ohm)'] = np.zeros(len(summary_df))
    summary_df['Discharge C2 (F)'] = np.zeros(len(summary_df))
    summary_df['Discharge Vocv (V)'] = np.zeros(len(summary_df))
    summary_df['Discharge RMSE (V)'] = np.zeros(len(summary_df))
    summary_df['Charge Rint (Ohm)'] = np.zeros(len(summary_df))
    summary_df['Charge R1 (Ohm)'] = np.zeros(len(summary_df))
    summary_df['Charge C1 (F)'] = np.zeros(len(summary_df))
    summary_df['Charge R2 (Ohm)'] = np.zeros(len(summary_df))
    summary_df['Charge C2 (F)'] = np.zeros(len(summary_df))
    summary_df['Charge Vocv (V)'] = np.zeros(len(summary_df))
    summary_df['Charge RMSE (V)'] = np.zeros(len(summary_df))
    ############# fit 2rc model to each cycle #############
    for i, cycle in enumerate(cycles[1:]):
        df1 = df[df['full cycle']==cycle].copy()
        half_cycles = df1['half cycle'].unique().tolist()
        for half_cycle in half_cycles:
            df2 = df1[df1['half cycle']==half_cycle].copy()
            df2.reset_index(drop=True, inplace=True)
            rest_idx = df2.index[df2['state']==0].tolist()[0]
            df2 = df2.iloc[rest_idx-1:]
            if half_cycle % 2 == 1:
                t = df2['Time'].to_numpy()
                t = t - t.min()
                V = df2['Voltage'].to_numpy()
                I = df2['Current'].to_numpy()
                Rint, R1, C1, R2, C2, Vocv, rmse = fit_2rc(t, V, I)
                summary_df.loc[summary_df['cycle']==cycle, 'Discharge Rint (Ohm)'] = Rint
                summary_df.loc[summary_df['cycle']==cycle, 'Discharge R1 (Ohm)'] = R1
                summary_df.loc[summary_df['cycle']==cycle, 'Discharge C1 (F)'] = C1
                summary_df.loc[summary_df['cycle']==cycle, 'Discharge R2 (Ohm)'] = R2
                summary_df.loc[summary_df['cycle']==cycle, 'Discharge C2 (F)'] = C2
                summary_df.loc[summary_df['cycle']==cycle, 'Discharge Vocv (V)'] = Vocv
                summary_df.loc[summary_df['cycle']==cycle, 'Discharge RMSE (V)'] = rmse
            else:
                t = df2['Time'].to_numpy()
                V = df2['Voltage'].to_numpy()
                I = df2['Current'].to_numpy()
                Rint, R1, C1, R2, C2, Vocv, rmse = fit_2rc(t, V, I)
                summary_df.loc[summary_df['cycle']==cycle, 'Charge Rint (Ohm)'] = Rint
                summary_df.loc[summary_df['cycle']==cycle, 'Charge R1 (Ohm)'] = R1
                summary_df.loc[summary_df['cycle']==cycle, 'Charge C1 (F)'] = C1
                summary_df.loc[summary_df['cycle']==cycle, 'Charge R2 (Ohm)'] = R2
                summary_df.loc[summary_df['cycle']==cycle, 'Charge C2 (F)'] = C2
                summary_df.loc[summary_df['cycle']==cycle, 'Charge Vocv (V)'] = Vocv
                summary_df.loc[summary_df['cycle']==cycle, 'Charge RMSE (V)'] = rmse
    return summary_df


def chemistry_se(output_dict, cell_id):
    (discharge_se_lst_cell_id_1,
    discharge_sc_lst_cell_id_1,
    charge_se_lst_cell_id_1,
    charge_sc_lst_cell_id_1,) = calc_se_cell(output_dict)
    spec_df = pd.read_csv('/scratch/venkvis_root/venkvis/shared_data/herald/In-house cells and syntheses - cell-design-input.csv')
    coeffs = {"70_3": 0.63,
    "75_3": 0.67,
    "80_3": 0.71,
    "85_1": 0.77,
    "85_2": 0.77,
    "85_3": 0.76,
    "85_5": 0.76,
    "87_3": 0.77,
    } # AM to chem coefficient
    df1 = spec_df[spec_df['cell id']==cell_id].copy()
    fef3_mass = int(df1['cathode FeF3 mass fraction'].values[0]*100)
    binder_mass = int(df1['cathode binder mass fraction'].values[0]*100)
    coeff = coeffs[f'{fef3_mass}_{binder_mass}']
    discharge_se_lst_chem = discharge_se_lst_cell_id_1 * coeff
    return discharge_se_lst_chem

if __name__ == "__main__":
    
    from scipy.integrate import simpson

    save_folder = "/scratch/venkvis_root/venkvis/shared_data/herald/email_cycling_plots/"
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
    yaml = YAML()
    with open(
        "/scratch/venkvis_root/venkvis/shared_data/herald/hypo_testing.yaml", "r"
    ) as file:
        hypo_test_dict = yaml.load(file)
    
    with open('/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_se_chem_dict.yaml','r') as file:
        hypo_se_chem_dict = yaml.load(file)
    
    with open('/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_se_am_dict.yaml', 'r') as file:
        hypo_se_am_dict = yaml.load(file)
    
    with open('/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_sc_am_dict.yaml', 'r') as file:
        hypo_sc_am_dict = yaml.load(file)
    
    with open('/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_vavg_am_dict.yaml', 'r') as file:
        hypo_vavg_am_dict = yaml.load(file)

    for hypo in hypo_test_dict.keys():
        se_am_dict = {}
        sc_am_dict = {}
        print(f"Plotting for Track {hypo}: {hypo_test_dict[hypo]['Track']}")
        cell_ids = list(hypo_test_dict[hypo]["cell_ids_and_specs"].keys())
        print(cell_ids)
        fig2, ax2 = plt.subplots(1, 1, figsize=(6,6), dpi=100)
        fig3, ax3 = plt.subplots(1, 1, figsize=(6,6), dpi=100)
        fig4, ax4 = plt.subplots(1, 1, figsize=(6,6), dpi=100)
        fig5, ax5 = plt.subplots(1, 1, figsize=(6,6), dpi=100)
        max_cell_id_1_cycles = 0
        colors = ['tab:blue','tab:red','tab:orange','tab:purple','tab:brown','tab:olive']*1000
        linestyles = [':','-.','--','-']*1000
        markers = ['o','v','^','s','*','+','<','>','x','p']*1000
        min_cell_id = int(cell_ids[0][:-1])
        latest_cell_id = int(cell_ids[0][:-1])
        color_idx = 0
        marker_idx = 0
        linestyle_idx = 0
        for ii, cell_id in enumerate(cell_ids):
            print(
                f"Processing Cell ID: {cell_id} with spec {hypo_test_dict[hypo]['cell_ids_and_specs'][cell_id]}"
            )
            fig1, ax1 = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
            png_file = f'/scratch/venkvis_root/venkvis/shared_data/herald/all_cycling_plots/{cell_id}_voltage_vs_capacity.png'
            # if os.path.exists(png_file):
            #     shutil.copy(png_file, save_folder)
            discharge_se_lst_cell_id_1 = hypo_se_am_dict[cell_id]
            discharge_se_lst_chem = hypo_se_chem_dict[cell_id]
            discharge_sc_lst_cell_id_1 = hypo_sc_am_dict[cell_id]
            avg_voltage_lst_cell_id_1 = hypo_vavg_am_dict[cell_id]

            discharge_se_lst_chem = np.array(discharge_se_lst_chem)
            discharge_sc_lst_cell_id_1 = np.array(discharge_sc_lst_cell_id_1)
            avg_voltage_lst_cell_id_1 = discharge_se_lst_cell_id_1 / discharge_sc_lst_cell_id_1

            linestyle_idx = ord(cell_id[-1]) - ord('A')
            cell_id_family = int(cell_id[:-1])
            if cell_id_family > latest_cell_id:
                color_idx += 1
                marker_idx += 1
                latest_cell_id = cell_id_family
            if cell_id_family == 276: # 276 series have a lot of cells
                color_marker_dict = {
                    'A': [0,0],
                    'B': [0,1],
                    'C': [0,2],
                    'D': [1,0],
                    'E': [1,1],
                    'F': [1,2],
                    'G': [2,0],
                    'H': [2,1],
                    'I': [2,2],
                    'J': [2,0],
                    'K': [2,1],
                    'L': [2,2],
                }
                color_idx, marker_idx = color_marker_dict[cell_id[-1]]
            ax2.plot(
                range(1, len(discharge_sc_lst_cell_id_1)),
                discharge_sc_lst_cell_id_1[1:],
                label=f"{cell_id}, {hypo_test_dict[hypo]['cell_ids_and_specs'][cell_id]}",
                color=colors[color_idx],
                mec="k",
                linestyle=linestyles[linestyle_idx],
                marker = markers[marker_idx],
                markersize=10,
                alpha=0.8,
            )    
            ax3.plot(
                range(1, len(discharge_se_lst_chem)),
                discharge_se_lst_chem[1:] / discharge_se_lst_chem[1] * 100,
                label=f"{cell_id}, {hypo_test_dict[hypo]['cell_ids_and_specs'][cell_id]}",
                color=colors[color_idx],
                mec="k",
                linestyle=linestyles[linestyle_idx],
                marker = markers[marker_idx],
                markersize=10,
                alpha=0.8,
            )

            ax4.plot(
                range(1, len(discharge_se_lst_cell_id_1)),
                discharge_se_lst_cell_id_1[1:],
                label=f"{cell_id}, {hypo_test_dict[hypo]['cell_ids_and_specs'][cell_id]}",
                color=colors[color_idx],
                mec="k",
                linestyle=linestyles[linestyle_idx],
                marker = markers[marker_idx],
                markersize=10,
                alpha=0.8,        
            )        
            ax5.plot(
                range(1, len(discharge_se_lst_chem)),
                discharge_se_lst_chem[1:],
                label=f"{cell_id}, {hypo_test_dict[hypo]['cell_ids_and_specs'][cell_id]}",
                color=colors[color_idx],
                mec="k",
                linestyle=linestyles[linestyle_idx],
                marker = markers[marker_idx],
                markersize=10,
                alpha=0.8,        
            )
            plt.close(fig1)

            # # # generate and save summary df
            # # try:
            # summary_df = gen_summary_df(output_dict, cell_id, cycles = [0,1])
            # summary_df = summary_df.round(3)
            # summary_df.to_csv(save_folder + f"{cell_id}_summary.csv", index=False)
            # # except:
            #     # print(f'cell id {cell_id} cannot generate summary df, skipping')
            #     # continue


        ax2.set_xlabel("Cycle Number")
        # ax2.set_ylabel("Specific Energy (Wh/kg-AM)")
        ax2.set_ylabel('Specific Capacity (mAh/kg-AM)')
        ax2.set_ylim([300, 650])
        # set legend on side of the figure
        ax2.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
        )
        max_cell_id_1_cycles = 10
        # n_ticks = 5 if max_cell_id_1_cycles > 10 else max_cell_id_1_cycles
        n_ticks = 10
        ax2.set_xticks(np.linspace(1, max_cell_id_1_cycles, n_ticks).astype(int))
        ax2.set_xlim([0.5, 10.5])
        fig2.suptitle(f'Track {hypo}: {hypo_test_dict[hypo]["Track"]}')
        fig2.savefig(
            save_folder + f"hypothesis_{hypo}_AM_sc.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig2)
        ax3.set_xlabel("Cycle Number")
        ax3.set_ylabel('Energy Retention (from cycle 1) (%)')
        ax3.set_xlim([0.5, 10.5])
        ax3.set_ylim([80, 105])
        ax3.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
        )
        max_cell_id_1_cycles = 10
        # n_ticks = 5 if max_cell_id_1_cycles > 10 else max_cell_id_1_cycles
        # n_ticks = 10
        # ax3.set_xticks(np.linspace(1, max_cell_id_1_cycles, n_ticks).astype(int))
        ax3.set_xlim([0.5, 10.5])
        fig3.suptitle(f'Track {hypo}: {hypo_test_dict[hypo]["Track"]}')
        fig3.savefig(
            save_folder + f"hypothesis_{hypo}_chem_er.png",
            dpi=200,
            bbox_inches="tight",
        )      
        plt.close(fig3)  
        ax4.set_xlabel("Cycle Number")
        ax4.set_ylabel('Specific Energy (Wh/kg-AM)')
        # ax4.set_ylim([600, 1300])
        # ax4.set_xlim([0, 5])
        ax4.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
        )
        max_cell_id_1_cycles = 10
        # n_ticks = 5 if max_cell_id_1_cycles > 10 else max_cell_id_1_cycles
        n_ticks = 10
        # ax4.set_xticks(np.linspace(1, 70, max_cell_id_1_cycles).astype(int))
        ax4.set_xlim([0.5, 10.5])
        fig4.suptitle(f'Track {hypo}: {hypo_test_dict[hypo]["Track"]}')
        fig4.savefig(
            save_folder + f"hypothesis_{hypo}_AM_se.png",
            dpi=200,
            bbox_inches="tight",
        ) 
        plt.close(fig4)
        ax5.set_xlabel("Cycle Number")
        ax5.set_ylabel('Specific Energy (Wh/kg-Chem)')
        ax5.set_ylim([600, 1000])
        # ax5.set_xlim([0, 5])
        ax5.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
        )
        max_cell_id_1_cycles = 10
        # n_ticks = 5 if max_cell_id_1_cycles > 10 else max_cell_id_1_cycles
        n_ticks = 10
        # ax5.set_xticks(np.linspace(1, 70, max_cell_id_1_cycles).astype(int))
        ax5.set_xlim([0.5, 10.5])
        fig5.suptitle(f'Track {hypo}: {hypo_test_dict[hypo]["Track"]}')
        fig5.savefig(
            save_folder + f"hypothesis_{hypo}_chem_se.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig5)

    # yaml = YAML()
    # yaml.indent(mapping=2, sequence=4, offset=2)
    # with open(
    #     "/scratch/venkvis_root/venkvis/shared_data/herald/hypo_se_am_dict.yaml", "w"
    # ) as file:
    #     yaml.dump(se_am_dict, file)
    # with open(
    #     "/scratch/venkvis_root/venkvis/shared_data/herald/hypo_sc_am_dict.yaml", "w"
    # ) as file:
    #     yaml.dump(sc_am_dict, file)
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
