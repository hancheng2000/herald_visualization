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
    "legend.fontsize": 25,  # Adjust the font size of the legend
    "legend.title_fontsize": 25,  # Increase legend title size if needed
    "legend.frameon": False,
}
plt.rcParams.update(default_params)

# cells_df = pd.read_csv('/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing/In-house cells and syntheses - Coin Cells.csv')
# available_ids = cells_df['Test ID'].tolist()

# function for handling id to absolute path
data_path = "/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing"  # here is where data path is taken care of


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

if __name__ == "__main__":
    from scipy.integrate import simpson
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=300)
    hypo_test_dict = {
        '144A':'Vacuum Fill',
        '144B':'Standard Fill',
        '145A': 'Semi Solid',
        '145B': 'Semi Solid',
    }
    save_folder = "/scratch/venkvis_root/venkvis/shared_data/herald/cycling_plots/"
    for cell_id in hypo_test_dict.keys():
        if not os.path.exists(id_to_path(cell_id)) or id_to_path(cell_id) is None:
            print(f"No data found for cell ID {cell_id}")
            # return None
        df = pd.read_csv(id_to_path(cell_id))
        cycle_df = df[df['half cycle'] == 1]
        specific_capacity = cycle_df['Specific Capacity Total AM'].to_numpy()
        valid_idxs = np.where(specific_capacity > 0)[0]
        specific_capacity = specific_capacity[valid_idxs]
        voltage = cycle_df['Voltage'].to_numpy()
        voltage = voltage[valid_idxs]
        ax.plot(specific_capacity, voltage, linewidth=4, label=cell_id+ ": "+hypo_test_dict[cell_id])

    # compare with Hagiwara
    df = pd.read_csv('/scratch/venkvis_root/venkvis/shared_data/herald/363K IL specific capacity vs voltage data.csv')
    specific_capcity = df['specific_capacity (mAh/g) cycle 0'].to_numpy()
    voltage = df['voltage (V) cycle 0'].to_numpy()
    valid_idxs = np.where(specific_capcity > 0)[0]
    specific_capacity = specific_capcity[valid_idxs]
    voltage = voltage[valid_idxs]
    specific_capacity = specific_capacity * 112.84 / 133.66
    ax.plot(specific_capacity, voltage, linewidth=4, label='Hagiwara et al.', linestyle='--', color='k')

    ax.legend()
    ax.set_xlabel("Specific Capacity (mAh/kg-AM)")
    ax.set_ylabel("Voltage (V)")
    # ax.set_title(f"Cell: {cell_id}")

    fig.savefig(
        save_folder + f"{cell_id}_voltage_vs_capacity.png",
        dpi=300,
        bbox_inches="tight",
    )

            # calculate energy retention
            # fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=300)
            # fig, ax = plot_er_cell(
            #     cell_id,
            #     output_dict,
            #     discharge_se_lst_cell_id_1,
            #     fig,
            #     ax,
            #     save_folder=save_folder,
            # )

            # # calculate chem level energy density
            # discharge_se_lst_chem = []
        #     for i, se in enumerate(discharge_se_lst_cell_id_1):
        #         chem_se = cell_design(se / 1000.0, cell_id=cell_id)
        #         discharge_se_lst_chem.append(chem_se)
        #     discharge_se_lst_chem = np.array(discharge_se_lst_chem)
        #     print(f"Discharge SE for {cell_id}: {discharge_se_lst_cell_id_1}")
        #     print(f"Discharge SE chem for {cell_id}: {discharge_se_lst_chem}")
        #     # plot discharge_se_lst_chem
        #     ax2.plot(
        #         output_dict["cycle_lst"],
        #         discharge_se_lst_chem,
        #         label=f"{cell_id}, {hypo_test_dict[hypo]['cell_ids_and_specs'][cell_id]}",
        #         mec="k",
        #         marker="o",
        #         linestyle="--",
        #         markersize=10,
        #         alpha=0.8,
        #     )
        #     if max(output_dict["cycle_lst"]) > max_cell_id_1_cycles:
        #         max_cell_id_1_cycles = max(output_dict["cycle_lst"])
        #         print(max_cell_id_1_cycles)
        #     plt.close(fig1)
        # ax2.set_xlabel("Cycle Number")
        # ax2.set_ylabel("Specific Energy (Wh/kg-Chem)")
        # # set legend on side of the figure
        # ax2.legend(
        #     loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False
        # )
        # n_ticks = 5 if max_cell_id_1_cycles > 10 else max_cell_id_1_cycles
        # ax2.set_xticks(np.linspace(1, max_cell_id_1_cycles, n_ticks).astype(int))
        # ax2.set_xlim([0.5, 3.5])
        # fig2.suptitle(f'Track {hypo}: {hypo_test_dict[hypo]["Track"]}')
        # fig2.savefig(
        #     save_folder + f"hypothesis_{hypo}_chem_energy_density.png",
        #     dpi=200,
        #     bbox_inches="tight",
        # )
        # plt.close(fig2)


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
