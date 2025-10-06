import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
import argparse
from herald_visualization.fancy_plot import (
    voltage_vs_capacity_cycling,
    plot_multiple_voltage_vs_cycling,
)
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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

if __name__ == '__main__':
    yaml = YAML()
    with open('/scratch/venkvis_root/venkvis/shared_data/herald/hypo_se_am_dict.yaml','r') as f:
        hypo_se_am_dict = yaml.load(f)
    with open('/scratch/venkvis_root/venkvis/shared_data/herald/hypo_testing.yaml','r') as f:
        hypo_testing_dict = yaml.load(f)

    norm = Normalize(vmin=1, vmax=10)
    sm = ScalarMappable(cmap='Blues_r', norm=norm)
    colors = ['tab:blue','tab:red','tab:orange','tab:purple','tab:brown','tab:olive']*1000
    linestyles = [':','-.','--','-']*1000
    markers = ['o','v','^','s','*','<','>','x','p']*1000
    for key in hypo_testing_dict.keys():
        print(hypo_testing_dict[key]['cell_ids_and_specs'])
        cell_ids = hypo_testing_dict[key]['cell_ids_and_specs'].keys()
        # take these cell_ids and create a new dictionary from hypo_se_am_dict
        se_dict = {cell_id: hypo_se_am_dict[cell_id] for cell_id in cell_ids if cell_id in hypo_se_am_dict}
        fig, ax = plt.subplots(figsize=(10, 10))
        cell_ids = list(se_dict.keys())
        try:
            min_cell_id = int(cell_ids[0][:-1])
        except IndexError:
            print(f"Hypo {key}: Cell IDs not found")
            continue
        latest_cell_id = int(cell_ids[0][:-1])
        color_idx = 0
        marker_idx = 0
        linestyle_idx = 0
        cell_id_families = set(int(cell_id[:-1]) for cell_id in cell_ids)
        # group cell_ids by family, key equal to int(cell_id[:-1])
        cell_id_family_dict = {family: [cell_id for cell_id in cell_ids if int(cell_id[:-1]) == family] for family in cell_id_families}
        max_energy_ids = []
        # for cell_id_family in sorted(cell_id_family_dict.keys()):
        #     # find the cell_id corresponding to max energy in the family
        #     family_cell_ids = cell_id_family_dict[cell_id_family]
        #     max_energy = -1
        #     max_energy_cell_id = None
        #     for cell_id in family_cell_ids:
        #         if cell_id in se_dict:
        #             energy = max(se_dict[cell_id][1:])
        #             if energy > max_energy:
        #                 max_energy = energy
        #                 max_energy_cell_id = cell_id
        #     if max_energy_cell_id is not None:
        #         family_cell_ids.remove(max_energy_cell_id)
        #         family_cell_ids.insert(0, max_energy_cell_id)
        #         max_energy_ids.append(max_energy_cell_id)
        max_energy = -1
        min_energy = 1e6
        for cell_id in cell_ids:
            se = se_dict[cell_id][1:] # skip formation cycle
            se = np.array(se)
            er = se / se.max() * 100
            cycles = np.arange(1, len(se) + 1)
            # for i, e in enumerate(se):
            #     # color = sm.cmap(norm(cycles[i]))
            #     linestyle_idx = ord(cell_id[-1]) - ord('A')
            #     marker_idx = ord(cell_id[-1]) - ord('A')
            #     cell_id_family = int(cell_id[:-1])
            #     if cell_id_family > latest_cell_id:
            #         color_idx += 1
            #         latest_cell_id = cell_id_family
            #     if i < len(se) - 1:
            #         ax.scatter(se[i], er[i], color=colors[color_idx], marker=markers[marker_idx], s=100, edgecolors='black',alpha=0.8,)
            #     else:
            #         spec = hypo_testing_dict[key]['cell_ids_and_specs'][cell_id]
            #     ax.scatter(se[i], er[i], color=colors[color_idx], marker=markers[marker_idx], s=100, edgecolors='black',alpha=0.8,label=f'{cell_id}:{spec}')
            linestyle_idx = ord(cell_id[-1]) - ord('A')
            # marker_idx = ord(cell_id[-1]) - ord('A')
            cell_id_family = int(cell_id[:-1])
            if cell_id_family > latest_cell_id:
                color_idx += 1
                marker_idx += 1
                latest_cell_id = cell_id_family
            er_cap = 80
            max_cycle_above_80 = np.where(er>=er_cap)[0]
            ax.plot(se, er, color=colors[color_idx], linestyle=linestyles[linestyle_idx], marker=markers[marker_idx], markersize=20, markeredgecolor='black', label=f'{cell_id}:{hypo_testing_dict[key]["cell_ids_and_specs"][cell_id]}\n{len(max_cycle_above_80)} of {len(er)} cycles above {er_cap}%')
            se = se[er>=er_cap]
            if len(se)>1 and se.max() > max_energy:
                max_energy = se.max()
            if len(se)>1 and se.min() < min_energy:
                min_energy = se.min()
        # cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height] in figure coordinates
        # cbar = fig.colorbar(sm, cax=cax)
        # cbar.set_label("Cycle Number")
        ax.set_xlabel("Specific Energy (Wh/kg-AM)")
        ax.set_ylabel('Energy Retention (%)')
        ax.set_ylim([er_cap, 105])
        
        # if min_energy < max_energy:
        #     ax.set_xlim([min_energy*0.95, max_energy*1.05])
        # else:
        #     ax.set_xlim([None, 1200])
        ax.set_xlim([600, 1200])
        ax.set_title(f'Track {key}: {hypo_testing_dict[key]["Track"]}')
        ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left', labelspacing=1.2)
        fig.savefig(f'/scratch/venkvis_root/venkvis/shared_data/herald/email_cycling_plots/hypothesis_{key}_se_er.png', bbox_inches='tight', dpi=100)
        plt.close(fig)