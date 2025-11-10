import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
from ruamel.yaml import YAML
from scipy.integrate import simpson
from galvani import BioLogic
import matplotlib as mpl
import matplotlib.font_manager as fm
from ruamel.yaml import YAML
import plotly.io as pio

# Your palettes and markers (Plotly names)
PL_COLORS  = ['tab:blue','orange','green','tab:brown','purple'] * 1000
PL_MARKERS = ['circle','triangle-up','square','star','diamond','triangle-down'] * 1000
BASELINE_COLOR = 'grey'
BAD_COLOR      = 'red'
HIGHLIGHT_COLOR= 'blue'

def plot_family_energy_retention(
    cell_id_list,
    cell_id_info_dict,
    baseline_cell_ids = ['214F'],
    yaml_path="/Users/mac/Research/herald/cell_id_energy_list/hypo_se_chem_dict.yaml",
    threshold=80,
    energy_threshold=800,
    fig=None,
    **kwargs,
):
    """
    Plotly scatter of (x, y) per cell:
      x = max energy
      y = max retention cycles
    Colored by numeric family (e.g., 227 from 227A).

    Robust YAML parsing + helper-calling with graceful fallbacks.

    Parameters
    ----------
    cell_ids : list[str]
        e.g., ['227A','227B','228A','228B','229A','229B']
    yaml_path : str
        Path to YAML mapping with energies for each cell id.
        Accepts entries like:
          - {cell_id: [energies...]}
          - {cell_id: {"energies": [...]}}
          - {cell_id: {"energy": [... or single number]}}
    find_max_num_cycles_above_threshold : callable | None
        Your helper. We try calling it as:
          (energies,) or (energies, threshold) or (energies, threshold, energy_threshold)
        and accept either a (x,y) tuple or a dict with keys like
        {'max_energy': ..., 'max_cycles': ...}.
    threshold, energy_threshold : numbers
        Passed to the helper if its signature supports them.
        Also used for a sensible fallback if the helper isn’t available.
    fig : plotly.graph_objects.Figure | None
        Add to an existing figure or create a new one.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import re
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    def _family(cid: str) -> str:
        m = re.match(r"^(\d+)", str(cid))
        return m.group(1) if m else str(cid)

    yaml = YAML()
    with open(yaml_path, "r") as f:
        hypo_se_chem_dict = yaml.load(f)

    def _extract_energies(cell_id):
        if cell_id is None:
            return []
        energies = hypo_se_chem_dict[cell_id]
        return energies

    # Build data
    rows = []
    for cid in cell_id_list:
        if cid in abnormal_cell_ids:
            continue
        energies = _extract_energies(cid)
        energies = energies[1:]
        start_idx, end_idx = find_max_num_cycles_above_threshold(energies, threshold=threshold)
        max_energy = float(np.nanmax(energies[start_idx:end_idx+1]))  # recalc max_energy over the selected range
        # if max_energy <= 500:
        #     continue
        max_idx = np.nanargmax(energies[start_idx:end_idx+1]) + start_idx
        n80 = end_idx - start_idx + 1 if end_idx >= start_idx else 0
        if max_energy < energy_threshold:
            continue
        x = max_energy
        y = n80
        if cid == '214F':
            rows.append({"cell_id": str(cid), "family": _family(cid), "x": 769, "y": 2, "n_points": len(energies)})
            continue
        rows.append({"cell_id": str(cid), "family": _family(cid), "x": x, "y": y, "n_points": len(energies)})

    df = pd.DataFrame(rows)
    df1 = df[~df['cell_id'].isin(baseline_cell_ids)].copy()
    families = df1["family"].unique().tolist()
    alpha = kwargs.get('alpha', 1.0)
    palette = [
        f'rgba(31, 119, 180, {alpha:.1f})',
        f'rgba(255, 127, 14, {alpha:.1f})',
        f'rgba(0,128,0,{alpha:.1f})',
        f'rgba(139, 69, 19, {alpha:.1f})',
        f'rgba(128,0,128,{alpha:.1f})',
    ]

    # map each family to one of the five tab colors (repeats if >5)
    color_map = {fam: palette[i % len(palette)] for i, fam in enumerate(families)}    
    print(color_map)

    # Plot
    if fig is None:
        fig = go.Figure()

    # if df.empty:
    #     fig.add_annotation(text="No data found for provided cell IDs.", x=0.5, y=0.5,
    #                        xref="paper", yref="paper", showarrow=False)
    #     fig.update_layout(template="plotly_white")
    #     return fig

    df1 = df[~df['cell_id'].isin(baseline_cell_ids)].copy()
    for fam in families:
        sub = df1[df1["family"] == fam]
        fig.add_trace(go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=cell_id_info_dict[fam],
            marker=dict(symbol=kwargs.get('markerstyle','circle'),size=16, color=color_map[fam],
                        line=dict(color="black", width=1.2)),
            hovertemplate=(
                "Family: " + fam + "<br>"
                "GED: %{x:.2f}Wh/kg-chem<br>"
                "Max Retention: %{y:d} cycles<extra></extra>"
            ),
        ))
    # plot baseline cells with special marker
    df1 = df[df['cell_id'].isin(baseline_cell_ids)]
    for i, row in df1.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['x']],
            y=[row['y']],
            mode='markers',
            name = 'Baseline {}'.format(row['cell_id']),
            marker=dict(symbol='star', size=20, color='grey'),
            hovertemplate=f"Cell ID: {row['cell_id']}<br>Family: {row['family']}<br>Max Energy: {row['x']:.3f}<br>Max Retention Cycles: {row['y']:.3f}<extra></extra>"
        ))        

    fig.update_traces(textposition="top center")
    fig.update_layout(
        # title="Energy vs. Retention Cycles by Cell Family",
        xaxis_title="GED (Wh/kg-chem)",
        yaxis_title=">{}% ER Cycles".format(threshold),
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig

def apply_mpl_like_style(
    fig,
    font_family="Poppins",  # matches your Poppins (with safe fallbacks)
    font_size=32,
    width=940,
    height=840,
    axis_line_width=2.5,
    tick_len=8,
    tick_width=2,
    legend_on=True
):
    # Canvas + global font
    fig.update_layout(
        template=None,                 # no built-in Plotly theme
        width=width, height=height,
        font=dict(family=font_family, size=font_size),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=legend_on,
        legend=dict(                    # frameless legend
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=font_size-4)
        ),
        margin=dict(l=80, r=40, t=20, b=80)
    )

    # Axes: full black box (mirror), ticks outside, no grids/zerolines
    ax_common = dict(
        showline=True, linecolor="black", linewidth=axis_line_width,
        mirror=True,                    # draws top & right spines too
        zeroline=False, showgrid=False,
        ticks="outside", ticklen=tick_len, tickwidth=tick_width,
        tickfont=dict(size=font_size)
    )
    fig.update_xaxes(**ax_common)
    fig.update_yaxes(**ax_common) 

    # Minor ticks (Plotly supports them per-axis)
    fig.update_xaxes(minor=dict(ticks="outside", ticklen=max(2, tick_len//2),
                                tickwidth=max(1, tick_width-1), showgrid=False))
    fig.update_yaxes(minor=dict(ticks="outside", ticklen=max(2, tick_len//2),
                                tickwidth=max(1, tick_width-1), showgrid=False))
    fig.update_yaxes(tickmode='linear', dtick=1)   

    # Make markers look like MPL (filled with black edge) + thicker lines
    fig.update_traces(
        selector=lambda tr: (tr.mode or "").find("markers") >= 0,
        # marker=dict(line=dict(color="black", width=1.5), size=10)
    )
    fig.update_traces(
        selector=lambda tr: (tr.mode or "").find("lines") >= 0,
        line=dict(width=2.5)
    )

    # Hover label typography to match
    fig.update_layout(hoverlabel=dict(font_size=font_size-6))

    return fig

default_params = {
    # --- Font ---
    "font.family": 'poppins',
    "font.size": 20,          
    "axes.grid": False,     
    # --- Tick marks ---
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": True,      # full box
    "axes.spines.right": True,    
    "xtick.top": False,
    "ytick.right": False,    
    "xtick.minor.visible": True,  # show minor ticks if needed
    "ytick.minor.visible": True,
    # --- Figure size & resolution ---
    "figure.figsize": (940/100, 840/100),  
    "figure.dpi": 100,                     
    "savefig.dpi": 600,
    "legend.frameon": False,              
}
colors = ['tab:blue','orange','green', 'tab:brown', 'purple']*1000
baseline_color = 'grey'
bad_color = 'red'
higlight_color = 'blue'
markers = ['o','^','s','*','D','v']*1000
plt.rcParams.update(default_params)
pio.json.config.default_engine = "json"

if __name__ == "__main__":
    threshold=80
    energy_threshold=100
    # cell_id_list=['214F','227A','227B','227C','228A','228B','228C','229A','229C','230B','230C'],
    # cell_id_info_dict={'214': 'Baseline', '227': "ELE1 80C", '228': 'ELE3 80C', '229': 'ELE1 RT', '230': 'ELE3 RT'},
    # cell_id_list = ['228A','228B','228C','233A','233B','233C','234A','234B','234C',]
    # cell_id_info_dict = {'228': 'ELE3 1.0-4.0V', '233': 'ELE3 1.2-4.0V', '234': 'ELE3 1.3-4.0V',}
    # cell_id_list = ['228A','228B','228C','237A','237B','237C','238A','238B','238C',]
    # cell_id_info_dict = {'228': '0.5mm spacer', '237': '1.0mm spacer', '238': '1.5mm spacer',}
    # cell_id_list=['238A','238B','238C','243A','243B','243C','244A','244B','244C','245A','245B','245C',]
    # cell_id_info_dict={'238': '4.0V', '243': '4.0V CV', '244': '4.2V', '245': '4.2V CV',}
    # cell_id_list = ['238A','238B','238C','247A','247C','251A','251B','251C',]
    # cell_id_info_dict = {'238': 'Wet mill + hand mix', '247': 'Wet mill + US', '251': 'Dry mill + US',}
    # cell_id_list = ['257A','257B','257C','257D','257E', '259A','259B','260A','260C', '242A','242B','242C',]
    # cell_id_info_dict = {'257': '80C', '242':'100C 1.0V LCV (no CV)', '259': '100C 1.5V LCV','260': '100C 2.0V LCV',}
    # cell_id_list = ['257A','257B','257C','257D','257E', '258A','258B','258C']
    # cell_id_info_dict = {'257': '1.0V', '258': '1.3V',}
    # cell_id_list = ['257D','266A','266B','266C','267A','267B','267C','268A','268B','268C','269A','269B','269C',]
    # cell_id_info_dict = {'266': '70%', '267': '75%', '268': '80%', '269': '85%',}
    # cell_id_list = ['285A','285B','285C','286A','286B','286C','287A','287B','287C','288A','288B','288C','289A','289B']
    # cell_id_info_dict = {'285': 'Batch control', '286': '70%+3%', '287': '75%+3%', '288': '80%+3%', '289': '85%+3%',}
    # cell_id_list = ['285A','285B','285C','289A','289B','290A','290B','290C','291A','291B',]
    # cell_id_info_dict = {'285':'5%','289':'3%','290':'2%','291':'1%',}
    cell_id_list = ['292A','292B','292C','293A','293B','293C','294A','294B','294C','295A','295B','295C',]
    cell_id_info_dict = {'292':'ELE1 85%+5%','293':'ELE1 70%+3%','294':'ELE1 75%+3%','295':'ELE1 80%+3%',}
    fig = plot_family_energy_retention(
        cell_id_list=cell_id_list,
        cell_id_info_dict=cell_id_info_dict,
        yaml_path = '/Users/mac/Research/herald/cell_id_energy_list/hypo_se_chem_dict.yaml',
        baseline_cell_ids = ['257D'],
        threshold=threshold,
        energy_threshold=energy_threshold,
        markerstyle='circle',
        )

    # update with ELE3
    cell_id_list = ['285A','285B','285C','286A','286B','286C','287A','287B','287C','288A','288B','288C','289A','289B']
    cell_id_info_dict = {'285': 'ELE3 85%+5%', '286': 'ELE3 70%+3%', '287': 'ELE3 75%+3%', '288': 'ELE3 80%+3%', '289': 'ELE 3 85%+3%',}
    fig = plot_family_energy_retention(
        cell_id_list=cell_id_list,
        cell_id_info_dict=cell_id_info_dict,
        yaml_path = '/Users/mac/Research/herald/cell_id_energy_list/hypo_se_chem_dict.yaml',
        baseline_cell_ids = ['257D'],
        threshold=threshold,
        energy_threshold=energy_threshold,
        fig=fig,
        markerstyle='diamond', alpha = 0.8,
        )
    # fig.update_yaxes(range=[0, 10])
    fig.update_xaxes(range=[500, None])
    # Export (optional)
    fig = apply_mpl_like_style(fig,font_size=25,
                                axis_line_width=2,
                                tick_len=8,
                                tick_width=2,
                                width=800,
                                height=600)
    # fig.update_xaxes(title='Specific Capacity (mAh/g-AM)')
    # fig.update_yaxes(title='> {}% Retention Cycles'.format(threshold))
    fig.show()
    # pio.write_html(fig, f"ged_er{threshold}_pareto.html", include_plotlyjs="inline")