import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import glob, os, re
from herald_visualization.fancy_plot import (
    voltage_vs_capacity_cycling,
    plot_multiple_voltage_vs_cycling,
)
from ruamel.yaml import YAML
from scipy.integrate import simpson
from galvani import BioLogic
import matplotlib as mpl
import matplotlib.font_manager as fm
from herald_visualization import echem as ec
from ruamel.yaml import YAML
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

default_params = {
    # --- Font ---
    # "font.family": poppins,
    "font.size": 32,          
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
plt.rcParams.update(default_params)
colors = ['tab:blue','orange','green', 'tab:brown', 'purple']*1000
baseline_color = 'grey'
bad_color = 'red'
higlight_color = 'blue'
markers = ['o','^','s','*','D','v']*1000

# Your palettes and markers (Plotly names)
PL_COLORS  = ['tab:blue','orange','green','tab:brown','purple'] * 1000
PL_MARKERS = ['circle','triangle-up','square','star','diamond','triangle-down'] * 1000
BASELINE_COLOR = 'grey'
BAD_COLOR      = 'red'
HIGHLIGHT_COLOR= 'blue'

list_of_aviation_cell_q4q5 = [
  # '123H',
  '214E',
  '214F',
  '214I',
  '214L',
  '214M',
  '225A',
  '225B',
  '226A',
  '226B',
  '227A',
  '227B',
  '227C',
  '228A',
  '228B',
  '228C',
  '229A',
  '229B',
  '229C',
  '230A',
  '230B',
  '230C',
  '231A',
  '231B',
  '231C',
  '232A',
  '232B',
  '232C',
  '233A',
  '233B',
  '233C',
  '234A',
  '234B',
  '234C',
  '235A',
  '235B',
  '235C',
  '236A',
  '236B',
  '236C',
  '237A',
  '237B',
  '237C',
  '238A',
  '238B',
  '238C',
  '239A',
  '239B',
  '240A',
  '240B',
  '241A',
  '241B',  
  '242A',
  '242B',
  '242C',
  '243A',
  '243B',
  '243C',
  '244A',
  '244B',
  '244C',
  '245A',
  '245B',
  '245C',
  '246A',
  '246B',
  '246C',
  '247A',
  '247B',
  '247C',
  '248A',
  '248B',
  '248C',
  '249A',
  '249B',
  '249C',
  '250A',
  '250B',
  '250C',
  '251A',
  '251B',
  '251C',
  '252A',
  '252B',
  '252C',
  '253A',
  '253B',
  '253C',
  '254A',
  '254B',
  '254C',
  '255A',
  '255B',
  '255C',
  '256A',
  '256B',
  '256C',
  '257A',
  '257B',
  '257C',
  '257D',
  '257E',
  # '257F',
  '258A',
  '258B',
  '258C',
  '259A',
  '259B',
  '259C',
  '260A',
  '260B',
  '260C',
  '261A',
  '261B',
  '261C',
  '262A',
  '262B',
  '262C',
  '263A',
  '263B',
  '263C',
  '264A',
  '264B',
  '264C',
  '265A',
  '265B',
  '265C',
  '266A',
  '266B',
  '266C',
  '267A',
  '267B',
  '267C',
  '268A',
  '268B',
  '268C',
  '269A',
  '269B',
  '269C',  
  '270A',
  '270B',
  '270C',
  '271A',
  '271B',
  '271C',
  '272A',
  '272B',
  '272C',
  '273A',
  '273B',
  '273C',
  '274A',
  '274B',
  '274C',
  '275A',
  '275B',
  '275C',  
  '276A',
  '276B',
  '276C',
  '276D',
  '276E',
  '276F',
  '276G',
  '276H',
  '276I',
  '276J',
  '276K',
  '276L',
  '277A',
  '277B',
  '277C',
  '278A',
  '278B',
  '278C',
  '279A',
  '279B',
  '279C',
  '280A',
  '280B',
  '280C',   
  '281A',
  '281B',
  '281C',
  '282A',
  '282B',
  '282C',
  '283A',
  '283B',
  '283C',
  '284A',
  '284B',
  '284C',
  '285A',
  '285B',
  '285C',
  '286A',
  '286B',
  '286C',
  '287A',
  '287B',
  '287C',
  '288A',
  '288B',
  '288C',
  '289A',
  '289B',
  '290A',
  '290B',
  '290C',
  '291A',
  '291B',
  '292A',
  '292B',
  '292C',
  '293A',
  '293B',
  '293C',
  '294A',
  '294B',
  '294C',
  '295A',
  '295B',
  '295C',
  '296A',
  '296B',
  '296C',
  '297A',
  '297B',
]

abnormal_cell_ids = ['121B','125A','121A','125B','121H','121D','257F','121I','275B','276I', '271B', '286C']

def find_max_num_cycles_above_threshold(energies, threshold=90):
    # max (y-x)
    # s.t. x>=700, y=threshold * x / 100
    energies = np.array(energies[:])
    available_xs = np.where(energies>=300)[0]
    if len(available_xs) == 0:
        return 0, 0
    best_y_minus_x = 0
    best_x = 0
    best_y = 0
    for x in available_xs:
        y = threshold * energies[x] / 100
        mask = (energies >= y)
        idxs = np.where(mask[x:])[0]
        if len(idxs) == 0:
            continue
        start = x + idxs[0]
        tail = mask[start:]
        inv = np.where(~tail)[0]
        run_len = inv[0] if len(inv) else len(tail)
        y = start + run_len - 1
        y_minus_x = run_len
        if y_minus_x > best_y_minus_x:
            best_y_minus_x = y_minus_x
            best_x = x
            best_y = y
    return best_x, best_y


def make_plotly_figure(threshold=80, energy_threshold=800, cell_id_list=None, fig=None, secondary_y = False):
    # pip install ruamel.yaml plotly
    from ruamel.yaml import YAML
    import numpy as np
    import plotly.graph_objects as go

    # --- load data ---
    yaml = YAML()
    with open('/Users/mac/Research/herald/cell_id_energy_list/hypo_se_chem_dict.yaml','r') as f:
        hypo = yaml.load(f)

    # --- collect points ---
    x_below, y_below, h_below = [], [], []
    x_above, y_above, h_above = [], [], []
    x_q4, y_q4, h_q4 = [], [], []
    x_q3, y_q3, h_q3 = [], [], []
    x_e1, y_e1, h_e1 = [], [], []
    q4ids = set([])  # Q4 cells
    q3ids = set([]) # Q3 cells
    # elyte1ids = set(['292B','292C'])
    elyte1ids = set([])
    x_q3avfront, y_q3avfront, h_q3avfront = [], [], [] # aviation
    x_q3spfront, y_q3spfront, h_q3spfront = [], [], [] # shipping
    x_q4c5front, y_q4c5front, h_q4c5front = [], [], [] # q4c5
    x_q4avfront, y_q4avfront, h_q4avfront = [], [], [] # q4 aviation

    if not cell_id_list:
        cell_id_list = list(hypo.keys())
    for cell_id in cell_id_list:
        if cell_id not in hypo:
            continue
        if cell_id in abnormal_cell_ids:
            continue
        energies = np.array(hypo[cell_id][1:])  
        if energies.size == 0 or not np.isfinite(energies).any():
            continue
        start_idx, end_idx = find_max_num_cycles_above_threshold(energies, threshold=threshold)
        max_energy = float(np.nanmax(energies[start_idx:end_idx+1]))  # recalc max_energy over the selected range
        max_idx = np.nanargmax(energies[start_idx:end_idx+1]) + start_idx
        n80 = end_idx - start_idx + 1 if end_idx >= start_idx else 0
        if max_energy < 480:
            continue
        if max_energy <500 and n80 <30:
            continue
        if n80 < 2:
            continue
        if max_energy < 800 and n80 < 5:
            # generate random number from 0-1, if <0.5, skip
            import random
            if random.random() < 0.0:
                print(f'skipping {cell_id} with max_energy {max_energy} and n80 {n80}')
                continue
        if cell_id == '123H':
            print(cell_id, start_idx, end_idx, max_idx, max_energy, n80)

        hov = f"cell_id: {cell_id}<br>Max GED {max_energy:.1f} Wh/kg-chem at cycle {max_idx+1}<br>{n80} cycles ≥{int(threshold)}% starting from cycle {start_idx+1}"

        # if cell_id in q4ids and threshold==90:
        #     x_q4.append(1000.6); y_q4.append(2); h_q4.append(hov)
        #     continue
        if cell_id in ['123H']:
            x_q3avfront.append(max_energy); y_q3avfront.append(n80); h_q3avfront.append(hov)
            continue
        elif cell_id in ['123E']:
            x_q3spfront.append(max_energy); y_q3spfront.append(n80); h_q3spfront.append(hov)
            continue
        elif cell_id in ['123D','184B','183C','172B','172A']:
            x_q4c5front.append(max_energy); y_q4c5front.append(n80); h_q4c5front.append(hov)
            continue
        elif cell_id in ['257D','257E', '288C']:
            x_q4avfront.append(max_energy); y_q4avfront.append(n80); h_q4avfront.append(hov)
            continue
        elif cell_id in ['179A'] and threshold==90:
            x_q4c5front.append(1000.6); y_q4c5front.append(2); h_q4c5front.append(hov)
        # elif cell_id in q4ids:
        #     x_q4.append(max_energy); y_q4.append(n80); h_q4.append(hov)
        #     continue
        # elif cell_id in elyte1ids:
        #     x_e1.append(max_energy); y_e1.append(n80); h_e1.append(hov)
        #     continue
        # elif cell_id in q3ids:
        #     print(cell_id, max_energy, n80)
        #     x_q3.append(max_energy); y_q3.append(n80); h_q3.append(hov)
        #     continue

        # if (max_energy < energy_threshold):
        #     x_below.append(max_energy); y_below.append(n80); h_below.append(hov)
        # else:
        #     x_above.append(max_energy); y_above.append(n80); h_above.append(hov)
        x_above.append(max_energy); y_above.append(n80); h_above.append(hov)

    # --- Pareto front over ALL points (maximize both axes) ---
    x_all = np.array(x_above + x_q4 + x_q3 + x_e1 + x_q3avfront + x_q3spfront + x_q4c5front + x_q4avfront)
    y_all = np.array(y_above + y_q4 + y_q3 + y_e1 + y_q3avfront + y_q3spfront + y_q4c5front + y_q4avfront)
    h_all = np.array(h_above + h_q4 + h_q3 + h_e1 + h_q3avfront + h_q3spfront + h_q4c5front + h_q4avfront)

    x_front = y_front = h_front = np.array([])
    if x_all.size:
        order = np.argsort(x_all)
        xs, ys, hs = x_all[order], y_all[order], h_all[order]
        keep, best_y = [], -np.inf
        eps = 0.0  # set small >0 to de-noise ties
        for i in range(len(xs)-1, -1, -1):  # sweep from high x
            if ys[i] > best_y + eps:
                keep.append(i); best_y = ys[i]
        keep.sort()
        x_front, y_front, h_front = xs[keep], ys[keep], hs[keep]

    # --- build figure (3 traces) ---
    if fig is None:
        fig = go.Figure()

    # # below threshold: grey circles
    # if x_below:
    #     fig.add_trace(go.Scattergl(
    #         x=x_below, y=y_below, mode="markers", name=f"<{int(energy_threshold)}Wh/kg-chem",
    #         marker=dict(symbol="circle", size=9, color="rgba(120,120,120,0.35)", line=dict(width=0)),
    #         hovertemplate="%{customdata}<extra></extra>", customdata=h_below
    #     ))

    if secondary_y:
        color = 'rgba(139, 69, 19, 1.0)'
        q4color = 'rgba(255, 127, 14, 0.5)'
        q3color = 'rgba(44, 160, 44, 0.5)'  
        e1color = 'rgba(0,128,0,1.0)'
        q3avcolor = 'rgba(255, 127, 14, 1.0)'
        q3spcolor = 'rgba(128,0,128,1.0)'
        q4c5color = 'rgba(0,128,0,1.0)'
        q4avcolor = 'rgba(139, 69, 19, 1.0)'
    else:
        color = 'rgba(211, 211, 211, 1.0)'
        q4color = 'rgba(255, 127, 14, 1.0)'
        q3color = 'rgba(44, 160, 44, 1.0)'
        e1color = 'rgba(0,128,0,1.0)'
        q3avcolor = 'rgba(255, 127, 14, 1.0)'
        q3spcolor = 'rgba(128,0,128,1.0)'
        q4c5color = 'rgba(0,128,0,1.0)'    
        q4avcolor = 'rgba(31, 119, 180, 1.0)'

    # above threshold: blue circles
    if x_above and not secondary_y:
        fig.add_trace(go.Scattergl(
            x=x_above, y=y_above, mode="markers", name=f"Tested Cells",
            marker=dict(symbol="circle", size=10, color=color, line=dict(color="black", width=0.7)),
            hovertemplate="%{customdata}<extra></extra>", customdata=h_above,
        ), secondary_y=secondary_y,)

    # Q4 cells: orange squares
    if x_q4 and not secondary_y:
        fig.add_trace(go.Scattergl(
            x=x_q4, y=y_q4, mode="markers", name="Q4 report",
            marker=dict(symbol="square", size=12, color=q4color, line=dict(color="black", width=1.2)),
            hovertemplate="%{customdata}<extra></extra>", customdata=h_q4, 
        ),secondary_y=secondary_y,)

    # Q3 cells: green squares
    if x_q3 and not secondary_y:
        fig.add_trace(go.Scattergl(
            x=x_q3, y=y_q3, mode="markers", name="Q3 report",
            marker=dict(symbol="square", size=12, color=q3color, line=dict(color="black", width=1.2)),
            hovertemplate="%{customdata}<extra></extra>", customdata=h_q3,
        ), secondary_y=secondary_y,)

    # Electrolyte 1 cells: green circles
    if x_e1 and not secondary_y:
        fig.add_trace(go.Scattergl(
            x=x_e1, y=y_e1, mode="markers", name="Electrolyte 1",
            marker=dict(symbol="circle", size=10, color=e1color, line=dict(color="black", width=1.2)),
            hovertemplate="%{customdata}<extra></extra>", customdata=h_e1,
        ), secondary_y=secondary_y,)

    if x_q3avfront and not secondary_y:
        fig.add_trace(go.Scattergl(
            x=x_q3avfront, y=y_q3avfront, mode="markers", name="Q3 Aviation Front",
            marker=dict(symbol="circle", size=20, color=q3avcolor, line=dict(color="black", width=1.5)),
            hovertemplate="%{customdata}<extra></extra>", customdata=h_q3avfront,
        ), secondary_y=secondary_y,)
    if x_q3spfront and not secondary_y:
        fig.add_trace(go.Scattergl(
            x=x_q3spfront, y=y_q3spfront, mode="markers", name="Q3 Shipping Front",
            marker=dict(symbol="circle", size=20, color=q3spcolor, line=dict(color="black", width=1.5)),
            hovertemplate="%{customdata}<extra></extra>", customdata=h_q3spfront,
        ), secondary_y=secondary_y,)
    if x_q4c5front and not secondary_y:
        fig.add_trace(go.Scattergl(
            x=x_q4c5front, y=y_q4c5front, mode="markers", name="Q4C5 Front",
            marker=dict(symbol="circle", size=20, color=q4c5color, line=dict(color="black", width=1.5)),
            hovertemplate="%{customdata}<extra></extra>", customdata=h_q4c5front,
        ), secondary_y=secondary_y,)
    if x_q4avfront and not secondary_y:
        fig.add_trace(go.Scattergl(
            x=x_q4avfront, y=y_q4avfront, mode="markers", name="Q4 Aviation Front",
            marker=dict(symbol="circle", size=20, color=q4avcolor, line=dict(color="black", width=1.5)),
            hovertemplate="%{customdata}<extra></extra>", customdata=h_q4avfront,
        ), secondary_y=secondary_y,)

    # Pareto front
    if x_front.size and not secondary_y:
        # optional line to connect the front; comment out if you want triangles only
        fig.add_trace(go.Scatter(
            x=x_front, y=y_front, mode="lines", name=f"Pareto frontier {int(threshold)}% retention",
            line=dict(color='rgba(216, 67, 21, 1.0)', width=10), hoverinfo="skip", showlegend=True,
        ), secondary_y=secondary_y,)
        # fig.add_trace(go.Scattergl(
        #     x=x_front[:], y=y_front[:], mode="markers", name="Pareto front",
        #     marker=dict(symbol="circle", color=color, size=20, line=dict(color="black", width=1.2)),
        #     hovertemplate="%{customdata}", customdata=h_front, showlegend=False,
        # ))        
        # fig.add_trace(go.Scattergl(
        #     x=x_front[:2], y=y_front[:2], mode="markers", name="Pareto front",
        #     marker=dict(symbol="circle", color=e1color, size=20, line=dict(color="black", width=1.2)),
        #     hovertemplate="%{customdata}", customdata=h_front, showlegend=False,
        # ))        
        # fig.add_trace(go.Scattergl(
        #     x=x_front[-1:], y=y_front[-1:], mode="markers", name="Pareto front",
        #     marker=dict(symbol="square", color=q4color, size=20, line=dict(color="black", width=1.2)),
        #     hovertemplate="%{customdata}", customdata=h_front, showlegend=False,
        # ))
    elif x_front.size and secondary_y:
        # fig.add_trace(go.Scattergl(
        #     x=x_front, y=y_front, mode="lines", name=f"Pareto frontier {int(threshold)}% retention",
        #     line=dict(color=color, width=2), hoverinfo="skip", showlegend=True,
        # ), secondary_y=secondary_y,)        
        fig.add_trace(go.Scattergl(
            x=x_front, y=y_front, mode="markers", name="Pareto front",
            marker=dict(symbol="circle", size=10, color=color, line=dict(color="black", width=1.2)),
            hovertemplate="%{customdata}", customdata=h_front, showlegend=False,
        ))

    fig.update_layout(
        # title="Max GED vs # Cycles ≥80% Retention",
        xaxis_title="Max GED (Wh/kg-chem)",
        yaxis_title=f"# Cycles ≥{int(threshold)}% Energy Retention",
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    # y axis only shows integer ticks
    # fig.update_yaxes(dtick=1)   
    
    # xlimit from 400 to 1000
    # fig.update_xaxes(range=[600, 1000])
    # # ylimit from 0 to 10
    # fig.update_yaxes(range=[0, 10])
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
        tickfont=dict(size=font_size-4),
        automargin=True,
    )
    fig.update_xaxes(**ax_common)
    fig.update_yaxes(**ax_common)

    # # Minor ticks (Plotly supports them per-axis)
    # fig.update_xaxes(minor=dict(ticks="outside", ticklen=max(2, tick_len//2),
    #                             tickwidth=max(1, tick_width-1), showgrid=False))
    # fig.update_yaxes(minor=dict(ticks="outside", ticklen=max(2, tick_len//2),
    #                             tickwidth=max(1, tick_width-1), showgrid=False))

    # Make markers look like MPL (filled with black edge) + thicker lines
    fig.update_traces(
        selector=lambda tr: (tr.mode or "").find("markers") >= 0,
        # marker=dict(line=dict(color="black", width=1.5), size=10)
    )
    fig.update_traces(
        selector=lambda tr: (tr.mode or "").find("lines") >= 0,
        line=dict(width=8)
    )

    # Hover label typography to match
    fig.update_layout(hoverlabel=dict(font_size=font_size-6))

    return fig


if __name__ == '__main__':
    threshold = 90
    energy_threshold = 800
    save_folder = '/Users/mac/Research/herald/'
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig = make_plotly_figure(threshold=threshold, energy_threshold=energy_threshold, cell_id_list=None, fig=fig, secondary_y=False)
    fig.update_layout(
        xaxis_title="GED (Wh/kg-chem)",
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),        
    )
    fig.update_yaxes(title_text=f"# Cycles ≥{int(threshold)}% Energy Retention", secondary_y=False,)
    fig = apply_mpl_like_style(fig,
                                font_size=22,
                                axis_line_width=2,
                                tick_len=8,
                                tick_width=2,
                                width=720,
                                height=630)
    fig.update_layout(showlegend=True)    
    fig_html = pio.to_html(fig, include_plotlyjs="inline", full_html=False)

    # Build an HTML shell that loads Poppins, then inject the Plotly figure HTML inside.
    head = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body { margin:0; padding:0; font-family: 'Poppins', 'Helvetica Neue', Arial, sans-serif; }
    </style>
    <title>Plotly Figure</title>
    </head>
    <body>
    """

    tail = """
    </body>
    </html>
    """

    out = Path(os.path.join(save_folder, f'pareto_ged_life_er{int(threshold)}.html'))
    out.write_text(head + fig_html + tail, encoding="utf-8")
    print(f"Wrote {out.resolve()}")    

    # save as png
    out_png = Path(os.path.join(save_folder, f'pareto_ged_life_er{int(threshold)}.png'))
    fig.write_image(out_png.resolve().as_posix(), scale=4)