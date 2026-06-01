import os
import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from scipy.stats import norm as sp_norm
import warnings
from collections import defaultdict
from matplotlib import colors as mcolors
from tqdm import tqdm

def pattern_to_df(pattern_dict):
    df = pd.DataFrame(
        data={'count': list(pattern_dict.values()),
              'pattern': list(pattern_dict.keys())},
    )
    return df

def count_unique_states(transition_tuple):
    unique_states = set()
    if isinstance(transition_tuple[0], tuple):
        for transition in transition_tuple:
            unique_states.update(transition)
    else:
        # Handle single-element tuple case
        unique_states.update(transition_tuple)
    return len(unique_states)

def count_num_transition(transition_tuple):
    """count the number of transitions in the transition tuple

    differentiate (2,) --> 0 and ((2,1),) --> 1
    """
    if isinstance(transition_tuple[0], tuple):
        return len(transition_tuple)
    else:
        return 0

def get_abbreviation(pred):
    output = [pred[0]]
    for i in range(1, len(pred)):
        if pred[i] == output[-1]:
            continue
        else:
            output.append(pred[i])
    return output


########################################################
# 1. transition plot
def plot_transition(df_pattern_record, save_path, save_name):
    unique_transitions = set()
    for pattern in df_pattern_record['pattern']:
        if isinstance(pattern, tuple) and isinstance(pattern[0], tuple):
            for transition in pattern:
                unique_transitions.add(transition)
                    
    transitions = list(unique_transitions)
    num_states = df_pattern_record['num_state'].max()+1

    G = nx.DiGraph()
    G.add_edges_from(transitions)

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(G)
            
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='<|-|>', arrowsize=15, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')

    plt.title('Transition Plot',fontsize=16)
    plt.axis('off') 
    plt.gca().set_facecolor('lightgray')
    plt.savefig(save_path + save_name)
    plt.close()

########################################################
# 2. pattern count plot
def plot_hist_pattern(df_pattern_record, save_path, save_name):
    groups = df_pattern_record.groupby('num_transition')
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))

    abbreviation_map = {}
    for idx, (pattern, num_transition) in enumerate(zip(df_pattern_record['pattern'], df_pattern_record['num_transition'])):
        abbreviation_map[pattern] = f'transition{num_transition}_{idx}'

    for (num_transition, group), color in zip(groups, colors):
        sorted_group = group.sort_values(by='count', ascending=False)
        patterns = sorted_group['pattern'].apply(lambda x: abbreviation_map.get(x, 'Unknown'))
        plt.bar(patterns, sorted_group['count'], color=color, label=f'Num Transition: {num_transition}')

    plt.xlabel('Pattern')
    plt.ylabel('Count')
    plt.title('Pattern Count Visualization')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.yscale('log')  # Log scale for y-axis
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.legend(title='Num Transition')  # Add legend

    # Display full pattern names below the figure
    plt.figtext(0.5, -0.1, ' '.join([f'{abbr}: {pattern}' for abbr, pattern in abbreviation_map.items()]), 
                wrap=True, horizontalalignment='center', fontsize=10)

    plt.savefig(save_path + save_name, bbox_inches='tight')
    plt.close()

########################################################
# 3. gaussian fret plot

def plot_gaussian_fret(predictions, save_path, save_name):
    state_assign = [single_prediction['predictions'] for single_prediction in predictions]
    fret = [single_prediction['fret_data'] for single_prediction in predictions]
    state_np = np.concatenate(state_assign)
    fret_np = np.concatenate(fret)

    unique_states = np.unique(state_np)
    total_count = len(state_np)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_states)))

    for color, state in zip(colors, unique_states):
        state_data = fret_np[state_np == state]
        bins = np.linspace(-0.2, 1.2, 100)
        counts, bins = np.histogram(state_data, bins=bins, density=False)
        bin_widths = np.diff(bins)
        refined_counts_density = counts / (total_count)
        # refined_counts_density = counts / total_count
        # print(refined_counts_density.shape, np.sum(refined_counts_density))
        # Plot histogram
        light_color = mcolors.to_rgba(color, alpha=0.15)
        plt.bar(bins[:-1], refined_counts_density, width=bin_widths, edgecolor=light_color, color=light_color, alpha=0.6)
                
        # Plot Gaussian fit
        mu, std = np.mean(state_data), np.std(state_data)
        x = np.linspace(min(state_data), max(state_data), 100)
        p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        
        # Normalize the Gaussian fit to match the histogram's density
        p *= np.sum(refined_counts_density) * bin_widths[0]
        plt.plot(x, p, color=color, linewidth=2, label=f'State {state} (μ={mu:.2f})')
        
    all_counts, all_bins = np.histogram(fret_np, bins=100, density=True)
    all_counts = all_counts * np.diff(all_bins)
    plt.bar(all_bins[:-1], all_counts, width=np.diff(all_bins), edgecolor="lightgrey", color="lightgrey", alpha=0.6)

    plt.title('Gaussian Fits for All States')
    plt.xlabel('FRET Data')
    plt.ylabel('Density Per Bin')
    plt.legend()
    output_file_gaussian = save_path + save_name
    plt.savefig(output_file_gaussian, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure

# 4. single trace plot
def plot_single_trace(single_prediction, save_dir, save_name, pattern):
    """Plots a single FRET trace with state assignments."""
    full_mean = single_prediction["full_mean"]
    fret = single_prediction['fret_data']

    plt.figure(figsize=(15, 5))
    time = np.arange(len(fret))
    plt.plot(time, fret, 'k-', alpha=0.3, label='FRET Data')

    n_states = len(full_mean)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_states))
    
    for state in range(n_states):
        mask = single_prediction['predictions'] == state
        if len(fret[mask]) > 0:
            mean_value = np.mean(fret[mask])
            plt.plot(time[mask], fret[mask], '.', 
                        color=colors[state], label=f'State {state} (^μ={mean_value:.2f})')    
                        
        else:
            mean_value = full_mean[state]
            plt.plot(time[mask], fret[mask], '.', 
                        color=colors[state], label=f'State {state} (μ={mean_value:.2f})')
    
    plt.title(f'FRET Trace with Top Pattern {pattern}')
    plt.xlabel('Time (frames)')
    plt.ylabel('FRET')
    plt.ylim(0, 1.1) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir + save_name)
    plt.close()

def plot_single_trace_with_signal(single_prediction, save_dir, save_name):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[1, 1])

    # Upper subplot for raw data features
    time = np.arange(len(single_prediction['raw_data']))
    raw_data = single_prediction['raw_data']
    full_mean = single_prediction['full_mean']
    ax1.plot(time, raw_data[:, 0], 'b-', label='Feature 1', alpha=0.7)
    ax1.plot(time, raw_data[:, 1], 'r-', label='Feature 2', alpha=0.7)
    ax1.set_title('Raw Data Features')
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Intensity')
    ax2.set_ylim(0, 1.1)
    ax1.legend()

    # Lower subplot for FRET trace (existing code)
    fret = single_prediction['fret_data']
    ax2.plot(time, fret, 'k-', alpha=0.3, label='FRET Data')

    n_states = len(full_mean)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_states))

    for state in range(n_states):
        mask = single_prediction['predictions'] == state
        if len(fret[mask]) > 0:
            mean_value = np.mean(fret[mask])
            ax2.plot(time[mask], fret[mask], '.', 
                    color=colors[state], label=f'State {state} (^μ={mean_value:.2f})')    
        else:
            mean_value = full_mean[state]
            ax2.plot(time[mask], fret[mask], '.', 
                    color=colors[state], label=f'State {state} (μ={mean_value:.2f})')

    ax2.set_title(f'FRET Trace with State Assignments for trace from {single_prediction["trace_label"]}')
    ax2.set_xlabel('Time (frames)')
    ax2.set_ylabel('FRET')
    ax2.set_ylim(0, 1.1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_dir + save_name)
    plt.close()


# ---------------------------------------------------------------------------
#  5. HMM summary figure  (histogram + occupancy + representative traces)
# ---------------------------------------------------------------------------

# Colour palette shared across all panels
_STATE_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def plot_hmm_summary(update_predictions_all, merged_means, data_set_name,
                     result_dir, max_traces=6):
    """
    Save a publication-quality summary figure:
      Panel A – FRET efficiency histogram with per-state Gaussian fits
      Panel B – State occupancy bar chart
      Panel C – Representative single-molecule traces (most transitions first)

    Parameters
    ----------
    update_predictions_all : list of dict
        Each dict must contain 'fret_data' (1-D array) and 'predictions' (1-D int array).
    merged_means : array-like
        Global state means from the best HMM result.
    data_set_name : str
        Used in the figure title and output filename.
    result_dir : Path
        Directory where the PNG is saved.
    max_traces : int
        Maximum number of trace panels in Panel C.
    """
    all_fret = np.concatenate([d["fret_data"] for d in update_predictions_all])
    all_pred = np.concatenate([d["predictions"] for d in update_predictions_all])
    states   = np.unique(all_pred)
    n_states = len(states)
    colors   = [_STATE_COLORS[i % len(_STATE_COLORS)] for i in range(n_states)]

    # ── Layout ───────────────────────────────────────────────────────────────
    n_trace_rows = min(max_traces, len(update_predictions_all))
    top_h   = 0.30          # fractional height of the top two panels
    gap     = 0.04
    fig_h   = 5 + 2.2 * n_trace_rows

    fig = plt.figure(figsize=(16, fig_h), facecolor="white")
    gs_top = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.07, right=0.97,
        top=0.93, bottom=0.93 - top_h,
        wspace=0.35,
    )
    gs_trace = gridspec.GridSpec(
        n_trace_rows, 1, figure=fig,
        left=0.07, right=0.97,
        top=0.93 - top_h - gap,
        bottom=0.04,
        hspace=0.15,
    )

    ax_hist   = fig.add_subplot(gs_top[0, 0])
    ax_bar    = fig.add_subplot(gs_top[0, 1])
    ax_traces = [fig.add_subplot(gs_trace[i, 0]) for i in range(n_trace_rows)]

    # ── Panel A: FRET Histogram ───────────────────────────────────────────────
    bins = np.linspace(-0.1, 1.1, 80)
    ax_hist.hist(all_fret, bins=bins, density=True,
                 color="#BBCFE8", edgecolor="white", linewidth=0.4,
                 zorder=1, label="All data")

    x_fit = np.linspace(-0.1, 1.1, 400)
    for i, s in enumerate(states):
        sd = all_fret[all_pred == s]
        if len(sd) < 2:
            continue
        mu     = np.mean(sd)
        sigma  = max(np.std(sd), 1e-4)
        weight = len(sd) / len(all_fret)
        ax_hist.plot(x_fit, weight * sp_norm.pdf(x_fit, mu, sigma),
                     color=colors[i], linewidth=2.0,
                     label=f"State {i}  μ={mu:.2f}", zorder=3)
        ax_hist.axvline(mu, color=colors[i], linewidth=1.0,
                        linestyle="--", alpha=0.55, zorder=2)

    ax_hist.set_xlabel("FRET Efficiency", fontsize=12)
    ax_hist.set_ylabel("Probability Density", fontsize=12)
    ax_hist.set_title("FRET Efficiency Distribution", fontsize=13, fontweight="bold")
    ax_hist.legend(fontsize=9, framealpha=0.85, loc="upper right")
    ax_hist.set_xlim(-0.1, 1.1)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.tick_params(labelsize=10)

    # ── Panel B: State Occupancy ──────────────────────────────────────────────
    occupancy = np.array([np.sum(all_pred == s) / len(all_pred) for s in states])
    state_labels = [
        f"S{s}\nμ={merged_means[s]:.2f}" if s < len(merged_means) else f"S{s}"
        for s in states
    ]
    bars = ax_bar.bar(state_labels, occupancy * 100,
                      color=colors, edgecolor="white", linewidth=0.8,
                      width=0.55, zorder=2)
    for bar, val in zip(bars, occupancy * 100):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax_bar.set_xlabel("State", fontsize=12)
    ax_bar.set_ylabel("Occupancy (%)", fontsize=12)
    ax_bar.set_title("State Occupancy", fontsize=13, fontweight="bold")
    ax_bar.set_ylim(0, max(occupancy * 100) * 1.25)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(labelsize=10)
    ax_bar.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=5))
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

    # ── Panel C: Representative Traces ───────────────────────────────────────
    trans_counts = [np.sum(np.diff(d["predictions"]) != 0)
                    for d in update_predictions_all]
    selected = np.argsort(trans_counts)[::-1][:n_trace_rows]

    for row, trace_idx in enumerate(selected):
        ax   = ax_traces[row]
        d    = update_predictions_all[trace_idx]
        fret = d["fret_data"]
        pred = d["predictions"]

        # Raw FRET (light grey scatter)
        ax.scatter(np.arange(len(fret)), fret, s=3,
                   color="#CCCCCC", zorder=1, rasterized=True)

        # Step-function coloured by state
        for s, c in zip(states, colors):
            idx_s = np.where(pred == s)[0]
            if len(idx_s) == 0:
                continue
            breaks   = np.where(np.diff(idx_s) > 1)[0] + 1
            for seg in np.split(idx_s, breaks):
                ax.hlines(np.mean(fret[seg]), seg[0], seg[-1],
                          colors=c, linewidth=2.5, zorder=3)

        # Dashed global-mean reference lines
        for s, c in zip(states, colors):
            if s < len(merged_means):
                ax.axhline(merged_means[s], color=c, linewidth=0.6,
                           linestyle=":", alpha=0.40, zorder=2)

        ax.set_ylim(-0.15, 1.15)
        ax.set_xlim(0, max(len(fret) - 1, 1))
        ax.set_ylabel("FRET", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"Trace #{trace_idx}  |  {len(fret)} frames  |  "
            f"{trans_counts[trace_idx]} transition(s)",
            fontsize=9, loc="left", pad=2,
        )
        if row < n_trace_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Frame", fontsize=10)

    # Colour legend on the first trace panel
    if ax_traces:
        handles = [
            Line2D([0], [0], color=colors[i], linewidth=3,
                   label=(f"State {s}  (μ={merged_means[s]:.2f})"
                          if s < len(merged_means) else f"State {s}"))
            for i, s in enumerate(states)
        ]
        ax_traces[0].legend(handles=handles, fontsize=8,
                            loc="upper right", framealpha=0.85,
                            ncol=min(n_states, 4))

    # ── Super-title & save ────────────────────────────────────────────────────
    fig.suptitle(
        f"HMM State Analysis  —  {data_set_name}\n"
        f"{n_states} states  |  {len(update_predictions_all)} segments  |  "
        f"{len(all_fret)} frames total",
        fontsize=13, fontweight="bold", y=0.985,
    )

    out_path = result_dir / f"step3_HMM_summary_{data_set_name}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary figure saved → {out_path}")
