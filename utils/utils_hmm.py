import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from hmmlearn.hmm import GMMHMM
from sklearn.metrics import silhouette_score
from utils.utils import calculate_bic
from utils.utils import preprocess_fret_cross
from utils.utils import mean_scale
from pipeline.step0_preprocess import process_trace_seg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def fit_hmm(fret_np, n_mix =1, nstates=6, algorithm="viterbi", covariance_type="diag",
            use_length=True, length_info=None):
    # Extract parameters from the model
    custom_transmat_prior = np.eye(nstates) * 5 + np.ones((nstates, nstates)) * 0.1
    
    hmm_vit = GMMHMM(n_components=nstates, n_mix=n_mix, covariance_type=covariance_type, n_iter=1000, random_state=42,
             params="stmcw", init_params="stmcw", algorithm=algorithm)

    fret_reshaped = fret_np.reshape(-1, 1)
    if use_length:
        hmm_vit.fit(fret_reshaped, lengths=length_info)
    else:
        hmm_vit.fit(fret_reshaped)


    print(f"model_converged: {hmm_vit.monitor_.converged}")
    print(f"hmm means, {hmm_vit.means_}")
    return hmm_vit



def plot_gaussian_fret(fret_np, predictions):
    state_np = predictions
    unique_states = np.unique(state_np)
    total_count = len(state_np)
    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(-0.2, 1.2, len(unique_states)))

    for color, state in zip(colors, unique_states):
        state_data = fret_np[state_np == state]
        # mask = (state_data > 0.00001) & (state_data < 1)
        # state_data = state_data[mask]
        
        # Calculate histogram using numpy
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
    plt.show()

def plot_single_trace_with_signal(single_prediction):
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
    plt.show()