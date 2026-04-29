import pickle
import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

STATE_MERGE_THRESHOLD = 0.09   # merge two FRET states whose means differ by less than this


def preprocess_fret_simple(segment):
    fret = segment[:, 1] / (segment.sum(axis=1))
    mask = fret <= 1.5
    fret_data = fret[mask].reshape(-1, 1)
    segment = segment[mask]
    fret_data[fret_data > 1.0] = 1.0
    fret_data[fret_data < 0.0] = 0.0
    return fret_data, segment

def preprocess_fret_cross(segment, cross=0.05):
    channel1 = segment[:, 0]
    channel2 = segment[:, 1]
    
    fret = (channel2- cross*channel1) /(channel2-cross*channel1 + channel1)
    mask = (fret > -0.21) & (fret <= 1.21)
    fret_data = fret[mask].reshape(-1, 1)
    segment = segment[mask]
    return fret_data, segment


def gauss_1D(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        mu = params[i + 1]
        sigma = params[i + 2]
        y += amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

def load_pickle_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def mean_scale(raw):
    """Normalize a trace by its top-5 total intensity average (donor + acceptor)."""
    column_sum = raw[:, 0] + raw[:, 1]
    average = np.mean(np.sort(column_sum)[-5:])
    return raw / average

def get_dwell_time(pred, num_states):
    state_dwell_times = [[] for _ in range(num_states)]
    all_dwell_times = []
    current_state = pred[0]
    current_length = 1
    
    for i in range(1, len(pred)):
        if pred[i] == current_state:
            current_length += 1
        else:
            state_dwell_times[current_state].append(current_length)
            all_dwell_times.append(current_length)
            
            current_state = pred[i]
            current_length = 1
    
    state_dwell_times[current_state].append(current_length)
    all_dwell_times.append(current_length)
    
    return state_dwell_times, all_dwell_times

def get_transition_counts(predictions):
    unique_states = np.unique(predictions)
    state_map = {state: i for i, state in enumerate(unique_states)}
    n_states = len(unique_states)
    
    trans_counts = np.zeros((n_states, n_states), dtype=int)
    
    for t in range(len(predictions)-1):
        from_state = state_map[predictions[t]]
        to_state = state_map[predictions[t+1]]
        trans_counts[from_state, to_state] += 1
        
    return trans_counts

def _sort_states(predictions, fret_data):
    means = _calculate_state_means(predictions, fret_data)
    sort_idx = np.argsort(means)
    order_map = {old: new for new, old in enumerate(sort_idx)}
    predictions = np.array([order_map[p] for p in predictions])
    sorted_means = means[sort_idx]
    return predictions, sorted_means

def _calculate_state_means(predictions, fret_data):
    n_states = np.unique(predictions)
    return np.array([np.mean(fret_data[predictions == state]) for state in n_states])


def _calculate_transition_probs(predictions):
    trans_counts = get_transition_counts(predictions)
    return trans_counts / (np.sum(trans_counts, axis=1, keepdims=True) + 1e-6)


def _merge_close_states(predictions, cluster_means, fret_data, threshold=STATE_MERGE_THRESHOLD):
    n_states = len(cluster_means)
    merge_map = list(range(n_states))
    
    updated_means = cluster_means.copy()
    
    for i in range(n_states-1):
        j = i+1
        if abs(cluster_means[i] - cluster_means[j]) < threshold or (cluster_means[i] > 1.0 and cluster_means[j] > 1.0):
            merge_map[j] = merge_map[i]
            merge_mask = (predictions == j) | (predictions == i)
            merge_mean = np.mean(fret_data[merge_mask])
            updated_means[i] = merge_mean
            updated_means[j] = merge_mean
    return merge_map, updated_means

def _merge_close_states_by_estimated_mean(predictions, cluster_means, estimated_means, fret_data, threshold=STATE_MERGE_THRESHOLD):
    n_states = len(cluster_means)
    merge_map = list(range(n_states))
    
    updated_means = estimated_means.copy()
    
    for i in range(n_states-1):
        j = i+1
        if abs(updated_means[i] - updated_means[j]) < threshold or (updated_means[i] > 1.0 and cluster_means[j] > 1.0):
            merge_mask = (predictions == j) | (predictions == i)
            merge_mean = np.mean(fret_data[merge_mask])
            updated_means[i] = merge_mean
            updated_means[j] = merge_mean
            
            dist = [abs(merge_mean - cluster_means[k]) for k in range(n_states)]
            update_token = np.argmin(dist)
            merge_map[j] = update_token
            merge_map[i] = update_token
    return merge_map, updated_means

# kolf 
def _make_states_contiguous(predictions):
    unique_labels = np.unique(predictions)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    return np.array([label_mapping[label] for label in predictions])


def fit_hmm_with_states(fret_data, n_states, n_attempts=1, 
                        algorithm_choice='viterbi', init_method='kmeans'):
    """
    Fit HMM with multiple attempts and return the best model based on score
    """
    best_score = float('-inf')
    best_model = None
    
    # Create histogram for initial guesses
    bins = np.linspace(-0.2, 1.2, num=50)
    hist_data, bins_h1D = np.histogram(fret_data, bins=bins)
    x_fit = np.linspace(bins_h1D[0]+(bins_h1D[1]-bins_h1D[0])/2,
                        bins_h1D[-1]-(bins_h1D[1]-bins_h1D[0])/2,
                        num=len(bins)-1)
    
    # Initial guesses based on evenly spaced means
    means_init = np.linspace(0.01, 0.99, n_states)
    guess = []
    for mean in means_init:
        guess.extend([hist_data.max(), mean, 0.1])
    
    try:
        # Fit Gaussian mixture
        if init_method == 'curve_fit':
            popt_init, _ = curve_fit(gauss_1D, x_fit, hist_data, p0=guess)
            means = [[popt_init[i+1]] for i in range(0, len(popt_init), 3)]
            covars = [[[popt_init[i+2]**2]] for i in range(0, len(popt_init), 3)]
            
        elif init_method == 'kmeans':
            kmeans = KMeans(n_clusters=n_states, n_init=10)
            cluster_labels = kmeans.fit_predict(fret_data)

            means = []
            covars = []
            for i in range(n_states):
                cluster_data = fret_data[cluster_labels == i]
                if len(cluster_data) > 0:  
                    means.append([np.mean(cluster_data)])
                    covars.append([[np.var(cluster_data)]])
                else:  
                    means.append([np.mean(fret_data)])
                    covars.append([[np.var(fret_data)]])

                    
        elif init_method == 'random':
            means = np.random.uniform(0, 1, (n_states, 1)) 
            covars = np.array([np.random.uniform(0.01, 0.05, (1, 1)) for _ in range(n_states)])

        for _ in range(n_attempts):
            model = hmm.GaussianHMM(n_components=n_states,
                                  covariance_type="full",
                                  init_params="t",
                                  n_iter=500,
                                  algorithm=algorithm_choice)
            
            # Initialize means and covars from Gaussian fitting

            model.means_ = np.array(means)
            model.covars_ = np.array(covars)
  
            # Fit model
            model.fit(fret_data)
            score = model.score(fret_data)
            
            if score > best_score:
                best_score = score
                best_model = model
                
    except Exception:
        return None, float('-inf')
    
    return best_model, best_score

def calculate_bic(fret_data, model, n_samples):
    """Calculate Bayesian Information Criterion"""
    n_parameters = (model.n_components * model.n_components + 
                   2 * model.n_components - 1)
    return -2 * model.score(fret_data) + n_parameters * np.log(n_samples)

def calculate_aic(fret_data, model, n_samples):
    """Calculate Akaike Information Criterion"""
    n_parameters = (model.n_components * model.n_components + 
                   2 * model.n_components - 1)
    return -2 * model.score(fret_data) + 2 * n_parameters