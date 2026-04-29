import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GMMHMM
from utils.utils import fit_hmm_with_states, calculate_bic, calculate_aic


def fit_predefined_states(fret_data, predefined_means):
    means_2D = np.array(predefined_means).reshape(-1, 1)
    
    nstates = means_2D.shape[0]
    kmeans = KMeans(n_clusters=nstates, n_init=1, init=means_2D, random_state=77)
    # Fit with a dummy dataset to set the cluster centers
    kmeans.fit(means_2D)
    predictions = kmeans.predict(fret_data)
    # print(f"kmeans.cluster_centers_: {kmeans.cluster_centers_}")
    return predictions


def fit_by_hand(fret_data, predefined_means):
    predefined_means = np.sort(predefined_means)

    predictions = []
    for i in fret_data:
        dis = [abs(i-j) for j in predefined_means]
        pred = np.argmin(dis)
        predictions.append(pred)
    return np.array(predictions)

def find_elbow_point( fret_data, max_states, verbose=True):
    """Finds the optimal number of states using the elbow method."""
    inertia_list = []
    silhouette_list = []

    for n_states in range(2, max_states):
        kmeans = KMeans(n_clusters=n_states, n_init=10, random_state=77)
        cluster_labels = kmeans.fit_predict(fret_data)
        inertia_list.append(kmeans.inertia_)
        silhouette_list.append(silhouette_score(fret_data, cluster_labels, sample_size=10000, random_state=77))

    inertia_slope = np.diff(np.diff(inertia_list))
    elbow_point = np.argmin(inertia_slope) + 2  
    best_silhouette_n_states = np.argmax(silhouette_list) + 2 
    if verbose:
        print(f"Best number of states (elbow method): {elbow_point}")
        print(f"Best number of states (silhouette score): {best_silhouette_n_states}")
        print(f"Best silhouette score: {max(silhouette_list)}")
    return elbow_point

def fit_gmmhmm(fret_data, max_states, verbose=True):
    score_bic = []
    score_sil = []
    models = []

    for n_states in range(2, max_states):
        hmm = GMMHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42,
                    params="stmcw", init_params="stmcw")
        
        fret_reshaped = fret_data.reshape(-1, 1)
        hmm.fit(fret_reshaped)
        hidden_states = hmm.predict(fret_reshaped)

        if len(np.unique(hidden_states)) > 1:
            sil_score = silhouette_score(fret_reshaped, hidden_states, sample_size=10000, random_state=42)
        else:
            sil_score = 0
        
        bic_hmm = calculate_bic(fret_data, hmm, len(fret_data))
        score_bic.append(bic_hmm)
        score_sil.append(sil_score)
        models.append(hmm)
        if verbose:
            print(f"BIC: {bic_hmm}, Silhouette: {sil_score}, model_converged: {hmm.monitor_.converged}")
    
    return models, score_bic, score_sil
    
def fit_hmm( fret_data, max_states, hmm_init_method, 
            algorithm_choice='viterbi', verbose=True):
    """
    Evaluate different numbers of states using multiple criteria
    """
    results = []
    n_samples = len(fret_data)
    
    for n_states in tqdm(range(2, max_states + 1)): 
        model, score = fit_hmm_with_states(fret_data, n_states, 
                                            init_method=hmm_init_method, 
                                            algorithm_choice=algorithm_choice)
        if model is None:
            continue
            
        # Calculate criteria
        bic = calculate_bic(fret_data, model, n_samples)
        aic = calculate_aic(fret_data, model, n_samples)
        
        predictions = model.predict(fret_data)
        
        results.append({
            'n_states': n_states,
            'model': model,
            'log_likelihood': score,
            'bic': bic,
            'aic': aic,
            'predictions': predictions
        })
        
    bic_scores = [r['bic'] for r in results]
    aic_scores = [r['aic'] for r in results]
    
    best_idx_bic = np.argmin(bic_scores)
    best_idx_aic = np.argmin(aic_scores)
    if verbose: 
        print(f"Best BIC chosed n_states: {results[best_idx_bic]['n_states']}, Best AIC chosed n_states: {results[best_idx_aic]['n_states']}")
    best_result_bic = results[best_idx_bic]
    
    return best_result_bic

def get_weighted_score(score_bic, score_sil):
    # Normalize BIC scores (lower is better, so we'll invert it)
    bic_normalized = (max(score_bic) - np.array(score_bic)) / (max(score_bic) - min(score_bic))
    sil_normalized = (np.array(score_sil) - min(score_sil)) / (max(score_sil) - min(score_sil))

    a = 0.5  # weight for BIC
    b = 0.5  # weight for silhouette
    mix_score = a * bic_normalized + b * sil_normalized
    for i, (bic_norm, sil_norm, mixed) in enumerate(zip(bic_normalized, sil_normalized, mix_score)):
        n_states = i + 2  # since we started from 2 states
        print(f"States: {n_states}, Normalized BIC: {bic_norm:.3f}, Normalized Sil: {sil_norm:.3f}, Mixed Score: {mixed:.3f}")
    return mix_score

def get_weighted_score_2(score_bic, score_sil):
    # Normalize BIC scores (lower is better, so we'll invert it)
    bic_normalized = (max(score_bic) - np.array(score_bic)) / (max(score_bic) - min(score_bic))
    
    # Apply exponential penalty to silhouette scores
    # This will more harshly punish low silhouette scores
    sil_exp = np.exp(2 * np.array(score_sil))  # exponential scaling
    sil_normalized = (sil_exp - min(sil_exp)) / (max(sil_exp) - min(sil_exp))

    a = 0.4  # weight for BIC
    b = 0.6  # increased weight for silhouette
    mix_score = a * bic_normalized + b * sil_normalized
    
    for i, (bic_norm, sil_norm, mixed) in enumerate(zip(bic_normalized, sil_normalized, mix_score)):
        n_states = i + 2
        print(f"States: {n_states}, Normalized BIC: {bic_norm:.3f}, Normalized Sil: {sil_norm:.3f}, Mixed Score: {mixed:.3f}")
    return mix_score