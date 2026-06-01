import os
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np

from utils.utils import _sort_states, _merge_close_states, _make_states_contiguous, _calculate_transition_probs, _merge_close_states_by_estimated_mean
from utils.utils import preprocess_fret_cross, preprocess_fret_simple, get_dwell_time


class StateAnalyzer:
    def __init__(self, root_dir, data_dir,
                 save_dir="result/group_kmeans",
                 verbose=True,
                 assign_state="kmeans",
                 hmm_init_method="curve_fit",
                 max_states=6,
                 handle_short_dwells=False,
                 process_fret="cross"):

        self.root_dir = root_dir
        self.data_path = os.path.join(root_dir, data_dir)
        if not os.path.isabs(save_dir):
            self.save_dir = os.path.join(root_dir, save_dir, assign_state)
        else:
            self.save_dir = os.path.join(save_dir, assign_state)

        self.verbose = verbose
        self.max_states = max_states
        self.assign_state = assign_state
        self.hmm_init_method = hmm_init_method
        self.handle_short_dwells = handle_short_dwells
        self.process_fret = process_fret
        os.makedirs(self.save_dir, exist_ok=True)

    def merge_states(self, predictions, fret_data,
                     threshold=0.09, reassign="none"):
        """Merges states based on their cluster mean difference."""
        from utils.utils_states_assign import fit_predefined_states, fit_by_hand

        predictions, sorted_means = _sort_states(predictions, fret_data)
        merge_map, _ = _merge_close_states(predictions, sorted_means, fret_data, threshold)
        new_predictions = np.array([merge_map[p] for p in predictions])
        new_means = [np.mean(fret_data[new_predictions == label]) for label in np.unique(new_predictions)]

        if reassign == "kmeans":
            new_predictions_refit = fit_predefined_states(fret_data, new_means)
        elif reassign == "byhand":
            new_predictions_refit = fit_by_hand(fret_data, new_means)
        else:  # "none"
            new_predictions_refit = _make_states_contiguous(new_predictions)

        new_predictions_final, means_final = _sort_states(new_predictions_refit, fret_data)
        trans_probs = _calculate_transition_probs(new_predictions)

        return new_predictions_final, means_final, trans_probs

    def _handle_short_dwells(self, pred, fret, state_means, min_last_frame=3):
        """Substitutes states in short residence (1 or 2 frames)."""
        _, all_dwell_initial = get_dwell_time(pred, len(state_means))
        pred_new = pred.copy()
        all_dwell_times = all_dwell_initial.copy()
        n = len(pred)

        for idx in range(len(all_dwell_initial)):
            start = int(np.sum(all_dwell_initial[:idx]))
            end = start + all_dwell_initial[idx]

            if all_dwell_times[idx] >= min_last_frame or start >= n:
                continue

            prev = pred_new[start - 1] if start > 0 else None
            next = pred_new[end] if end < n else None
            curr_mean = np.mean(fret[start:end])

            if prev == next and prev is not None:
                pred_new[start:end] = prev
                if idx < n - 1:
                    all_dwell_times[idx + 1] += all_dwell_initial[idx]
            else:
                diff_prev = abs(curr_mean - state_means[prev]) if prev is not None else 100
                diff_next = abs(curr_mean - state_means[next]) if next is not None else 100

                # Match ver1: ties prefer the next state. Only guard the degenerate
                # single-dwell case where both neighbors are absent.
                if diff_prev < diff_next and prev is not None:
                    pred_new[start:end] = prev
                elif next is not None:
                    pred_new[start:end] = next
                    if idx < n - 1:
                        all_dwell_times[idx + 1] += 1

        return pred_new

    def print_process(self, means=None, stage_desc=None):
        print("\n" + "=" * 50)
        print(stage_desc)
        if means is not None:
            print(f"Final number of states: {len(means)}")
            print("State means:")
            for i, mean in enumerate(means):
                print(f"  State {i}: {mean:.3f}")
        print("=" * 50 + "\n")


class StateAnalyzer_Test(StateAnalyzer):
    """Thin subclass used by step3 — only merge_states and _handle_short_dwells are needed."""
    pass