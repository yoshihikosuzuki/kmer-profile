from typing import Sequence, List, Tuple, Dict
from pykalman import KalmanFilter
import numpy as np
from numpy import ma
from scipy.stats import norm


def smooth_count_profile(counts: Sequence[int],
                         init_mean: float,
                         var_sampling: float,
                         var_error: float) -> List[int]:
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      transition_covariance=[var_sampling],
                      observation_covariance=[var_error],
                      initial_state_mean=init_mean,
                      n_dim_state=1,
                      n_dim_obs=1)
    smoothed_state_means, smoothed_state_covariances = kf.smooth(counts)
    return ([x[0] for x in smoothed_state_means],
            [x[0][0] for x in smoothed_state_covariances])


def smooth_count_profiles(counts: Sequence[int],
                          estimated_states: str,
                          depth_prior: Dict[str, int] = {'E': 1, 'H': 20, 'D': 40},
                          read_length: int = 24000) -> Dict[str, Tuple[List[int], List[int]]]:
    def fill_missing_values(data: List[int]) -> ma.array:
        """Replace zeros to missing values."""
        X = ma.array(data)
        for i in range(len(X)):
            if X[i] == 0:
                X[i] = ma.masked
        return X

    # Extract each count profile while labeling missing values
    states = set(estimated_states)
    counts_per_state = {state: [0] * len(counts) for state in states}
    for i, (count, state) in enumerate(zip(counts, estimated_states)):
        counts_per_state[state][i] = count
    # Smooth the counts with a Kalman smoother after filling annotating values
    # NOTE: Observation variance is much larger than the model for tolerance
    #       against outliers by misclassification
    return {state:
            smooth_count_profile(fill_missing_values(data),
                                 init_mean=depth_prior[state],
                                 var_sampling=depth_prior[state] / read_length,
                                 var_error=depth_prior[state])
            for state, data in counts_per_state.items()}


def update_states(data: Sequence[int],
                  smoothed_count_profiles: Dict[str, Tuple[List[int], List[int]]]) -> str:
    def logp_obs(x: float, mean: float, stdev: float) -> str:
        return norm.logpdf(x, loc=mean, scale=stdev)

    # For each data, assign the nearest profile
    # TODO: Use profile-specific observation probability!
    states = list(smoothed_count_profiles.keys())
    return [states[np.argmax([logp_obs(x, counts[i], np.sqrt(counts[i] * 0.92 * 0.08))
                              for state, (counts, variances) in smoothed_count_profiles.items()])]
            for i, x in enumerate(data)]
