from typing import Sequence, List, Tuple, Dict
from pykalman import KalmanFilter
import numpy as np
from numpy import ma
from scipy.stats import norm
from .poisson_mixture_model import variational_inference
from ..visualizer import gen_trace_profiled_read

states = ['E', 'H', 'D']
ASGN_TO_STATE = {0: 'E', 1: 'H', 2: 'D'}
STATE_COLS = {'E': "red",
              'H': "green",
              'D': "blue",
              'R': "yellow"}


def run_em_algorithm(counts: Sequence[int],
                     show_plot: bool = False) -> List[int]:
    assignments, lambdas = \
        variational_inference(counts,
                              K=3,
                              gamma_hyperparams=[(1, 1),
                                                 (4000, 200),
                                                 (8000, 200)],
                              alpha_hyperparams=[100, 1000, 5000],
                              n_max_iter=10,
                              verbose=False)
    estimated_states = ''.join([ASGN_TO_STATE[x] for x in assignments])
    depth_prior = dict(zip(states, lambdas))
    if show_plot:
        traces = [(None,
                   gen_trace_profiled_read(counts,
                                           estimated_states))]
    prev_states = estimated_states
    while True:
        smoothed_profiles = smooth_profiles(counts,
                                            estimated_states,
                                            depth_prior)
        estimated_states = update_states(counts,
                                         smoothed_profiles)
        if estimated_states == prev_states:
            break
        prev_states = estimated_states
        if show_plot:
            traces.append(([gen_trace_profiled_read(smoothed_profiles[state],
                                                    line_col=STATE_COLS[state])
                            for state in {x for x in estimated_states}],
                           gen_trace_profiled_read(counts,
                                                   estimated_states=estimated_states)))
    return estimated_states


def smooth_profiles(counts: Sequence[int],
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
            smooth_profile(fill_missing_values(data),
                           init_mean=depth_prior[state],
                           var_sampling=depth_prior[state] / read_length,
                           var_error=depth_prior[state])
            for state, data in counts_per_state.items()}


def smooth_profile(counts: Sequence[int],
                   init_mean: float,
                   var_sampling: float,
                   var_error: float) -> List[int]:
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      transition_covariance=[var_sampling],
                      observation_covariance=[var_error],
                      # initial_state_mean=init_mean,
                      n_dim_state=1,
                      n_dim_obs=1)
    smoothed_state_means, smoothed_state_covariances = kf.smooth(counts)
    return ([x[0] for x in smoothed_state_means],
            [x[0][0] for x in smoothed_state_covariances])


def update_states(data: Sequence[int],
                  smoothed_profiles: Dict[str, Tuple[List[int], List[int]]]) -> str:
    def logp_obs(x: float, mean: float, stdev: float) -> str:
        return norm.logpdf(x, loc=mean, scale=stdev)

    # For each data, assign the nearest profile
    # TODO: Use profile-specific observation probability!
    states = list(smoothed_profiles.keys())
    return ''.join([states[np.argmax([logp_obs(x, counts[i], np.sqrt(counts[i] * 0.92 * 0.08))
                                      for state, (counts, _) in smoothed_profiles.items()])]
                    for i, x in enumerate(data)])
