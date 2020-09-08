from typing import Sequence, Tuple, List
import numpy as np
from scipy.special import digamma


def gibbs_sampling(data: Sequence[int],
                   K: int,
                   gamma_hyperparams: Sequence[Tuple[float, float]],
                   alpha_hyperparams: Sequence[int],
                   n_max_iter: int,
                   verbose: bool = False) -> List[int]:
    """Perform Gibbs sampling of a Poisson mixture model with K components.

    positional arguments:
      @ data              : To be classified into K components.
      @ K                 : Number of Poisson components.
      @ gamma_hyperparams : Hyperparameters alpha and beta (i.e. shape and rate)
                            of Gamma distribution for each component.
      @ alpha_hyperparams : Hyperparameter alpha of Dirichlet distribution
                            for each component.
      @ n_max_iter        : Maximum number of iterations.
    """
    def update_assignment(n: int):
        nonlocal assignments, lambdas, partitions
        etas = [None] * K
        for k in range(K):
            etas[k] = np.exp(data[n] * np.log(lambdas[k])
                             - lambdas[k]
                             + np.log(partitions[k]))
        tot = sum(etas)
        etas = [eta / tot for eta in etas]
        assignments[n] = np.random.choice(K, p=etas)

    def update_lambda(k: int):
        nonlocal N, lambdas, assignments
        a_prior, b_prior = gamma_hyperparams[k]
        a = sum([data[n] for n in range(N) if assignments[n] == k]) + a_prior
        b = len(list(filter(lambda x: x == k, assignments))) + b_prior
        lambdas[k] = np.random.gamma(a, 1 / b)

    def update_partition():
        nonlocal N, partitions
        alphas = [None] * K
        for k in range(K):
            alphas[k] = (len(list(filter(lambda x: x == k, assignments)))
                         + alpha_hyperparams[k])
        partitions = np.random.dirichlet(alphas).transpose()

    assert len(gamma_hyperparams) == len(alpha_hyperparams) == K, \
        "Invalid number of hyperparameters"
    N = len(data)
    # Start from parameters of components computed from prior knowledge
    lambdas = [a / b for a, b in gamma_hyperparams]
    partitions = [alpha / sum(alpha_hyperparams)
                  for alpha in alpha_hyperparams]
    assignments = [None] * N
    for t in range(n_max_iter):
        for n in range(N):
            update_assignment(n)
        for k in range(K):
            update_lambda(k)
        if verbose:
            print(f"lamdas={lambdas}")
        update_partition()
        if verbose:
            print(f"partitions={partitions}")
        # TODO: check if the permutation of clusters increases the likelihood
        #       (when component-specific hyperparameters are specified)
    return np.array(assignments, dtype=np.int64)


def variational_inference(data: Sequence[int],
                          K: int,
                          gamma_hyperparams: Sequence[Tuple[float, float]],
                          alpha_hyperparams: Sequence[int],
                          n_max_iter: int,
                          verbose: bool = False) -> List[int]:
    """Perform Gibbs sampling of a Poisson mixture model with K components.

    positional arguments:
      @ data              : To be classified into K components.
      @ K                 : Number of Poisson components.
      @ gamma_hyperparams : Hyperparameters alpha and beta (i.e. shape and rate)
                            of Gamma distribution for each component.
      @ alpha_hyperparams : Hyperparameter alpha of Dirichlet distribution
                            for each component.
      @ n_max_iter        : Maximum number of iterations.
    """
    def update_q_assignment(n: int):
        nonlocal assignment_etas, lambda_as, lambda_bs, partition_alphas
        for k in range(K):
            assignment_etas[n][k] = \
                np.exp(data[n] * (digamma(lambda_as[k]) - np.log(lambda_bs[k]))
                       - lambda_as[k] / lambda_bs[k]
                       + digamma(partition_alphas[k]) - digamma(sum(partition_alphas)))
        tot = sum(assignment_etas[n])
        for k in range(K):
            assignment_etas[n][k] /= tot

    def update_q_lambda(k: int):
        nonlocal assignment_etas, lambda_as, lambda_bs
        a_prior, b_prior = gamma_hyperparams[k]
        lambda_as[k] = (sum([assignment_etas[n][k] * data[n] for n in range(N)])
                        + a_prior)
        lambda_bs[k] = (sum([assignment_etas[n][k] for n in range(N)])
                        + b_prior)

    def update_q_partition():
        nonlocal assignment_etas, partition_alphas
        for k in range(K):
            partition_alphas[k] = (sum([assignment_etas[n][k] for n in range(N)])
                                   + alpha_hyperparams[k])

    N = len(data)
    assignment_etas = [[None] * K for _ in range(N)]
    lambda_as, lambda_bs = map(list, zip(*gamma_hyperparams))
    partition_alphas = alpha_hyperparams
    for t in range(n_max_iter):
        for n in range(N):
            update_q_assignment(n)
        for k in range(K):
            update_q_lambda(k)
        if verbose:
            print(f"lambda_abs={list(zip(lambda_as, lambda_bs))}")
        update_q_partition()
        if verbose:
            print(f"partition_alphas={partition_alphas}")
    return np.argmax(assignment_etas, axis=1)
