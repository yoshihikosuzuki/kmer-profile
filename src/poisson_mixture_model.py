from typing import Sequence, Tuple, List
import random
import numpy as np


def gibbs_sampling(data: Sequence[int],
                   K: int,
                   gamma_hyperparams: Sequence[Tuple[float, float]],
                   alpha_hyperparams: Sequence[int],
                   n_max_iter: int) -> List[int]:
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
        a_prioir, b_prior = gamma_hyperparams[k]
        a = sum([data[n] for n in range(N) if assignments[n] == k]) + a_prioir
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
    # Start from random assignments
    assignments = random.choices(list(range(K)), k=N)
    lambdas = [None] * K
    partitions = [None] * K
    for t in range(n_max_iter):
        for k in range(K):
            update_lambda(k)
        print(f"lamdas={lambdas}")
        update_partition()
        print(f"partitions={partitions}")
        for n in range(N):
            update_assignment(n)
        #print(f"assignments={assignments}")
        # TODO: check if the permutation of clusters increases the likelihood
        #       (when component-specific hyperparameters are specified)
    return assignments


def variational_bayes(data: Sequence[int],
                      K: int) -> List[int]:
    pass
