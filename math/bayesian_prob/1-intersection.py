#!/usr/bin/env python3
"""
Module that  Calculate the intersection of obtaining observed data
with various hypothetical probabilities
of developing severe side effects in a drug trial.
"""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining observed
    data with various hypothetical probabilities
    of developing severe side effects in a drug trial.

    This function uses the binomial distribution to
    model the probability of observing
    a certain number of patients with severe side
    effects given different probabilities
    of side effect occurrence, and combines this with prior beliefs.
    """
    # Input validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_n_x = np.math.factorial(n - x)
    binomial_coeff = fact_n / (fact_x * fact_n_x)

    likelihoods = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))

    intersection = likelihoods * Pr

    return intersection
