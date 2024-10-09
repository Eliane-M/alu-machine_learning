#!/usr/bin/env python3
""""
This class represents a poisson distribution
"""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor for Poisson distribution."""
        # Validate lambtha if data is not given
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate data is a list
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            # Ensure the data list has at least two points
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha as the average of data points
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """Calculates the factorial of n."""
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def exp(self, x):
        """Calculates the exponential of x using Taylor series expansion."""
        result = 1.0
        term = 1.0
        for i in range(1, 100):
            term *= x / i
            result += term
        return result

    def pmf(self, k):
        """Calculate the PMF for the Poisson distribution."""
        # Validate k is an integer
        if k < 0:
            return 0

        # Convert k to an integer if it's not already
        k = int(k)

        # PMF formula: P(k)=(lambtha^k * e^-lambtha) / k!
        lambtha_pow_k = 1
        for _ in range(k):
            lambtha_pow_k *= self.lambtha

        exp_neg_lambtha = 1 / self.exp(self.lambtha)

        pmf_value = (lambtha_pow_k * exp_neg_lambtha) / self.factorial(k)
        return pmf_value
