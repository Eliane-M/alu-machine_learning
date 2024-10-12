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
        result = 1
        term = 1
        for i in range(1, 200):
            term *= x / i
            result += term
        return result

    def pmf(self, k):
        """Calculate the PMF for the Poisson distribution."""
        # Validate k is an integer
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        factorial = 1
        for i in range(k):
            factorial *= i + 1
        pmf = ((lambtha**k) * (e**-lambtha)) / factorial
        return pmf

    def cdf(self, k):
        """Calculates the CDF for a given number of 'successes' k."""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
