#!/usr/bin/env python3
""""
This class represents a poisson distribution
"""


import math


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


    def pmf(self, k):
        """Calculate the PMF for the Poisson distribution."""
        # Validate k is an integer
        if k < 0:
            return 0
        
        # Convert k to an integer if it's not already
        k = int(k)
        
        # PMF formula for Poisson distribution: P(k) = (lambtha^k * e^-lambtha) / k!
        try:
            pmf_value = (self.lambtha ** k) * (
                math.exp(-self.lambtha)) / math.factorial(k)
            return pmf_value
        except OverflowError:
            return 0
