#!/usr/bin/env python3
"""
class that represents an
exponential distribution
"""


class Exponential():
    """Represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Class constructor for Exponential distribution."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha as the inverse of the mean of the data
            self.lambtha = float(1 / (sum(data) / len(data)))
