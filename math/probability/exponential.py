#!/usr/bin/env python3
"""
class that represents an
exponential distribution
"""


e = 2.7182818285


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

    def exp(self, x, precision=250):
        """Calculates the exponential of x using a simple series expansion."""
        result = 1.0
        term = 1.0
        for i in range(1, precision):
            term *= x / i
            result += term
            if term < 1e-10:  # Stop if the term becomes very small
                break
        return result

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period x."""
        if x < 0:
            return 0
        # Calculate PDF using the formula
        pdf_value = self.lambtha * e ** (-self.lambtha * x)
        return pdf_value

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period x."""
        if x < 0:
            return 0

        # CDF formula: F(x) = 1 - e^(-lambtha * x)
        exp_neg_lambtha_x = 1 / self.exp(self.lambtha * x)
        cdf_value = 1 - exp_neg_lambtha_x
        return cdf_value
