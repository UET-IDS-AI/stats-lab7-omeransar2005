"""
AI_stats_lab.py
Lab: Uniform Random Variable Analysis
"""
import numpy as np


def uniform_analysis(a, n_samples=10000):
    """
    Analyzes a Uniform(0, a) random variable — both theoretically and
    through simulation — then examines the linear transformation Z = 2X + 1.

    Parameters:
        a         : upper bound of the uniform distribution
        n_samples : number of random samples to draw (default: 10000)

    Returns:
    (
        theoretical_mean,
        theoretical_variance,
        sample_mean,
        sample_variance,
        mean_error,
        variance_error,
        transformed_mean,
        transformed_variance
    )
    """

    # ── Theoretical values ────────────────────────────────────────────────────
    # For X ~ Uniform(0, a):
    #   E[X]   = a / 2
    #   Var[X] = a² / 12
    theoretical_mean     = a / 2
    theoretical_variance = (a ** 2) / 12

    # ── Sample-based estimates ────────────────────────────────────────────────
    # Draw n_samples points from Uniform(0, a) and estimate mean & variance.
    # ddof=0 (population variance) keeps us consistent with numpy.var's default.
    samples          = np.random.uniform(0, a, n_samples)
    sample_mean      = np.mean(samples)
    sample_variance  = np.var(samples)           # ddof=0 by default

    # ── Errors ────────────────────────────────────────────────────────────────
    # How far are our sample estimates from the true theoretical values?
    mean_error      = abs(sample_mean     - theoretical_mean)
    variance_error  = abs(sample_variance - theoretical_variance)

    # ── Linear transformation Z = 2X + 1 ─────────────────────────────────────
    # By the rules of expectation and variance for linear transforms:
    #   E[2X + 1]   = 2·E[X] + 1   = a + 1
    #   Var[2X + 1] = 2²·Var[X]    = 4 · (a² / 12) = a² / 3
    transformed_mean     = 2 * theoretical_mean     + 1
    transformed_variance = 4 * theoretical_variance

    return (
        theoretical_mean,
        theoretical_variance,
        sample_mean,
        sample_variance,
        mean_error,
        variance_error,
        transformed_mean,
        transformed_variance,
    )


if __name__ == "__main__":
    print("Implement uniform_analysis(a, n_samples=10000)")
