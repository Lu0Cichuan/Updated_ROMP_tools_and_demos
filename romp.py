import numpy as np

from tools import generate_random_signal, generate_t_line, draw_single_signal, draw_triple_signal, draw_double_signal


def romp(original_signal, dictionary, time_tolerance=100):
    support = []
    try:
        support_all = np.zeros(dictionary.shape[1])
    except IndexError:
        support_all = np.zeros(1)
    residual = original_signal
    max_steps = dictionary.shape[0]
    for _ in range(min(time_tolerance,max_steps)):
        correlations = np.dot(dictionary.T, residual)
        max_correlation = np.max(abs(correlations))
        new_support = np.where(abs(correlations) >= 0.9 * max_correlation)[0]
        support = np.union1d(support, new_support)
        A = dictionary[:, list(map(int,support))]
        x_hat, _, _, _ = np.linalg.lstsq(A, original_signal, rcond=None)
        residual = count_residual(original_signal,A,x_hat)

        support_all[support.astype(int)] = x_hat
        mse = np.mean((residual) ** 2)
        if mse < 1e-6:
            break

    return support_all, residual


def count_residual(original_signal, dictionary, support):
    recovery = np.dot(dictionary,support)
    residual = original_signal - recovery
    return residual


