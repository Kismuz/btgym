import numpy as np
import scipy.signal

from scipy.special import logsumexp

import tensorflow as tf


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def log_uniform(lo_hi, size):
    """
    Samples from log-uniform distribution in range specified by `lo_hi`.
    Takes:
        lo_hi: either scalar or [low_value, high_value]
        size: sample size
    Returns:
         np.array or np.float (if size=1).
    """
    r = np.asarray(lo_hi)
    try:
        lo = r[0]
        hi = r[-1]
    except:
        lo = hi = r
    x = np.random.random(size)
    log_lo = np.log(lo + 1e-12)
    log_hi = np.log(hi + 1e-12)
    v = log_lo * (1 - x) + log_hi * x
    if size > 1:
        return np.exp(v)
    else:
        return np.exp(v)[0]


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def kl_divergence(logits_1, logits_2):
    a0 = logits_1 - tf.reduce_max(logits_1, axis=-1, keepdims=True)
    a1 = logits_2 - tf.reduce_max(logits_2, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)


# def softmax(x):
#     if len(x.shape) > 1:
#         tmp = np.max(x, axis = 1)
#         x -= tmp.reshape((x.shape[0], 1))
#         x = np.exp(x)
#         tmp = np.sum(x, axis = 1)
#         x /= tmp.reshape((x.shape[0], 1))
#     else:
#         tmp = np.max(x)
#         x -= tmp
#         x = np.exp(x)
#         tmp = np.sum(x)
#         x /= tmp
#
#     return x


def softmax(a, axis=None):
    """
    Computes exp(a)/sumexp(a); relies on scipy logsumexp implementation.
    Credit goes to https://stackoverflow.com/users/4115369/yibo-yang

    Args:
        a: ndarray/tensor
        axis: axis to sum over; default (None) sums over everything
    """
    lse = logsumexp(a, axis=axis)  # this reduces along axis
    if axis is not None:
        lse = np.expand_dims(lse, axis)  # restore that axis for subtraction
    return np.exp(a - lse)


def sample_dp(logits, alpha=200.0):
    """
    Given vector of unnormalised log probabilities,
    returns sample of probability distribution taken from induced Dirichlet Process,
    where `logits` define DP mean and `alpha` is inverse variance.

    Args:
        logits:     vector of unnormalised probabilities
        alpha:      scalar, concentration parameter

    Returns:
        vector of probabilities
    """
    return softmax(np.random.multivariate_normal(mean=logits, cov=np.eye(logits.shape[-1]) * alpha ** -1))

