import numpy as np
import scipy.signal
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
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    v = log_lo * (1 - x) + log_hi * x
    if size > 1:
        return np.exp(v)
    else:
        return np.exp(v)[0]


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def kl_divergence(logits_1, logits_2):
    a0 = logits_1 - tf.reduce_max(logits_1, axis=-1, keep_dims=True)
    a1 = logits_2 - tf.reduce_max(logits_2, axis=-1, keep_dims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
