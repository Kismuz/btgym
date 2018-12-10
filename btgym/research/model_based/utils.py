import numpy as np


def log_uniform(lo_hi, size):
    """
    Samples from log-uniform distribution in range specified by `lo_hi`.
    Takes:
        lo_hi: either scalar or iterable in form [low_value, high_value]
        size: sample size
    Returns:
         np.array or np.float (if size=1).
    """
    r = np.asarray(lo_hi)
    try:
        lo = r[0]
        hi = r[-1]
    except IndexError:
        lo = hi = r
    x = np.random.random(size)
    log_lo = np.log(lo + 1e-12)
    log_hi = np.log(hi + 1e-12)
    v = log_lo * (1 - x) + log_hi * x
    if size > 1:
        return np.exp(v)
    else:
        return np.exp(v)[0]


def ou_mle_estimator(data, dt=1):
    """
    Estimates vanilla OU max. log-likelihood parameters from given data of size [num_trajectories, num_points].

    Returns:
         tuple of vectors (mu, lambda, sigma) of size [num_trajectories] each.

    Note: robust against highly biased data i.e. where data.mean / data.std  >> 1
    """
    if len(data.shape) == 1:
        data = data[None, :]
    elif len(data.shape) > 2:
        raise AssertionError('Only 1D and 2D data accepted')

    # Remove bias from every trajectory:
    bias = data.mean(axis=-1)
    data -= bias[:, None]

    n = data.shape[-1]
    x = data[:, :-1]
    y = data[:, 1:]
    sx = x.sum(axis=-1)
    sy = y.sum(axis=-1)
    sxx = (x ** 2).sum(axis=-1)
    sxy = (x * y).sum(axis=-1)
    syy = (y ** 2).sum(axis=-1)

    mu_denom = (n * (sxx - sxy) - (sx**2 - sx * sy))
    mu_denom[np.logical_and(0 <= mu_denom, mu_denom < 1e-10)] = 1e-10
    mu_denom[np.logical_and(-1e-10 < mu_denom, mu_denom < 0)] = -1e-10
    mu = (sy * sxx - sx * sxy) / mu_denom

    l_denom = sxx - 2 * mu * sx + n * (mu ** 2)
    l_denom[np.logical_and(0 <= l_denom, l_denom < 1e-10)] = 1e-10
    l_denom[np.logical_and(-1e-10 < l_denom, l_denom < 0)] = -1e-10

    l = - (1 / dt) * np.log(
        np.clip(
            (sxy - mu * sx - mu * sy + n * (mu ** 2)) / l_denom,
            1e-10,
            None
        )
    )
    l = np.clip(l, 1e-10, None)

    a = np.exp(-l * dt)

    sigma_sq_hat = (1 / n) * (
            syy - 2 * a * sxy + a ** 2 * sxx - 2 * mu * (1 - a) * (sy - a * sx) + n * (mu ** 2) * (1 - a) ** 2
    )

    sigma_sq_denom = 1 - a ** 2
    sigma_sq_denom = np.clip(sigma_sq_denom, 1e-10, None)

    sigma_sq = sigma_sq_hat * (2 * l / sigma_sq_denom)

    sigma = np.clip(sigma_sq, 1e-10, None) ** .5

    # Set bias back:
    mu += bias
    data += bias[:, None]  # in_place cleanup

    return np.squeeze(mu), np.squeeze(l), np.squeeze(sigma)


def ou_lsr_estimator(data, dt=1):
    """
    Estimates vanilla OU parameters via least squares method from given data of size [num_trajectories, num_points].
    Returns tuple of vectors (mu, lambda, sigma) of size [num_trajectories] each.

    Note: robust against highly biased data i.e. where data.mean / data.std  >> 1
    """
    if len(data.shape) == 1:
        data = data[None, :]
    elif len(data.shape) > 2:
        raise AssertionError('Only 1D and 2D data accepted')

    # Remove bias from every trajectory:
    bias = data.mean(axis=-1)
    data -= bias[:, None]

    n = data.shape[-1]
    x = data[:, :-1]
    y = data[:, 1:]
    sx = x.sum(axis=-1)
    sy = y.sum(axis=-1)
    sxx = (x ** 2).sum(axis=-1)
    sxy = (x * y).sum(axis=-1)
    syy = (y ** 2).sum(axis=-1)

    a = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    b = (sy - a * sx) / n
    sd_e = ((n * syy - sy ** 2 - a * (n * sxy - sx * sy)) / (n * (n - 2))) ** .5

    l = - np.log(a) / dt
    mu = b / (1 - a)
    sigma = sd_e * (-2 * np.log(a) / (dt * (1 - a ** 2))) ** .5

    # Set bias back:
    mu += bias
    data += bias[:, None]  # in_place cleanup

    return np.squeeze(mu), np.squeeze(l), np.squeeze(sigma)


def ou_variance(l, sigma, **kwargs):
    """
    Returns true OU process variance.
    """
    return np.clip(sigma**2, 0, None) / (2 * np.clip(l, 1e-10, None))




