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


def ou_mle_estimator(data, dt=1, force_zero_mean=True):
    """
    Estimates vanilla OU max. log-likelihood parameters from given data of size [num_trajectories, num_points].

    Returns:
         tuple of vectors (mu, lambda, sigma) of size [num_trajectories] each.

    Note:
        robust against:
            highly biased data i.e. where data.mean / data.std  >> 1
            border conditions i.e. OU --> Weiner (unit root process)
    """
    if len(data.shape) == 1:
        data = data[None, :]
    elif len(data.shape) > 2:
        raise AssertionError('Only 1D and 2D data accepted')

    # Center every trajectory:
    bias = data.mean(axis=-1)
    data -= bias[:, None]

    n = data.shape[-1] - 1
    x = data[:, :-1]
    y = data[:, 1:]
    sx = x.sum(axis=-1)
    sy = y.sum(axis=-1)
    sxx = (x ** 2).sum(axis=-1)
    sxy = (x * y).sum(axis=-1)
    syy = (y ** 2).sum(axis=-1)

    if force_zero_mean:
        # Assume OU mean is zero for centered data, compromises MLE but prevents
        # trashy MU values for unit-root processes:
        mu = 0

    else:
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

    n = data.shape[-1] - 1
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


def ou_log_likelihood(mu, l, sigma, data):
    """
    Estimates OU model parameters log likelihood given data log[P(mu, lambda, sigma|X)]
    """
    x = data[1:]
    x_prev = data[:-1]
    logL = - .5 * np.log(2 * np.pi) - np.log(sigma) \
           - 1 / (2 * sigma ** 2) * ((x - x_prev * np.exp(-l) - mu * (1 - np.exp(-l))) ** 2).mean(axis=-1)
    return logL


def batch_covariance(x):
    """
    Batched covariance matrix estimation.
    Credit to: Divakar@stackoverflow.com, see:
    https://stackoverflow.com/questions/40394775/vectorizing-numpy-covariance-for-3d-array

    Args:
        x:  array of size [batch_dim, num_variables, num_observations]

    Returns:
        estimated covariance matrix of size [batch_dim, num_variables, num_variables]
    """
    n = x.shape[2]
    m1 = x - x.sum(2, keepdims=1) / n
    return np.einsum('ijk,ilk->ijl', m1, m1) / (n - 1)


def multivariate_t_rvs(mean, cov, df, size):
    """
    Student's T random variable.
    Generates random samples from multivariate t distribution.

    Code credit:
    written by Enzo Michelangeli, style changes by josef-pktd;
    https://github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/distributions/multivariate.py#L90

    Args:
        mean:   array_like, mean of random variable of size [dim], length determines dimensionality of random variable
        cov:    array_like, covariance  matrix of size [dim, dim]
        df:     array_like > 0, degrees of freedom of size [dim]
        size:   array_like, size of observations to draw

    Returns:
        rvs as ndarray of size: size + [dim], i.e. if size=[m, n] than returned sample is: [m, n, dim]
    """
    # t-variance memo: ((df - 2) / df) ** .5
    mean = np.asarray(mean)
    df = np.asarray(df)

    if type(size) in [int, float]:
        size = [int(size)]
    else:
        size = list(size)

    assert mean.ndim == 1 and df.shape == mean.shape, \
        'Expected `mean` and `df` be 1d array_like of same size, got shapes: {} and {}'.format(mean.shape, df.shape)

    d = len(mean)

    assert cov.shape == (d, d), 'Dimensionality: {} does not match covariance shape: {}'.format(d, cov.shape)

    x = np.random.chisquare(df, size + [d]) / df
    z = np.random.multivariate_normal(np.zeros(d), cov, size)

    return mean[None, :] + z / np.sqrt(x)


def cov2corr(cov):
    """
    Converts covariance matrix to correlation matrix.

    Args:
        cov:    square matrix

    Returns:
        correlation matrix of the same size.
    """
    cov = np.asanyarray(cov)
    std = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    corr = cov / np.outer(std, std)
    return corr


def log_stat2stat(log_mean, log_variance):
    """
    Converts mean and variance of log_transformed RV
    to mean and variance of original near-normally distributed RV.

    Args:
        log_mean:       array_like
        log_variance:   array_like

    Returns:
        mean, variance of the same size
    """
    mean = np.exp(log_mean + 0.5 * log_variance)
    variance = mean**2 * (np.exp(log_variance) - 1)

    return mean, variance








