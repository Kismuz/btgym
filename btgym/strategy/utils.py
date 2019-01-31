import  numpy as np


def log_transform(x):
    return np.sign(x) * np.log(np.fabs(x) + 1)


def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def abs_norm_ratio(x, a, b):
    """
    Norm. V-shaped realtive position of x in [a,b], a<=x<=b.
    """
    return abs((2 * x - a - b) / (abs(a) + abs(b)))


def norm_log_value(current_value, start_value, drawdown_call, target_call, epsilon=1e-4):
    """
    Current value log-normalized in [-1,1] wrt p/l limits.
    """
    x = np.asarray(current_value)
    x = (x / start_value - 1) * 100
    x = (x - target_call) / (drawdown_call + target_call) + 1
    x = np.clip(x, epsilon, 1 - epsilon)
    x = 1 - 2 * np.log(x) / np.log(epsilon)
    return x


def norm_value(current_value, init_value, lower_bound, upper_bound, epsilon=1e-8):
    """
    Current value normalized in [-1,1] wrt upper and lower bounds.
    """
    x = np.asarray(current_value)
    x = (x / init_value - 1) * 100
    x = (x - upper_bound) / (lower_bound + upper_bound) + 1
    x = 2 * np.clip(x, epsilon, 1 - epsilon) - 1
    return x


def __norm_value(current_value, init_value, lower_bound, upper_bound, epsilon=1e-8):
    """
    Current value, piece-wise linear normalized in [-1,1] and zero-centered  at `start_value`
    """
    lower_bound /= 100
    upper_bound /= 100
    x = np.asarray(current_value)
    x1 = (1 / (init_value * lower_bound)) * x[x < init_value] - 1 / lower_bound
    x2 = (1 / (init_value * upper_bound)) * x[x >= init_value] - 1 / upper_bound
    x = np.concatenate([x1, x2], axis=-1)
    x = np.squeeze(np.clip(x, -1, 1))
    return x


def decayed_result(trade_result, current_value, base_value, lower_bound, upper_bound, gamma=1.0):
    """
    Normalized in [-1,1] trade result, lineary decayed wrt current_value.
    """
    target_value = base_value * (1 + upper_bound / 100)
    value_range = base_value * (lower_bound + upper_bound) / 100
    decay = (gamma - 1) * (current_value - target_value) / value_range + gamma
    x = trade_result * decay / value_range
    return x


# def decayed_result(trade_result, current_value, base_value, lower_bound, upper_bound, gamma=1.0):
#     """
#     Normalized in [-1,1] trade result, lineary decayed wrt current_value.
#     """
#     return (trade_result - base_value) / (upper_bound - lower_bound)


def exp_scale(x, gamma=4, epsilon=1e-10):
    """
    Returns exp. scaled value in [epsilon, 1] for x in [0, 1]; gamma controls steepness.
    """
    x = np.asarray(x) + 1
    return np.clip(np.exp(x ** gamma - 2 ** gamma), epsilon, 1)


def discounted_average(x, gamma=1):
    """
    Returns gamma_power weighted average of 2D input array along 0-axis.

    Args:
        x:      scalar or array of rank <= 2.
        gamma: discount, <=1.

    Returns:
        Array of rank 1 of averages along zero axis. For x of shape[n,m] AVG computed as:
        AVG[j] = (x[0,j]*gamma^(n) + x[1,j]*gamma^(n-1) +...+  x[n,j]*gamma^(0)) / n

    """
    x = np.asarray(x)
    while len(x.shape) < 2:
        x = x[..., None]
    gamma = gamma * np.ones(x.shape)
    return np.squeeze(np.average(x, weights=(gamma ** np.arange(x.shape[0])[..., None])[::-1], axis=0))