import torch, numpy as np


def gauss_window(M: float, std: torch.FloatTensor, sym: bool = True):
    """
    Returns a Gaussian window tensor.

    This function generates a Gaussian window, given the window length and standard deviation.
    It's an adapted version from scipy.signal.gaussian.

    Args:
        M (float): The number of points in the output window.
        std (torch.FloatTensor): Standard deviation of the Gaussian window.
        sym (bool, optional): When True (default), generates a symmetric window for filter design.
            If False, the window is not symmetric.

    Returns:
        torch.Tensor: The generated Gaussian window tensor of shape (M,).

    Notes:
        If M < 1, an empty tensor is returned.
        If M == 1, a tensor with a single value of 1 is returned.
    """
    if M < 1:
        return torch.array([])
    if M == 1:
        return torch.ones(1, "d").type_as(std)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    n = n.type_as(std)

    sig2 = 2 * std * std
    w = torch.exp(-(n**2) / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def generate_am_chirp(
    theta: torch.FloatTensor,
    bw: float = 2,
    duration: float = 4,
    sr: float = 2**13,
    delta: int = 0,
):
    """
    Generate a batch of amplitude-modulated (AM) chirp signal.

    The function generates an AM chirp signal windowed in the time domain by a Gaussian function.

    Args:
        theta (torch.FloatTensor): A tensor of shape (N, 3) containing:
            - The carrier frequency (f_c) in Hz.
            - The modulator frequency (f_m) in Hz.
            - The chirp rate (gamma) in octaves/second.
        bw (float, optional): The bandwidth of the chirp signal in octaves. Defaults to 2.
        duration (float, optional): The duration of the chirp signal in seconds. Defaults to 4.
        sr (float, optional): The sample rate of the chirp signal in samples per second. Defaults to 2**13.
        delta (int, optional): Applies a time shift in samples. (Currently unused in the function.)

    Returns:
        torch.Tensor: The generated batch of amplitude-modulated chirp signals.

    Example:
        >>> theta = torch.stack([torch.tensor([512.0, 1024.0]), torch.tensor([8.0, 4.0]), torch.tensor([1.0, 2.0])])
        >>> signal = generate_am_chirp(theta, bw=5, duration=2, sr=44100, delta=100)
        >>> signal.shape
    """
    f_c, f_m, gamma = theta[:, 0][:, None], theta[:, 1][:, None], theta[:, 2][:, None]
    t = torch.arange(-duration / 2, duration / 2, 1 / sr).type_as(f_m)[None, :]
    carrier = sine(f_c / (gamma * np.log(2)) * (2 ** (gamma * t) - 1))
    modulator = sine(t * f_m)
    sigma0 = 0.1
    window_std = (torch.tensor(sigma0 * bw).type_as(gamma)) / gamma
    window = gauss_window(duration * sr, std=window_std * sr)

    x = carrier * modulator * window * gamma
    if delta:
        x = time_shift(x, delta)
    return x


def sine(f):
    """
    Compute the sine of a given instantaneous frequency.

    Args:
        f (torch.Tensor): The instantaneous frequencies.

    Returns:
        torch.Tensor: sine signal.
    """
    return torch.sin(2 * torch.pi * f)


def time_shift(x, delta):
    y = torch.zeros_like(x)
    y[delta:] = x[:-delta]
    return y


def harmonic_signal(
    theta: torch.Tensor, duration: float = 1, sr: int = 8192
) -> torch.Tensor:
    """
    Generate an additive Fourier signal based on the given parameters.

    Parameters:
    - theta (torch.Tensor): A tensor containing three values:
        - f1 (float): The frequency of the first harmonic.
        - alpha (float): Fourier decay coefficient (controls brightness).
        - r (float): odd-even harmonic ratio in [-1, 1].
      The shape of `theta` should be (batch_size, 3) where batch_size is the number of examples.

    - duration (float, optional): The duration of the generated signal in seconds. Default is 1 second.
    - sr (int, optional): The sample rate of the generated signal. Default is 8192 Hz.

    Returns:
    - torch.Tensor: The generated harmonic signal with shape (batch_size, duration*sr).
    """

    t = torch.arange(0, duration, 1 / sr)
    f1, alpha, r = theta[:, 0], theta[:, 1], theta[:, 2]
    N = torch.round(sr / (2 * f1)).int() - 1
    a = torch.tensor(
        [(1 + np.power(-1, n) * r) / (np.power(n, alpha)) for n in range(1, N + 1)]
    )[:, None]
    harmonics = torch.stack(
        ([torch.cos(2 * np.pi * n * f1 * t) for n in range(1, N + 1)])
    )
    window = torch.hann_window(duration * sr)[None, :]
    x = (a * harmonics * window).sum(dim=0)

    return x


def generate_harmonic_signals(
    theta: torch.Tensor, duration: float = 1, sr: int = 8192
) -> torch.Tensor:
    """
    Generate multiple harmonic signals based on a batch of given parameters.

    Parameters:
    - theta (torch.Tensor): A tensor containing the parameters for each harmonic signal.
      Each row should contain three values for f1, alpha, and r.
      The shape of `theta` should be (batch_size, 3) where batch_size is the number of examples.

    - duration (float, optional): The duration of the generated signal in seconds. Default is 1 second.
    - sr (int, optional): The sample rate of the generated signal. Default is 8192 Hz.

    Returns:
    - torch.Tensor: A batch of generated harmonic signals with shape (batch_size, duration*sr).
    """

    X = torch.stack([harmonic_signal(th[None, :], duration, sr) for th in theta])
    return X
