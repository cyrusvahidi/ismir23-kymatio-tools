import librosa
import torch, numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_spec(y, hop_length=256, n_fft=4096, sr=2**13):
    fig, ax = plt.subplots()
    D = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="magma_r"
    )


def plot_cqt(y, hop_length=32, bins_per_octave=24, sr=2**13, ax=None):
    """
    Plot the Constant-Q Transform (CQT) of a given signal.

    This function uses the librosa library to compute and display the CQT of the input signal `y`.

    Args:
        y (np.ndarray): The input signal for which the CQT will be computed.
        hop_length (int, optional): The number of samples between successive CQT columns. Defaults to 32.
        bins_per_octave (int, optional): Number of bins per octave. Determines the resolution of the CQT. Defaults to 24.
        sr (int, optional): The sampling rate of the input signal in Hz. Defaults to 2**13.
        ax (matplotlib.axes.Axes, optional): The axes on which to display the CQT. If None (default), the current active axis will be used.

    Notes:
        The function uses a fixed fmin (minimum frequency) of 2**6 Hz.
        The `n_bins` parameter for the CQT computation is fixed at 120.
        The displayed CQT magnitude is scaled by raising to the power of 0.33 to improve visibility.
        The y-axis limit of the plot is commented out, but can be adjusted by uncommenting and modifying the `plt.ylim` line.
    """
    cqt_kwargs = {
        "sr": sr,
        "fmin": 2**6,
        "bins_per_octave": bins_per_octave,
        "hop_length": hop_length,
    }
    CQT = librosa.cqt(y, n_bins=120, **cqt_kwargs)
    librosa.display.specshow((np.abs(CQT) ** 0.33), **cqt_kwargs, ax=ax)


def grid2d(x1: float, x2: float, y1: float, y2: float, n: float):
    a = torch.logspace(np.log10(x1), np.log10(x2), n)
    b = torch.logspace(np.log10(y1), np.log10(y2), n)
    X = a.repeat(n)
    Y = b.repeat(n, 1).t().contiguous().view(-1)
    return X, Y


def plot_gradient_field(x, y, u, v, x_range, y_range, target, save_path):
    plt.figure()

    plt.scatter(x, y, color="r")

    u = np.array(u) / np.max(np.abs(u))  # gradient wrt AM
    v = np.array(v) / np.max(np.abs(v))  # gradient wrt FM
    grads = np.stack([u, v])
    grads = grads / (np.linalg.norm(grads, axis=0) + 1e-8)
    plt.quiver(x, y, grads[0, :], grads[1, :])

    plt.scatter([target[0]], [target[1]], color="g")

    plt.xticks(np.arange(x_range[0], x_range[1] + 1))
    plt.yticks(np.arange(y_range[0], y_range[1] + 1))
    plt.xlabel("AM (Hz)")
    plt.ylabel("FM (oct / s)")
    plt.show()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)


def plot_contour_gradient(X, Y, Z, target_idx, grads, ylabel="FM (oct / s)"):
    """
    X, Y, Z: meshgrid (N, N) matrices
    target_idx: index of the target to scatter in green
    grads: list of gradients [u, v]
    save_path: where to save
    """
    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, 20, cmap=cm.coolwarm)

    target = [X.ravel()[target_idx], Y.ravel()[target_idx]]
    plt.scatter(target[0], target[1], color="g", alpha=1)

    grads = np.stack(grads)
    u = np.array(grads[:, 0]) / np.max(np.abs(grads[:, 0]))  # gradient wrt AM
    v = np.array(grads[:, 1]) / np.max(np.abs(grads[:, 1]))  # gradient wrt FM
    grads = np.stack([u, v])
    grads = grads / (np.linalg.norm(grads, axis=0) + 1e-8)
    ax.quiver(X.reshape(-1), Y.reshape(-1), grads[0, :], grads[1, :])

    # Plot Labelling
    plt.xlabel("AM (Hz)")
    ax.loglog()
    plt.ylabel(ylabel)
    plt.rcParams["axes.formatter.min_exponent"] = 2
    plt.show()


def mesh_plot_3d(X, Y, Z, target_idx, ylabel="FM (oct / s)"):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    target = [X.ravel()[target_idx], Y.ravel()[target_idx]]

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.5)

    ax.scatter(target[0], target[1], color="g", alpha=1)

    ax.set_xlabel("AM (Hz)")
    ax.set_ylabel(ylabel)

    plt.show()


def plot_isomap(Z, params, labels=[]):
    """
    Visualize the Isomap embedding in a 3D plot with parameters for color-coding.

    Parameters:
    -----------
    Z : numpy.array
        The reduced data after Isomap embedding.
        Shape: (n_samples, 3)

    params : list of numpy.array
        A list containing 3 arrays which represent the parameters to be used
        for color-coding in the 3D plots. Each array should match the number of
        data points in `Z`.

    labels : list of str
        A list containing 3 labels, one for each of the plots to indicate the parameter
        that has been used for color-coding

    Returns:
    --------
    ax
        The function visualizes the 3D plots for the Isomap embedding with
        different color codings based on `params`.
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    axs = []

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], c=params[i], cmap='bwr');

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.title.set_text(labels[i])
        axs.append(ax)

    plt.subplots_adjust(wspace=0, hspace=0)

    # rotate the axes and update
    for angle in range(60, 360, 60):
        for ax in axs:
            ax.view_init(30, angle)
            plt.draw()