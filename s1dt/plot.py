import os
import imageio
from typing import List, Dict, Any, Tuple, Optional
import librosa
import torch, numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from fractions import Fraction
from tqdm import tqdm

import torch as tr
from torch import Tensor

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


def plot_scalogram(scalogram: Tensor,
                   sr: float,
                   y_coords: List[float],
                   title: str = "scalogram",
                   hop_len: int = 1,
                   cmap: str = "magma",
                   vmax: Optional[float] = None,
                   save_path: Optional[str] = None,
                   x_label: str = "Time (seconds)",
                   y_label: str = "Frequency (Hz)") -> None:
    """
    Plots a scalogram of the provided data.

    The scalogram is a visual representation of the wavelet transform of a signal over time.
    This function uses matplotlib and librosa to create the plot.

    Parameters:
        scalogram (T): The scalogram data to be plotted.
        sr (float): The sample rate of the audio signal.
        y_coords (List[float]): The y-coordinates for the scalogram plot.
        title (str, optional): The title of the plot. Defaults to "scalogram".
        hop_len (int, optional): The hop length for the time axis (or T). Defaults to 1.
        cmap (str, optional): The colormap to use for the plot. Defaults to "magma".
        vmax (Optional[float], optional): The maximum value for the colorbar. If None, the colorbar scales with the data. Defaults to None.
        save_path (Optional[str], optional): The path to save the plot. If None, the plot is not saved. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to "Time (seconds)".
        y_label (str, optional): The label for the y-axis. Defaults to "Frequency (Hz)".
    """
    assert scalogram.ndim == 2
    assert scalogram.size(0) == len(y_coords)
    x_coords = librosa.times_like(scalogram.size(1), sr=sr, hop_length=hop_len)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(scalogram.numpy(),
                             sr=sr,
                             x_axis="time",
                             x_coords=x_coords,
                             y_axis="cqt_hz",
                             y_coords=np.array(y_coords),
                             cmap=cmap,
                             vmin=0.0,
                             vmax=vmax)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    if len(y_coords) < 12:
        ax = plt.gca()
        ax.set_yticks(y_coords)
    plt.minorticks_off()
    plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def grid2d(x1: float, x2: float, y1: float, y2: float, n: float):
    """
    Generates a 2D grid of logarithmically spaced points.

    This function generates two 1D arrays: one for the x-coordinates and one for the y-coordinates. 
    The points are logarithmically spaced between the provided bounds (x1, x2) and (y1, y2).

    Parameters:
    x1 (float): The lower bound for the x-coordinates.
    x2 (float): The upper bound for the x-coordinates.
    y1 (float): The lower bound for the y-coordinates.
    y2 (float): The upper bound for the y-coordinates.
    n (float): The number of points to generate for each coordinate.

    Returns:
    X (Tensor): A 1D tensor of the x-coordinates of the grid points.
    Y (Tensor): A 1D tensor of the y-coordinates of the grid points.
    """
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

def plot_knn_regression(ratios):
    plt.clf()
    yticklabels = ["1/3", "1/2", "1", "2", "3"]
    objs = ["Carrier freq.", "Modulation freq.", "Chirp rate"]

    N = len(ratios[list(ratios.keys())[0]][:, 0])

    fig, axes = plt.subplots(ncols=3, figsize=plt.figaspect(.5), sharey=True)
    for i, ratio in enumerate(ratios.values()):
        for idx, ax in enumerate(axes.flat):
            ax.plot(np.random.uniform(i - 0.1, i + 0.1, N),
                    np.log2(ratio[:, idx]), ".", markersize=1)

            ax.set_yticks(np.log2(np.array([float(Fraction(label))
                                            for label in yticklabels])))
            ax.set_xticks([i for i in range(len(ratios))])
            ax.set_xticklabels(list(ratios.keys()), fontsize=13)
            ax.set_yticklabels(yticklabels)
            ax.grid(linestyle="--")
            ax.set_title(objs[idx], fontsize=14)
            if idx == 0:
                ax.set_ylabel("Relative estimate", fontsize=14)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=.01)
    plt.tight_layout()


def extract_s2_scalogram(s_scat_1d: Dict[Tuple[int, int], Tensor],
                         sr: float,
                         psi1_f_idx: int,
                         psi2_f: List[Dict[str, Any]],
                         pad: bool = True,
                         is_jtfst: bool = False) -> (Tensor, List[float]):
    """
    Extracts a second-order scalogram from a dictionary of scattering coefficients.

    This function iterates over the dictionary `s_all` and collects the scattering coefficients corresponding to the provided `s1_idx`.
    If a coefficient is not found and `pad` is True, a tensor of zeros is added instead.
    The collected coefficients are then stacked into a tensor to form the scalogram.

    Parameters:
        s_scat_1d (Dict[Tuple[int, int], torch.Tensor]): A dictionary where the keys are tuples of indices and the values are scattering coefficients.
        sr: Sampling rate.
        psi1_f_idx (int): The index of the first-order scattering coefficient to extract.
        psi2_f (List[Dict[str, Any]]): The list of second-order filters.
        pad (bool, optional): Whether to add a tensor of zeros when a coefficient is not found. Defaults to True.
        is_jtfst (bool, optional): Whether the scattering coefficients are from a JTFST. Defaults to False.

    Returns:
        scalogram (torch.Tensor): The extracted second-order scalogram.
        filter_freqs (List[float]): The frequencies of the second-order filters.
    """
    assert s_scat_1d, "s_scat_1d is empty"
    n_psi2_f = len(psi2_f)
    rows = []
    psi2_f_indices = []
    for curr_psi2_f_index in range(n_psi2_f):
        if is_jtfst:
            key = (psi1_f_idx, curr_psi2_f_index, 0)
        else:
            key = (psi1_f_idx, curr_psi2_f_index)
        if key in s_scat_1d:
            rows.append(s_scat_1d[key])
            psi2_f_indices.append(curr_psi2_f_index)
        else:
            if pad:
                rows.append(tr.zeros_like(list(s_scat_1d.values())[0]))
                psi2_f_indices.append(curr_psi2_f_index)

    assert rows, f"No scattering coefficients were found for psi1_f_idx of {psi1_f_idx}"
    scalogram = tr.stack(rows, dim=0)
    filter_freqs = [psi2_f[idx]["xi"] * sr for idx in psi2_f_indices]
    return scalogram, filter_freqs


def create_s2_gif(s_scat_1d: Dict[Tuple[int, int], Tensor],
                  sr: float,
                  psi1_f: List[Dict[str, Any]],
                  psi2_f: List[Dict[str, Any]],
                  save_dir: str,
                  save_name: str,
                  title_prefix: str = "S2",
                  min_s1_freq_hz: float = 0,
                  max_s1_freq_hz: float = np.inf,
                  hop_len: int = 1,
                  use_vmax: bool = True,
                  discard_empty: bool = False,
                  fps: int = 7,
                  is_jtfst: bool = False,
                  x_label: str = "Time (seconds)",
                  y_label: str = "2nd Order Filter Frequency (Hz)") -> None:
    """
    Creates a gif of second-order scalograms.

    Parameters:
        s_scat_1d (Dict[Tuple[int, int], torch.Tensor]): A dictionary where the keys are tuples of indices and the values are scattering coefficients.
        sr: Sampling rate.
        psi1_f (List[Dict[str, Any]]): The list of first-order filters.
        psi2_f (List[Dict[str, Any]]): The list of second-order filters.
        save_dir (str): The directory to save the gif.
        save_name (str): The name of the gif.
        title_prefix (str, optional): The prefix for the title of each frame. Defaults to "S2".
        min_s1_freq_hz (float, optional): The minimum frequency of the first-order filter. Defaults to 0.
        max_s1_freq_hz (float, optional): The maximum frequency of the first-order filter. Defaults to np.inf.
        hop_len (int, optional): The hop length for the time axis (or T). Defaults to 1.
        use_vmax (bool, optional): Whether to use a fixed maximum value for the colorbar. Defaults to True.
        discard_empty (bool, optional): Whether to discard frames with no scattering coefficients. Defaults to False.
        fps (int, optional): The frames per second for the gif. Defaults to 7.
        is_jtfst (bool, optional): Whether the scattering coefficients are from a JTFST. Defaults to False.
        x_label (str, optional): The label for the x-axis. Defaults to "Time (seconds)".
        y_label (str, optional): The label for the y-axis. Defaults to "2nd Order Filter Frequency (Hz)".
    """
    frames_s2 = []
    for psi1_f_idx in range(len(psi1_f)):
        s1_freq = psi1_f[psi1_f_idx]["xi"] * sr
        if s1_freq < min_s1_freq_hz or s1_freq > max_s1_freq_hz:
            continue
        s2, filter_freqs = extract_s2_scalogram(s_scat_1d, sr, psi1_f_idx, psi2_f,
                                                pad=True, is_jtfst=is_jtfst)
        if (discard_empty and s2.max() > 0) or not discard_empty:
            frames_s2.append((s2, filter_freqs, psi1_f_idx, s1_freq))

    if use_vmax:
        s2_max = max([s2.max().item() for s2, _, _, _ in frames_s2])
    else:
        s2_max = None

    frames = []
    for s2, filter_freqs, psi1_f_idx, s1_freq in tqdm(frames_s2):
        frame_save_path = os.path.join(save_dir, f"s2_{psi1_f_idx}.png")
        plot_scalogram(s2,
                       sr=sr,
                       y_coords=filter_freqs,
                       hop_len=hop_len,
                       vmax=s2_max,
                       title=f"{title_prefix} at {s1_freq:.2f} Hz",
                       save_path=frame_save_path,
                       x_label=x_label,
                       y_label=y_label)
        img = imageio.v2.imread(frame_save_path)
        frames.append(img)

    frames.reverse()
    save_path = os.path.join(save_dir, f"{save_name}.gif")
    imageio.mimsave(save_path, frames, fps=fps, loop=50000)


def extract_jtfst_scalogram(s_jtfst: Dict[Tuple[int, int, int], Tensor],
                            sr: float,
                            psi2_f_idx: int,
                            filters_fr_idx: int,
                            psi1_f: List[Dict[str, Any]],
                            pad: bool = True) -> (Tensor, List[float]):
    """
    Extracts a scalogram from a dictionary of scattering coefficients.

    Parameters:
        s_jtfst (Dict[Tuple[int, int, int], torch.Tensor]): A dictionary where the keys are tuples of indices and the values are scattering coefficients.
        sr: Sampling rate.
        psi2_f_idx (int): The index of the second-order scattering coefficient to extract.
        filters_fr_idx (int): The index of the filter to extract.
        psi1_f (List[Dict[str, Any]]): The list of first-order filters.
        pad (bool, optional): Whether to add a tensor of zeros when a coefficient is not found. Defaults to True.

    Returns:
        scalogram (torch.Tensor): The extracted scalogram.
        filter_freqs (List[float]): The frequencies of the first-order filters.
    """
    assert s_jtfst, "s_jtfst is empty"
    n_psi1_f = len(psi1_f)
    rows = []
    psi1_f_indices = []
    for curr_psi1_f_idx in range(n_psi1_f):
        key = (curr_psi1_f_idx, psi2_f_idx, filters_fr_idx)
        if key in s_jtfst:
            rows.append(s_jtfst[key])
            psi1_f_indices.append(curr_psi1_f_idx)
        else:
            if pad:
                rows.append(tr.zeros_like(list(s_jtfst.values())[0]))
                psi1_f_indices.append(curr_psi1_f_idx)

    filter_freqs = [psi1_f[idx]["xi"] * sr for idx in psi1_f_indices]

    assert rows, f"No scattering coefficients were found for psi2_f_idx of {psi2_f_idx} and filters_fr_idx of {filters_fr_idx}"
    scalogram = tr.stack(rows, dim=0)
    return scalogram, filter_freqs


def create_jtfst_gif(s_jtfst: Dict[Tuple[int, int, int], Tensor],
                     sr: float,
                     psi1_f: List[Dict[str, Any]],
                     psi2_f: List[Dict[str, Any]],
                     filters_fr: List[Dict[str, Any]],
                     save_dir: str,
                     save_name: str,
                     show_both_thetas: bool = True,
                     title_prefix: str = "JTFST",
                     min_psi2_f_freq_hz: float = 0,
                     max_psi2_f_freq_hz: float = np.inf,
                     min_filters_fr_freq_hz: float = 0,
                     max_filters_fr_freq_hz: float = np.inf,
                     hop_len: int = 1,
                     use_vmax: bool = False,
                     discard_empty: bool = True,
                     fps: int = 7) -> None:
    """
    Creates a gif of second-order scalograms.

    Parameters:
        s_jtfst (Dict[Tuple[int, int, int], torch.Tensor]): A dictionary where the keys are tuples of indices and the values are scattering coefficients.
        sr: Sampling rate.
        psi1_f (List[Dict[str, Any]]): The list of first-order filters.
        psi2_f (List[Dict[str, Any]]): The list of second-order filters.
        filters_fr (List[Dict[str, Any]]): The list of frequential filters.
        save_dir (str): The directory to save the gif.
        save_name (str): The name of the gif.
        show_both_thetas (bool, optional): Whether to show both positive and negative theta values. Defaults to True.
        title_prefix (str, optional): The prefix for the title of each frame. Defaults to "JTFST".
        min_psi2_f_freq_hz (float, optional): The minimum frequency of the second-order filter. Defaults to 0.
        max_psi2_f_freq_hz (float, optional): The maximum frequency of the second-order filter. Defaults to np.inf.
        min_filters_fr_freq_hz (float, optional): The minimum frequency of the filter. Defaults to 0.
        max_filters_fr_freq_hz (float, optional): The maximum frequency of the filter. Defaults to np.inf.
        hop_len (int, optional): The hop length for the time axis (or T). Defaults to 1.
        use_vmax (bool, optional): Whether to use a fixed maximum value for the colorbar. Defaults to False.
        discard_empty (bool, optional): Whether to discard frames with no scattering coefficients. Defaults to True.
        fps (int, optional): The frames per second for the gif. Defaults to 7.
    """
    frames_s2 = []

    filters_fr_indices_pos = []
    filters_fr_indices_neg = []
    for idx in range(len(filters_fr)):
        xi = filters_fr[idx]["xi"]
        if xi == 0:
            filters_fr_indices_pos.append(idx)
            filters_fr_indices_neg.append(None)
        elif xi > 0:
            filters_fr_indices_pos.append(idx)
            neg_idx = len(filters_fr) // 2 + idx
            assert filters_fr[neg_idx]["xi"] == -xi
            filters_fr_indices_neg.append(neg_idx)
    assert len(filters_fr_indices_pos) == len(filters_fr_indices_neg)

    for psi2_f_idx in range(len(psi2_f)):
        psi2_f_freq = psi2_f[psi2_f_idx]["xi"] * sr
        if psi2_f_freq < min_psi2_f_freq_hz or psi2_f_freq > max_psi2_f_freq_hz:
            continue
        for filters_fr_idx_pos, filters_fr_idx_neg in zip(filters_fr_indices_pos,
                                                          filters_fr_indices_neg):
            filters_fr_freq = filters_fr[filters_fr_idx_pos]["xi"] * sr
            # TODO(cm): check if 0 should be included
            if filters_fr_freq == 0 or filters_fr_freq < min_filters_fr_freq_hz or filters_fr_freq > max_filters_fr_freq_hz:
                continue
            s2, filter_freqs = extract_jtfst_scalogram(s_jtfst, sr, psi2_f_idx,
                                                       filters_fr_idx_pos, psi1_f,
                                                       pad=True)
            if (discard_empty and s2.max() > 0) or not discard_empty:
                s2_neg = None
                if show_both_thetas:
                    s2_neg, _ = extract_jtfst_scalogram(s_jtfst, sr, psi2_f_idx,
                                                        filters_fr_idx_neg, psi1_f,
                                                        pad=True)
                frames_s2.append((s2, filter_freqs, psi2_f_idx, psi2_f_freq,
                                  filters_fr_idx_pos, filters_fr_freq, s2_neg))

    if use_vmax:
        s2_max = max([s2.max().item() for s2, _, _, _, _, _, _ in frames_s2])
    else:
        s2_max = None

    frames = []
    for s2, filter_freqs, psi2_f_idx, psi2_f_freq, filters_fr_idx, filters_fr_freq, s2_neg in tqdm(
            frames_s2):
        frame_save_path = os.path.join(save_dir,
                                       f"jtfst_{psi2_f_idx}_{filters_fr_idx}_pos.png")
        plot_scalogram(s2,
                       sr=sr,
                       y_coords=filter_freqs,
                       hop_len=hop_len,
                       vmax=s2_max,
                       title=f"{title_prefix} at {psi2_f_freq:.2f} Hz, {filters_fr_freq:.2f} Hz, 1",
                       save_path=frame_save_path)
        img = imageio.v2.imread(frame_save_path)
        if show_both_thetas:
            if s2_neg is None:
                img_neg = np.zeros_like(img)
            else:
                frame_save_path = os.path.join(save_dir,
                                               f"jtfst_{psi2_f_idx}_{filters_fr_idx}_neg.png")
                plot_scalogram(s2_neg,
                               sr=sr,
                               y_coords=filter_freqs,
                               hop_len=hop_len,
                               vmax=s2_max,
                               title=f"{title_prefix} at {psi2_f_freq:.2f} Hz, {filters_fr_freq:.2f} Hz, -1",
                               save_path=frame_save_path)
                img_neg = imageio.v2.imread(frame_save_path)
            img = np.concatenate((img, img_neg), axis=1)

        frames.append(img)

    frames.reverse()
    save_path = os.path.join(save_dir, f"{save_name}.gif")
    imageio.mimsave(save_path, frames, fps=fps, loop=50000)
