import os, glob
from typing import Union, Callable

from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.io import wavfile

import torch
import torch.nn.functional as F

from s1dt.features import JTFS, Scat1d

from s1dt.core import FEATURES_TABLE


def load_audio_file(audio_file: str, preprocessors: Union[Callable] = []):
    """Load audio, applying a series of preprocessing transforms
    Args:
        audio_file: audio file path
        preprocessors: iterable of transforms to be composed
    """
    sr, audio = wavfile.read(audio_file)

    for processor in preprocessors:
        audio = processor(audio)

    return audio, sr



def load_numpy(file_path: str, preprocessors: Union[Callable] = [], **kwargs):
    npy = np.load(file_path)
    for processor in preprocessors:
        npy = processor(npy)
    return npy


def normalize_audio(audio: np.ndarray, eps: float = 1e-10):
    max_val = max(np.abs(audio).max(), eps)

    return audio / max_val

def torch_float32(x):
    """
    Convert a numpy array or torch tensor
    to a 32-bit torch float tensor
    """
    if isinstance(x, torch.FloatTensor):
        return x
    elif isinstance(x, torch.Tensor):
        return x.type(torch.FloatTensor)
    else:
        return torch.from_numpy(x).type(torch.FloatTensor)


def prepare_input_tensor(x: torch.Tensor, preprocessors: Union[Callable]):
    """Prepare data tensors for input to the network
    with a series of preprocessing functions
    1. Add the channel dimension
    2. convert to float32 tensor
    """
    for processor in preprocessors:
        x = processor(x)
    return x


def pad_or_trim_along_axis(arr: np.ndarray, output_length: int, axis=-1):
    if arr.shape[axis] < output_length:
        n_pad = output_length - arr.shape[axis]

        n_dims_end = len(arr.shape) - axis - 1 if axis >= 0 else 0
        n_dims_end *= n_dims_end > 0

        padding = [(0, 0)] * axis + [(0, n_pad)] + [(0, 0)] * n_dims_end

        return np.pad(arr, padding)
    else:
        return np.take_along_axis(arr, np.arange(0, output_length, 1), axis)


def make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            print("Failed to create directory %s" % dir_path)


def get_fname(fpath):
    return os.path.splitext(os.path.basename(fpath))[0]


class SOLExtractor:
    def __init__(
        self,
        sol_dir,
        output_dir,
        in_shape,
    ):
        self.sol_dir = sol_dir
        self.output_dir = output_dir
        self.in_shape = in_shape

        make_directory(self.output_dir)

        self.filelist = glob.glob(os.path.join(sol_dir, "*.wav"))

    def load_audio(self, filepath):
        audio, sr = load_audio_file(filepath, preprocessors=[normalize_audio])
        audio = pad_or_trim_along_axis(normalize_audio(audio), self.in_shape)
        audio = torch_float32(audio)
        return audio

    def save_stats(self):
        stats_path = os.path.join(self.output_dir, "stats")
        make_directory(stats_path)

        for stat in ["mu", "var", "mean"]:
            if hasattr(self, stat):
                np.save(os.path.join(stats_path, stat), getattr(self, stat))


class FeatureExtractorSOL(SOLExtractor):
    def __init__(
        self,
        sol_dir,
        device="cuda",
        batch_size=1,
        feature_id="jtfs",
        **feature_kwargs
    ):
        super().__init__(
            sol_dir,
            os.path.join(sol_dir, feature_id),
            in_shape=feature_kwargs["shape"][0],
        )
        self.feature = FEATURES_TABLE[feature_id](sr=44100, batch=batch_size, device=device, **feature_kwargs)
        self.device = device
        self.samples = []
        self.fnames = []

    def save_features(self):
        self.fnames = [os.path.join(self.output_dir, get_fname(filepath)) for filepath in tqdm(self.filelist)]
        
        # reshape to [batch * time, paths] and save the mean for subsequent pathwise mean-normalization: torch.log1p(s / (1e-3 * self.mean))
        self.mean = self.Sx.reshape((-1, self.Sx.shape[1])).mean(axis=0) 
        for i, fname in enumerate(self.fnames):
            np.save(fname, self.Sx[i])

    def extract(self):
        print("Loading audio ...")
        audio = torch.stack([self.load_audio(os.path.join(self.sol_dir, filepath)) for idx, filepath in enumerate(tqdm(self.filelist))])
        print("Extracting features ...")
        self.Sx = self.feature.compute_features(audio.to(self.device)).cpu().numpy()

def extract_features(
    sol_dir="/import/c4dm-datasets/SOL_0.9_HQ-PMT/",
    batch_size=16, 
    device="cuda"
):
    features = {
        "scat1d": {
            "shape": (2**16, ),
            "Q": (12, 2),
            "J": 12,
            "global_avg": False
        }, 
        "jtfs": {
            "shape": (2**16, ),
            "Q": (12, 2),
            "J": 12,
            "J_fr": 3,
            "Q_fr": 2,
            "F": 12,
            "T": 2**13,
            "global_avg": False
        }
    }
    for feature_id, feature_kwargs in features.items():
        extractor = FeatureExtractorSOL(sol_dir, device=device, batch_size=batch_size, feature_id=feature_id, **feature_kwargs)
        extractor.extract()
        extractor.save_features()
        extractor.save_stats()

if __name__ == "__main__":
    extract_features()