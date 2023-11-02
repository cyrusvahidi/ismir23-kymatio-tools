import math
from functools import partial
import numpy as np
import torch
import torchaudio.transforms as T
import tensorflow_hub as hub
from tqdm import tqdm

from kymatio.torch import Scattering1D, TimeFrequencyScattering
import openl3

import nnAudio.features as nnFeatures


class AcousticFeature:
    def __init__(self, sr=44100, batch=1):
        """
        The AcousticFeature class is an abstract base class that defines the interface for computing audio features.
        It cannot be instantiated directly and must be subclassed to provide a concrete implementation of the features.

        Attributes:
            sr (int): The sample rate of the audio signal.
            batch (int): The number of audio signals to process at once.

        Methods:
            compute_features(x): This method must be implemented in the subclass and should compute the features for the given audio signal(s).
            get_id(): This method must be implemented in the subclass and should return a string identifier for the features.
            computed: This property returns a boolean indicating whether the features have been computed or not.
        """
        self.sr = sr
        self.batch = batch

    def compute_features(self, x):
        """
        This method must be implemented in the subclass and should compute the features for the given audio signal(s).

        Args:
            x (ndarray): The audio signal(s) to compute the features for.

        Raises:
            NotImplementedError: This method must be implemented in the subclass.
        """
        raise NotImplementedError(
            "This method must contain the actual " "implementation of the features"
        )

    @classmethod
    def get_id(cls):
        """
        This method must be implemented in the subclass and should return a string identifier for the features.

        Raises:
            NotImplementedError: This method must be implemented in the subclass.
        """
        raise NotImplementedError(
            "This method must return a string identifier" " for the features"
        )

    def to_device(self, device="cpu"):
        self.transform = (
            self.transform.cuda() if device == "cuda" else self.transform.cpu()
        )


class MFCC(AcousticFeature):
    """
    The MFCC class is a subclass of the AcousticFeature class that computes the Mel-frequency cepstral coefficients (MFCCs) of an audio signal.

    Attributes:
        sr (int): The sample rate of the audio signal.
        batch (int): The number of audio signals to process at once.
        n_mfcc (int): The number of MFCCs to compute.
        log_mels (bool): Whether to compute the log-mel spectrogram.

    Methods:
        compute_features(x): Computes the MFCCs for the given audio signal(s).
        get_id(): Returns a string identifier for the MFCCs.

    Raises:
        ImportError: Raised if the librosa library is not installed.
    """

    def __init__(self, sr=44100, batch=0, device="cpu", n_mfcc=40, log_mels=True):
        """
        Initializes a new instance of the MFCC class.

        Args:
            sr (int): The sample rate of the audio signal.
            batch (int): The number of audio signals to process at once.
            device (str): The device to use for computation. Either "cpu" or "cuda".
            n_mfcc (int): The number of MFCCs to compute.
            log_mels (bool): Whether to compute the log-mel spectrogram.
        Raises:
            ImportError: Raised if the librosa library is not installed.
        """
        super().__init__(sr=sr, batch=batch)

        self.transform = T.MFCC(sample_rate=sr, n_mfcc=n_mfcc, log_mels=log_mels)

        self.to_device(device)

    def compute_features(self, x):
        """
        Computes the MFCCs for the given audio signal(s).

        Args:
            x (ndarray): The audio signal(s) to compute the MFCCs for.

        Returns:
            ndarray: The computed MFCCs.

        Raises:
            ValueError: Raised if the input audio signal(s) have an invalid shape.
        """
        if self.batch:
            X = torch.cat(
                [
                    self.transform(x[i * self.batch : (i + 1) * self.batch, :]).mean(
                        dim=-1
                    )
                    for i in range(math.ceil(x.shape[0] / self.batch))
                ]
            )
        else:
            X = self.transform(x).mean(dim=-1)
        return X

    @classmethod
    def get_id(cls):
        """
        Returns a string identifier for the MFCCs.

        Returns:
            str: The string identifier for the MFCCs.
        """
        return "mfcc"


class Scat1d(AcousticFeature):
    """
    The Scat1d class is a subclass of the AcousticFeature class that computes the 1D scattering transform of an audio signal.

    Attributes:
        sr (int): The sample rate of the audio signal.
        batch (int): The number of audio signals to process at once.
        device (str): The device to use for computation. Either "cpu" or "cuda".
        J (int): The maximum scale of the scattering transform.
        Q (int): The number of wavelets per octave.
        T (int or None): The size of the temporal window used for the scattering transform. If None, the size is automatically determined based on the signal length.
        shape (int or None): The size of the input signal. If None, the size is automatically determined based on the shape of the input tensor.

    Methods:
        __init__(sr=44100, batch=1, device="cpu", shape=None, J=8, Q=1, T=None): Initializes a new instance of the Scat1d class.
        compute_features(x): Computes the 1D scattering transform for the given audio signal(s).
        get_id(): Returns a string identifier for the 1D scattering transform.
    """

    def __init__(
        self,
        sr=44100,
        batch=1,
        device="cpu",
        shape=None,
        J=8,
        Q=(1, 1),
        T=None,
        global_avg=True,
    ):
        """
        Initializes a new instance of the Scat1d class.

        Args:
            sr (int): The sample rate of the audio signal.
            batch (int): The number of audio signals to process at once.
            device (str): The device to use for computation. Either "cpu" or "cuda".
            shape (int or None): The size of the input signal. If None, the size is automatically determined based on the shape of the input tensor.
            J (int): The maximum scale of the scattering transform.
            Q (int): The number of wavelets per octave.
            T (int or None): The size of the temporal window used for the scattering transform. If None, the size is automatically determined based on the signal length.
        """
        super().__init__(sr=sr, batch=batch)
        self.sr = sr
        self.batch = batch

        self.transform = Scattering1D(shape=shape, T=T, Q=Q, J=J)

        self.to_device(device)

        self.global_avg = global_avg

    def compute_features(self, x):
        """
        Computes the 1D scattering transform for the given audio signal(s).

        Args:
            x (ndarray): The audio signal(s) to compute the 1D scattering transform for.

        Returns:
            ndarray: The computed 1D scattering transform coefficients.

        Raises:
            ValueError: Raised if the input audio signal(s) have an invalid shape.
        """
        X = torch.cat(
            [
                self.transform(x[i * self.batch : (i + 1) * self.batch, :])
                for i in tqdm(range(math.ceil(x.shape[0] / self.batch)))
            ]
        )
        if self.global_avg:
            X = X.mean(dim=-1)
        self.mu = X.mean(dim=0)
        self.median = X.median(dim=0)[0]
        return X

    @classmethod
    def get_id(cls):
        """
        Returns a string identifier for the 1D scattering transform.

        Returns:
            str: The string identifier for the 1D scattering transform.
        """
        return "scat1d"


class JTFS(AcousticFeature):
    """
    The JTFS class is a subclass of the AcousticFeature class that computes the joint time-frequency scattering transform of an audio signal.

    Attributes:
        sr (int): The sample rate of the audio signal.
        batch (int): The number of audio signals to process at once.
        device (str): The device to use for computation. Either "cpu" or "cuda".
        J (int): The maximum scale of the scattering transform.
        Q (tuple): The number of wavelets per octave for the first and second order scattering coefficients.
        T (int or None): The size of the temporal window used for the scattering transform. If None, the size is automatically determined based on the signal length.
        Q_fr (int): The number of wavelets per octave for the frequency scattering coefficients.
        J_fr (int): The maximum scale of the frequency scattering transform.
        F (int): The number of frequency bins to use for the frequency scattering transform. If 0, the number of frequency bins is automatically determined based on the signal length.

    Methods:
        __init__(sr=44100, batch=1, device="cpu", shape=None, J=13, Q=(8, 1), T=None, Q_fr=2, J_fr=5, F=0): Initializes a new instance of the JTFS class.
        compute_features(x): Computes the joint time-frequency scattering transform for the given audio signal(s).
        get_id(): Returns a string identifier for the joint time-frequency scattering transform.
    """

    def __init__(
        self,
        sr=44100,
        batch=1,
        device="cpu",
        shape=None,
        J=13,
        Q=(8, 1),
        T=None,
        Q_fr=2,
        J_fr=5,
        F=0,
        global_avg=True,
    ):
        """
        Initializes a new instance of the JTFS class.

        Args:
            sr (int): The sample rate of the audio signal.
            batch (int): The number of audio signals to process at once.
            device (str): The device to use for computation. Either "cpu" or "cuda".
            shape (int or None): The size of the input signal. If None, the size is automatically determined based on the shape of the input tensor.
            J (int): The maximum scale of the scattering transform.
            Q (tuple): The number of wavelets per octave for the first and second order scattering coefficients.
            T (int or None): support of temporal averaging lowpass filter \phi_T
            Q_fr (int): The number of wavelets per octave for the frequency scattering coefficients.
            J_fr (int): The maximum scale of the frequency scattering transform.
            F (int): support of frequential averaging lowpass filter  \phi_F
        """
        self.sr = sr
        self.batch = batch

        self.transform = TimeFrequencyScattering(
            J=J, J_fr=J_fr, shape=shape, Q=Q, T=T, Q_fr=Q_fr, F=F, format="time"
        )
        self.to_device(device)

        self.global_avg = global_avg

    def compute_features(self, x):
        """
        Computes the joint time-frequency scattering transform for the given audio signal(s).

        Args:
            x (ndarray): The audio signal(s) to compute the joint time-frequency scattering transform for.

        Returns:
            ndarray: The computed joint time-frequency scattering transform coefficients.

        Raises:
            ValueError: Raised if the input audio signal(s) have an invalid shape.
        """
        X = torch.cat(
            [
                self.transform(x[i * self.batch : (i + 1) * self.batch, :])
                for i in tqdm(range(math.ceil(x.shape[0] / self.batch)))
            ]
        )
        if self.global_avg:
            X = X.mean(dim=-1)
        self.mu = X.mean(dim=0)
        self.median = X.median(dim=0)[0]
        return X

    @classmethod
    def get_id(cls):
        """
        Returns a string identifier for the joint time-frequency scattering transform.

        Returns:
            str: The string identifier for the joint time-frequency scattering transform.
        """
        return "jtfs"


class OpenL3(AcousticFeature):
    def __init__(self, sr=44100, batch=1, device="cpu", embedding_size=6144):
        self.sr = sr
        self.batch = batch
        self.transform = partial(
            openl3.get_audio_embedding,
            sr=sr,
            content_type="music",
            batch_size=batch,
            embedding_size=embedding_size,
            input_repr="mel128",
            frontend="kapre" if device == "cuda" else "librosa",
        )

    def compute_features(self, x):
        X = torch.cat(
            [
                torch.tensor(
                    self.transform(
                        list(x[i * self.batch : (i + 1) * self.batch, None, :].numpy())
                    )[0]
                ).mean(axis=1)
                for i in tqdm(range(math.ceil(x.shape[0] / self.batch)))
            ]
        )
        return X

    @classmethod
    def get_id(cls):
        return "openl3"


class YAMNet(AcousticFeature):
    def __init__(self, sr=44100, batch=1):
        self.sr = sr
        self.batch = batch

        self.transform = hub.load("https://tfhub.dev/google/yamnet/1")

    def compute_features(self, x):
        X = np.array(
            [
                self.transform(x[i * self.batch : (i + 1) * self.batch, :])
                for i in tqdm(range(math.ceil(x.shape[0] / self.batch)))
            ]
        )
        return X

    @classmethod
    def get_id(cls):
        return "yamnet"


class CQT(AcousticFeature):
    """
    The Scat1d class is a subclass of the AcousticFeature class that computes the 1D scattering transform of an audio signal.

    Attributes:
        sr (int): The sample rate of the audio signal.
        batch (int): The number of audio signals to process at once.
        device (str): The device to use for computation. Either "cpu" or "cuda".
        J (int): The maximum scale of the scattering transform.
        Q (int): The number of wavelets per octave.
        T (int or None): The size of the temporal window used for the scattering transform. If None, the size is automatically determined based on the signal length.
        shape (int or None): The size of the input signal. If None, the size is automatically determined based on the shape of the input tensor.

    Methods:
        __init__(sr=44100, batch=1, device="cpu", shape=None, J=8, Q=1, T=None): Initializes a new instance of the Scat1d class.
        compute_features(x): Computes the 1D scattering transform for the given audio signal(s).
        get_id(): Returns a string identifier for the 1D scattering transform.
    """

    def __init__(
        self,
        sr=44100,
        batch=1,
        device="cpu",
        n_bins=144,
        bins_per_octave=12,
        hop_length=512,
        global_avg=True,
    ):
        """
        Initializes a new instance of the Scat1d class.

        Args:
            sr (int): The sample rate of the audio signal.
            batch (int): The number of audio signals to process at once.
            device (str): The device to use for computation. Either "cpu" or "cuda".
        """
        super().__init__(sr=sr, batch=batch)
        self.sr = sr
        self.batch = batch

        self.transform = nnFeatures.CQT(
            sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length
        )

        self.to_device(device)

        self.global_avg = global_avg

    def compute_features(self, x):
        """
        Computes the CQT for the given audio signal(s).

        Args:
            x (ndarray): The audio signal(s) to compute the CQT for.

        Returns:
            ndarray: The computed CQT coefficients.

        Raises:
            ValueError: Raised if the input audio signal(s) have an invalid shape.
        """
        X = torch.cat(
            [
                self.transform(x[i * self.batch : (i + 1) * self.batch, :])
                for i in tqdm(range(math.ceil(x.shape[0] / self.batch)))
            ]
        )
        if self.global_avg:
            X = X.mean(dim=-1)
        self.mu = X.mean(dim=0)
        self.median = X.median(dim=0)[0]
        return X

    @classmethod
    def get_id(cls):
        """
        Returns a string identifier for the CQT

        Returns:
            str: The string identifier for the CQT
        """
        return "cqt"
