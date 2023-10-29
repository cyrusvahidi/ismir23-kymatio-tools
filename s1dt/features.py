import math
from functools import partial
import numpy as np
import torch
import torchaudio.transforms as T 

from kymatio.torch import Scattering1D, TimeFrequencyScattering 
import openl3

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
        J (int): The maximum scale of the scattering transform.
        Q (int): The number of wavelets per octave.
        device (str): The device to use for computation. Either "cpu" or "cuda".
    Methods:
        compute_features(x): Computes the 1D scattering transform for the given audio signal(s).
        get_id(): Returns a string identifier for the 1D scattering transform.
    """
    def __init__(self, sr=44100, batch=1, device="cpu", shape=None, J=8, Q=1, T=None):
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
        
        self.transform = Scattering1D(shape=shape, T=T, Q=Q, J=int(np.log2(shape) - 1))

        self.to_device(device)

    def compute_features(self, x):
        """
        Computes the Scattering1D coefficients for the given audio signal(s).

        Args:
            x (ndarray): The audio signal(s) to compute the Scattering1D for.

        Returns:
            ndarray: The computed Scattering1D coefficients.

        Raises:
            ValueError: Raised if the input audio signal(s) have an invalid shape.
        """ 
        X = torch.cat(
            [
                self.transform(x[i * self.batch : (i + 1) * self.batch, :]).mean(dim=-1)
                for i in range(math.ceil(x.shape[0] / self.batch))
            ]
        )
        return X

    @classmethod
    def get_id(cls):
        return "scat1d"


class JTFS(AcousticFeature):
    def __init__(self, sr=44100, batch=1, device="cpu", shape=None, J=13, Q=(8, 1), T=None, Q_fr=2, J_fr=5, F=0):
        self.sr = sr
        self.batch = batch

        self.transform = TimeFrequencyScattering(shape=shape,
                               T=T,
                               Q=Q,
                               J=J, # int(np.log2(N) - 1),
                               Q_fr=Q_fr,
                               J_fr=J_fr,
                               F=F,
                               format="time")

    def compute_features(self, x):
        X = torch.cat(
            [
                self.transform(x[i * self.batch : (i + 1) * self.batch, :]).mean(dim=-1)
                for i in range(math.ceil(x.shape[0] / self.batch))
            ]
        )
        return X

    @classmethod
    def get_id(cls):
        return "jtfs"


class OpenL3(AcousticFeature):
    def __init__(self, sr=44100, batch=1, device="cpu"):
        self.sr = sr
        self.batch = batch

        self.transform = partial(openl3.get_audio_embedding, 
                                 sr=sr, batch_size=batch, content_type="music", embedding_size=6144, 
                                 input_repr="mel128",
                                 frontend="kapre" if device == "cuda" else "librosa")

    def compute_features(self, x):
        X, _ = self.transform(list(x.numpy()))
        # average at time dimension axis=1
        X = np.mean(np.array(X), axis=1)
        return X

    @classmethod
    def get_id(cls):
        return "openl3"


class YAMNet(AcousticFeature):
    def __init__(self, sr=44100, batch=1):
        self.sr = sr
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError(
            "This method must contain the actual " "implementation of the features"
        )

    @classmethod
    def get_id(cls):
        return "yamnet"
