import math
import torch
import torchaudio.transforms as T 


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
        self.computed = False

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

    def __init__(self, sr=44100, batch=0, n_mfcc=40, log_mels=True, device="cpu"):
        """
        Initializes a new instance of the MFCC class.

        Args:
            sr (int): The sample rate of the audio signal.
            batch (int): The number of audio signals to process at once.
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
        self.computed = True
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
    def __init__(self, sr=44100, batch=1):
        self.sr = sr
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError(
            "This method must contain the actual " "implementation of the features"
        )

    @classmethod
    def get_id(cls):
        return "scat1d"


class JTFS(AcousticFeature):
    def __init__(self, sr=44100, batch=1):
        self.sr = sr
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError(
            "This method must contain the actual " "implementation of the features"
        )

    @classmethod
    def get_id(cls):
        return "jtfs"


class OpenL3(AcousticFeature):
    def __init__(self, sr=44100, batch=1):
        self.sr = sr
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError(
            "This method must contain the actual " "implementation of the features"
        )

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
