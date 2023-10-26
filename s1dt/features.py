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

    @property
    def computed(self):
        """
        This property returns a boolean indicating whether the features have been computed or not.
        """
        return self.__computed


class MFCC(AcousticFeature):
    def __init__(self, sr=44100, batch=1):
        self.sr = sr
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError(
            "This method must contain the actual " "implementation of the features"
        )

    @classmethod
    def get_id(cls):
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
