

class AcousticFeature:
    def __init__(
        self,
        sr=44100,
        batch=1
    ):
        self.sr = sr 
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    @classmethod
    def get_id(cls):
        raise NotImplementedError("This method must return a string identifier"
                                  " for the features")
    

class MFCC(AcousticFeature):
    def __init__(
        self,
        sr=44100,
        batch=1
    ):
        self.sr = sr 
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    @classmethod
    def get_id(cls):
        return "mfcc"
    

class Scat1d(AcousticFeature):
    def __init__(
        self,
        sr=44100,
        batch=1
    ):
        self.sr = sr 
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    @classmethod
    def get_id(cls):
        return "scat1d"
    

class JTFS(AcousticFeature):
    def __init__(
        self,
        sr=44100,
        batch=1
    ):
        self.sr = sr 
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    @classmethod
    def get_id(cls):
        return "jtfs"
    

class OpenL3(AcousticFeature):
    def __init__(
        self,
        sr=44100,
        batch=1
    ):
        self.sr = sr 
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    @classmethod
    def get_id(cls):
        return "openl3"
    

class YAMNet(AcousticFeature):
    def __init__(
        self,
        sr=44100,
        batch=1
    ):
        self.sr = sr 
        self.batch = batch

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    @classmethod
    def get_id(cls):
        return "yamnet"