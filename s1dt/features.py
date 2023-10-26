

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
    

class MFCC:
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
    

class Scat1d:
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
    

class JTFS:
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
    

class OpenL3:
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
    

class YAMNet:
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