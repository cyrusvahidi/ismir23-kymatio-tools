from sklearn.manifold import Isomap

from .core import FEATURES_TABLE
from .plot import plot_isomap
from .exceptions import FeatureNotFoundError, FeaturesNotComputedError


class GEAR:
    """
    The GEAR class is a wrapper for computing and evaluating acoustic features of a dataset of parametric synthesizer sounds.

    Attributes:
        x (ndarray): The audio dataset of parametric synthesizer sounds.
        theta (ndarray): The parameters of the parametric synthesizer sounds.
        feature_id (str): The string identifier for the acoustic features to compute.
        n_neighbors (int): The number of neighbors to consider in the Isomap algorithm.
        n_components (int): The number of dimensions to reduce the data to in the Isomap algorithm.

    Methods:
        fit_model(): Fits the Isomap model to the computed acoustic features.
        plot(): Plots the Isomap embedding of the audio dataset.

    Raises:
        FeatureNotFoundError: Raised if the specified feature_id is not found in the FEATURES_TABLE.
    """
    def __init__(self, x, theta, feature_id, n_neighbors=40, n_components=3, **feature_kwargs):
        """
        Initializes a new instance of the GEAR class.

        Args:
            x (ndarray): The audio dataset of parametric synthesizer sounds.
            theta (ndarray): The parameters of the parametric synthesizer sounds.
            feature_id (str): The string identifier for the acoustic features to compute.
            n_neighbors (int): The number of neighbors to consider in the Isomap algorithm.
            n_components (int): The number of dimensions to reduce the data to in the Isomap algorithm.
            **feature_kwargs: Additional keyword arguments to pass to the AcousticFeature subclass.

        Raises:
            FeatureNotFoundError: Raised if the specified feature_id is not found in the FEATURES_TABLE.
        """
        if feature_id not in FEATURES_TABLE:
            raise FeatureNotFoundError(feature_id)

        self.feature = FEATURES_TABLE[feature_id](**feature_kwargs)
        self.model = Isomap(n_components=n_components, n_neighbors=n_neighbors)

        self.x = x
        self.X = None
        self.theta = theta

    def fit_model(self):
        """
        Fits the Isomap model to the computed acoustic features.

        Raises:
            FeaturesNotComputedError: Raised if the acoustic features have not been computed yet.
        """
        if not self.features.computed():
            self.feature.compute_features()
        self.Z = self.model.fit_transform(self.X)

    def plot(self, labels=["$f_0$", "$f_m$", "$\gamma$"]):
        """
        Plots the Isomap embedding of the computed acoustic features.

        Args:
            labels (list): The list of labels to use for the plot.

        Raises:
            FeaturesNotComputedError: Raised if the acoustic features have not been computed yet.
        """
        plot_isomap(self.embedding_, self.theta.T)
