from sklearn.manifold import Isomap

from .core import FEATURES_TABLE
from .plot import plot_isomap
from .exceptions import FeatureNotFoundError, FeaturesNotComputedError


class GEAR:
    """Generative Evaluate Audio Representations with respect to
    a dataset `x` of parametric synthesizer sounds with parameters `theta`.
    Provides a wrapper to extract acoustic features `X` from the audio dataset `x`
    given a `feature_id` corresponding to the id returned by the `get_id` method
    of a subclass of `s1dt.feature.AcousticFeature`.
    """

    def __init__(self, x, theta, feature_id, n_neighbors=40, n_components=3, **feature_kwargs):
        if feature_id not in FEATURES_TABLE:
            raise FeatureNotFoundError(feature_id)

        self.feature = FEATURES_TABLE[feature_id](**feature_kwargs)
        self.model = Isomap(n_components=n_components, n_neighbors=n_neighbors)

        self.x = x
        self.X = None
        self.theta = theta

    def fit_model(self):
        if self.X is None:
            raise FeaturesNotComputedError()
        Z = self.model.fit_transform(self.X)

    def compute_features(self):
        raise NotImplementedError

    def plot(self):
        plot_isomap(self.embedding_, self.theta)
