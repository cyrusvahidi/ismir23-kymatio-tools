class FeatureNotFoundError(Exception):
    """Exception raised when the given feature ID is not found."""

    def __init__(self, feature_id):
        self.feature_id = feature_id
        message = f"Feature with ID '{self.feature_id}' not found."
        super().__init__(message)


class FeaturesNotComputedError(Exception):
    """Exception raised when required features are not precomputed."""

    def __init__(
        self,
        message="Features must be extracted from the audio dataset prior to fitting the GEAR model.",
    ):
        self.message = message
        super().__init__(self.message)
