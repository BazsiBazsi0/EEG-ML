from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import numpy as np
from sklearn.preprocessing import LabelEncoder


class CSPTransformer:
    """
    CSPTransformer class.
    The following is needed to run this:
    X in the shape of (n_trials, n_channels, n_samples)
    """

    def __init__(
        self, n_components=4, reg=None, log=True, norm_trace=False, metric="riemann"
    ):
        self.csp = CSP(
            n_components=n_components, reg=reg, log=log, norm_trace=norm_trace
        )
        self.cov = Covariances(stimator="lwf")
        self.ts = TangentSpace(metric=metric)

    def fit_transform(self, X, y=None):
        # Convert one-hot encoded labels to integer labels
        # Convert string-encoded labels to integer labels
        label_encoder = LabelEncoder()
        y_int = label_encoder.fit_transform(y)

        X = X.astype(np.float64)
        # Apply CSP
        csp_data = self.csp.fit_transform(X, y_int)

        # Reshape CSP transformed data to have an extra dimension
        csp_data = csp_data[..., np.newaxis]

        # Compute covariance matrices from CSP transformed data
        cov_data = self.cov.transform(csp_data)

        # Map CSP components to tangent space
        ts_data = self.ts.fit_transform(cov_data)

        return ts_data

    @staticmethod
    def simple_csp(X, y, n_components=4):
        csp = CSP(n_components=n_components)
        csp_data = csp.fit_transform(X, y)
        return csp_data
