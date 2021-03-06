from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import numpy as np

class NoScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x):
        return x

    def transform(self, x):
        return x

    def fit_transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

class Scaler(BaseEstimator, TransformerMixin):
    """ Class used for scaling the dataset x.
        Removes the mean and than scales to values from [-1, 1].
        Implements inverse_transform method.
        
        Args:
        axis: if 1, standardize samples (rows)
              if 0, standardize features (columns)
    """

    def __init__(self, axis=1):
        assert axis in [0,1]
        self.axis = axis

    def fit(self, x):
        shape = (len(x), 1) if self.axis == 1 else (1, len(x[0]))
        self.means = np.mean(x, self.axis).reshape(shape)
        self.maxs = np.max(np.abs(x - self.means), self.axis).reshape(shape)

    def transform(self, x):
        return np.nan_to_num((x - self.means)/self.maxs)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        return (x*self.maxs) + self.means

class Preprocess(BaseEstimator, TransformerMixin):
    """ Class used for preprocessing dataset x.
        Standard scaling and PCA transformation
        are performed.

        Args:
        pca_n: if integer, pca_n main components are selected
               if 0 < pca_n < 1, selects n components so that
               the fraction of explained variance is greater than pca_n
               
        Notes:
        Call fit than transform instead of fit_transform to get
        the variance explained printed.
    """

    def __init__(self, pca_n=None, scaler=Scaler(axis=1)):
        self.pca = PCA(pca_n)
        self.scaler = scaler

    def fit(self, x):
        self.pca.fit(self.scaler.fit_transform(x))
        print "Variance explained:", np.sum(self.pca.explained_variance_ratio_)

    def transform(self, x):
        return self.pca.transform(self.scaler.transform(x))

    def fit_transform(self, x):
        """ Override fit_transform to avoid calling transform
            twice on the standard scaler (in fit and transform)
        """
        return self.pca.fit_transform(self.scaler.fit_transform(x))

    def inverse_transform(self, x, only_pca=False):
        """ First undo the PCA transformation, then undo the scaling unless only_pca """
        if only_pca:
            return self.pca.inverse_transform(x)
        return self.scaler.inverse_transform(self.pca.inverse_transform(x))



if __name__ == '__main__':
    from dataset import load_dataset
    x,y = load_dataset(['../data/tea_cup', '../data/spoon'])

    # Test fit -> transform vs fit_transform
    p = Preprocess(0.7)
    p.fit(x)
    x2 = p.transform(x)
    x3 = p.fit_transform(x)
    assert np.all(np.isclose(x2, x3))

    # Test inverse transform.
    # With all PCA components retained, the inverse should be equal original.
    p2 = Preprocess()
    x4 = p2.fit_transform(x)
    x5 = p2.inverse_transform(x4)
    assert np.all(np.isclose(x5, x))
