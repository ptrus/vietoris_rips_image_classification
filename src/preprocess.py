from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataset import load_dataset

class Preprocess(BaseEstimator, TransformerMixin):
    """ Class used for preprocessing dataset x.
        Standard scaling and PCA are performed.

        Args:
        pca_n: if integer, pca_n main components are selected
               if 0 < pca_n < 1, selects n components so that
               the fraction of explained variance is greater than pca_n
    """

    def __init__(self, pca_n=None):
        self.ss = StandardScaler()
        self.pca = PCA(pca_n)

    def fit(self, x):
        self.pca.fit(self.ss.fit_transform(x))

    def transform(self, x):
        return self.pca.transform(self.ss.transform(x))

    def fit_transform(self, x):
        """ Override fit_transform to avoid calling transform
            twice on the standard scaler (in fit and transform)
        """
        return self.pca.fit_transform(self.ss.fit_transform(x))

    def inverse_transform(self, x):
        """ First undo the PCA transformation, than undo the scaling """
        return self.ss.inverse_transform(self.pca.inverse_transform(x))



if __name__ == '__main__':
    import numpy as np

    x,y = load_dataset(['../data/tea_cup', '../data/spoon'])

    # Test fit -> transform and fit_transform
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
