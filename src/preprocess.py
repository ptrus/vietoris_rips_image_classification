from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from dataset import load_dataset
import numpy as np

class Scaler(BaseEstimator, TransformerMixin):
    """ Class used for scaling the dataset x.
        Removes the mean and than scales to unit variance.
        Implements inverse_transform method.
        
        Args:
        axis: if 1, standardize samples (rows)
              if 0, standardize features (columns)
    """    
    
    def __init__(self, axis=1):
        assert axis in [0,1]
        self.axis = axis
        
    def fit(self, x):
        if self.axis:
            self.means = np.mean(x, self.axis).reshape((len(x), 1))
            self.stds = np.std(x, self.axis).reshape((len(x), 1))
        else:
            self.means = np.mean(x, self.axis).reshape((1, len(x)))
            self.stds = np.std(x, self.axis).reshape((1, len(x)))
            
    def transform(self, x):
        return (x - self.means)/self.stds
        
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
        
    def inverse_transform(self, x):
        return (x*self.stds) + self.means

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

    def __init__(self, pca_n=None, axis=1):
        self.pca = PCA(pca_n)
        self.scaler = Scaler(axis)

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

    def inverse_transform(self, x):
        """ First undo the PCA transformation, than undo the scaling """
        return self.scaler.inverse_transform(self.pca.inverse_transform(x))



if __name__ == '__main__':

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
