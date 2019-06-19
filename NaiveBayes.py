from Strategy import Strategy

class NaiveBayes(Strategy):
    '''
    Parameters
    ----------
    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    var_smoothing : float, optional (default=1e-9)
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    '''
    def __init__(self, priors=None, var_smoothing=1e-09):
        self.priors=priors
        self.smoothing=var_smoothing

    def build(self, x, y):
        self.model = GaussianNB(self.priors, self.smoothing)

    def reshape(self, x, y):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        return x, y

    def evaluate(self, x, y):
        x, y = self.reshape(x, y)
        return self.model.score(x, y)

    def fit(self, x, y):
        x, y = self.reshape(x, y)
        return self.model.fit(x, y)
    def predict(self, x,y=None):
        x, y = self.reshape(x, y)
        return self.model.predict(x)