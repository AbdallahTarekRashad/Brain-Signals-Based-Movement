from Strategy import Strategy
class LDA(Strategy):
    '''
    Parameters
    ----------
    solver : string, optional
        Solver to use, possible values:
          - 'svd': Singular value decomposition (default).
            Does not compute the covariance matrix, therefore this solver is
            recommended for data with a large number of features.
          - 'lsqr': Least squares solution, can be combined with shrinkage.
          - 'eigen': Eigenvalue decomposition, can be combined with shrinkage.

    shrinkage : string or float, optional
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

    priors : array, optional, shape (n_classes,)
        Class priors.

    n_components : int, optional
        Number of components (< n_classes - 1) for dimensionality reduction.

    store_covariance : bool, optional
        Additionally compute class covariance matrix (default False), used
        only in 'svd' solver.

        .. versionadded:: 0.17

    tol : float, optional, (default 1.0e-4)
        Threshold used for rank estimation in SVD solver.
    '''
    
    def __init__(self, solver='svd', shrinkage=None, priors=None,
                 n_components=None, store_covariance=False, tol=1e-4):
        self.type = "LDA"
        self.model = None
        
        self.solver           = solver
        self.shrinkage        = shrinkage
        self.priors           = priors
        self.n_components     = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol              = tol  # used only in svd solver

        

    def build(self, x, y):
        self.model = LinearDiscriminantAnalysis(self.solver, self.shrinkage, self.priors,
                 self.n_components, self.store_covariance, self.tol)

    

    def reshape(self, x, y):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

        return x, y

    def evaluate(self, x, y):
        x, y = self.reshape(x, y)
        return self.model.score(x, y)

    def fit(self, x, y):
        x, y = self.reshape(x, y)
        return self.model.fit(x,y)
      
    def predict(self, x,y=None):
        x, y = self.reshape(x, y)
        return self.model.predict(x)