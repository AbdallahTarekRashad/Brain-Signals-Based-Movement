from Strategy import Strategy
class KNN(Strategy):
    
    '''
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    weights : str or callable, optional (default = 'uniform')
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : string or callable, default 'minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.
    '''
    
    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None):
      
        self.type = "KNN"
        self.model = None
        
        self.n_neighbors   = n_neighbors
        self.weights       = weights
        self.algorithm     = algorithm
        self.leaf_size     = leaf_size
        self.p             = p
        self.metric        = metric
        self.metric_params = metric_params
        self.n_jobs        = n_jobs
        
    def build(self, x, y):
        self.model = KNeighborsClassifier(self.n_neighbors,
                 self.weights, self.algorithm, self.leaf_size,
                 self.p, self.metric, self.metric_params, self.n_jobs)

    

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

    
    def predict(self, x,y= None):
        x, y = self.reshape(x, y)
        return self.model.predict(x)