class Preprocess:
    def __init__(self):
        '''
        this class based on Pipeline structure in scikit learn
        '''
        self.preproces = list()
        self.pip = None

    def CSP(self, n_components=4, reg=None, log=None, cov_est="concat",
                 transform_into='average_power', norm_trace=False,
                 cov_method_params=None, rank=''):
        '''
        Parameters
        ----------
        n_components : int, defaults to 4
            The number of components to decompose M/EEG signals.
            This number should be set by cross-validation.
        reg : float | str | None (default None)
            If not None (same as ``'empirical'``, default), allow
            regularization for covariance estimation.
            If float, shrinkage is used (0 <= shrinkage <= 1).
            For str options, ``reg`` will be passed to ``method`` to
            :func:`mne.compute_covariance`.
        log : None | bool (default None)
            If transform_into == 'average_power' and log is None or True, then
            applies a log transform to standardize the features, else the features
            are z-scored. If transform_into == 'csp_space', then log must be None.
        cov_est : 'concat' | 'epoch', defaults to 'concat'
            If 'concat', covariance matrices are estimated on concatenated epochs
            for each class.
            If 'epoch', covariance matrices are estimated on each epoch separately
            and then averaged over each class.
        transform_into : {'average_power', 'csp_space'}
            If 'average_power' then self.transform will return the average power of
            each spatial filter. If 'csp_space' self.transform will return the data
            in CSP space. Defaults to 'average_power'.
        norm_trace : bool
            Normalize class covariance by its trace. Defaults to False. Trace
            normalization is a step of the original CSP algorithm [1]_ to eliminate
            magnitude variations in the EEG between individuals. It is not applied
            in more recent work [2]_, [3]_ and can have a negative impact on
            patterns ordering.
        cov_method_params : dict | None
            Parameters to pass to :func:`mne.compute_covariance`.

            .. versionadded:: 0.16
        rank : None | int | dict | 'full'
            See :func:`mne.compute_covariance`.

        '''
        
        csp = CSP(n_components, reg, log, cov_est,transform_into, norm_trace,cov_method_params, rank)
        
        self.preproces.append(('CSP', csp))

    def ICA(self, n_components=None, algorithm='parallel', whiten=True,
                 fun='logcosh', fun_args=None, max_iter=200, tol=1e-4,
                 w_init=None, random_state=None):
        
        '''
        Parameters
        ----------
        n_components : int, optional
            Number of components to use. If none is passed, all are used.

        algorithm : {'parallel', 'deflation'}
            Apply parallel or deflational algorithm for FastICA.

        whiten : boolean, optional
            If whiten is false, the data is already considered to be
            whitened, and no whitening is performed.

        fun : string or function, optional. Default: 'logcosh'
            The functional form of the G function used in the
            approximation to neg-entropy. Could be either 'logcosh', 'exp',
            or 'cube'.
            You can also provide your own function. It should return a tuple
            containing the value of the function, and of its derivative, in the
            point. Example:

            def my_g(x):
                return x ** 3, 3 * x ** 2

        fun_args : dictionary, optional
            Arguments to send to the functional form.
            If empty and if fun='logcosh', fun_args will take value
            {'alpha' : 1.0}.

        max_iter : int, optional
            Maximum number of iterations during fit.

        tol : float, optional
            Tolerance on update at each iteration.

        w_init : None of an (n_components, n_components) ndarray
            The mixing matrix to be used to initialize the algorithm.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        '''
        
        ica = UnsupervisedSpatialFilter(FastICA(n_components, algorithm, whiten,fun, fun_args, max_iter, tol,w_init, random_state)
                                        ,average=False)
        self.preproces.append(('ICA',ica))

    def PCA(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
      
        '''
        Parameters
        ----------
        n_components : int, float, None or string
            Number of components to keep.
            if n_components is not set all components are kept::

                n_components == min(n_samples, n_features)

            If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
            MLE is used to guess the dimension. Use of ``n_components == 'mle'``
            will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

            If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
            number of components such that the amount of variance that needs to be
            explained is greater than the percentage specified by n_components.

            If ``svd_solver == 'arpack'``, the number of components must be
            strictly less than the minimum of n_features and n_samples.

            Hence, the None case results in::

                n_components == min(n_samples, n_features) - 1

        copy : bool (default True)
            If False, data passed to fit are overwritten and running
            fit(X).transform(X) will not yield the expected results,
            use fit_transform(X) instead.

        whiten : bool, optional (default False)
            When True (False by default) the `components_` vectors are multiplied
            by the square root of n_samples and then divided by the singular values
            to ensure uncorrelated outputs with unit component-wise variances.

            Whitening will remove some information from the transformed signal
            (the relative variance scales of the components) but can sometime
            improve the predictive accuracy of the downstream estimators by
            making their data respect some hard-wired assumptions.

        svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
            auto :
                the solver is selected by a default policy based on `X.shape` and
                `n_components`: if the input data is larger than 500x500 and the
                number of components to extract is lower than 80% of the smallest
                dimension of the data, then the more efficient 'randomized'
                method is enabled. Otherwise the exact full SVD is computed and
                optionally truncated afterwards.
            full :
                run exact full SVD calling the standard LAPACK solver via
                `scipy.linalg.svd` and select the components by postprocessing
            arpack :
                run SVD truncated to n_components calling ARPACK solver via
                `scipy.sparse.linalg.svds`. It requires strictly
                0 < n_components < min(X.shape)
            randomized :
                run randomized SVD by the method of Halko et al.

            .. versionadded:: 0.18.0

        tol : float >= 0, optional (default .0)
            Tolerance for singular values computed by svd_solver == 'arpack'.

            .. versionadded:: 0.18.0

        iterated_power : int >= 0, or 'auto', (default 'auto')
            Number of iterations for the power method computed by
            svd_solver == 'randomized'.

            .. versionadded:: 0.18.0

        random_state : int, RandomState instance or None, optional (default None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.
        
        '''
        
        pca = UnsupervisedSpatialFilter( PCA(n_components, copy, whiten, svd_solver, tol, iterated_power,random_state) ,
                                        average=False)
        self.preproces.append(('PCA',pca))
        
    
    def _fft(self, x,fft_comp):
        x=rfft(x)
        return x[:, :, 0:fft_comp]

    def fft(self, fft_comp):
        #this is the parameter of function _fft
        dict={'fft_comp' : fft_comp}
        f_c = FunctionTransformer(func=self._fft,kw_args=dict,validate=False)
        self.preproces.append(('fft_com', f_c))

    def fit_transform(self, x, y=None):
        self.pip = Pipeline(self.preproces)
        return self.pip.fit_transform(x, y)

    def transform(self, x):
        return self.pip.transform(x)
