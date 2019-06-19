class Session:
    def __init__(self, data, preprocess, classifier):
        print('Start session')
        self.data = data
        self.preprocess = preprocess
        self.classifier = classifier

    def cross_val(self, n_splits):
        np.random.seed(43)
        scores_test = list()
        scores_train = list()
        kfold = StratifiedKFold(n_splits=n_splits, random_state = 0,shuffle=True)
        for train, test in kfold.split(self.data.data, self.data.labels):
            x_train = self.preprocess.fit_transform(self.data.data[train], self.data.labels[train])
            x_test = self.preprocess.transform(self.data.data[test])
            self.classifier.build(x_train, self.data.labels)
            self.classifier.fit(x_train, self.data.labels[train])
            scores_test.append(self.classifier.evaluate(x_test, self.data.labels[test]))
            scores_train.append(self.classifier.evaluate(x_train, self.data.labels[train]))
        scores_test  = np.array(scores_test)
        mean_test    = np.mean(scores_test, axis=0)
        std_test     = np.std(scores_test, axis=0)
        scores_train = np.array(scores_train)
        mean_train   = np.mean(scores_train, axis=0)
        std_train    = np.std(scores_train, axis=0)
        return mean_test ,std_test ,mean_train ,std_train

    def cross_val_sub(self, n_splits, subjects):
        mean_test   = list()
        std_test    = list()
        mean_train  = list()
        std_train   = list()
        for i in subjects:
            self.data.change_subject(i)
            m_test , s_test ,m_train,s_train=self.cross_val(n_splits)
            mean_test.append(m_test)
            std_test.append(s_test)
            mean_train.append(m_train)
            std_train.append(s_train)
        return mean_test ,std_test ,mean_train ,std_train
    
    def fitting(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data.data, self.data.labels, test_size=0.33, random_state=42)
        x_train = self.preprocess.fit_transform(X_train, y_train)
        x_test = self.preprocess.transform(X_test)
        self.classifier.build(x_train, y_train)
        self.classifier.fit(x_train, y_train)
        y_test_predict  = self.classifier.predict(x_test)
        y_train_predict = self.classifier.predict(x_train)
        con_test  = confusion_matrix(y_test,y_test_predict)
        con_train = confusion_matrix(y_train,y_train_pridict)
        return con_test,con_train
      
    def ROC(self,X_train, X_test, y_train, y_test):
        #X_train, X_test, y_train, y_test = train_test_split(self.data.data, self.data.labels, test_size=0.33, random_state=42)
        x_train = self.preprocess.fit_transform(X_train, y_train)
        x_test = self.preprocess.transform(X_test)
        self.classifier.build(x_train, y_train)
        self.classifier.fit(x_train, y_train)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
        y_test_predict  = self.classifier.model.model.predict_proba(x_test)
        y_train_predict = self.classifier.model.model.predict_proba(x_train)
        fpr, tpr, thresholds = roc_curve(y_test, y_test_predict[:, 1])
        return fpr, tpr, thresholds
      
    def ROC_Cross_Val(self,n_splits):
        fpr_list = []
        tpr_list = []
        thr_list = []
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 1
        mpl.style.use('seaborn-white')
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.ylim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        ax.set(ylabel='True Positive Rate',xlabel='False Positive Rate')
        ax.set_title('Receiver operating characteristic')
        
        kfold = StratifiedKFold(n_splits=n_splits, random_state = 0,shuffle=True)
        for train, test in kfold.split(self.data.data, self.data.labels):
            fpr, tpr, thresholds = self.ROC(self.data.data[train],self.data.data[test],
                                            self.data.labels[train],self.data.labels[test])
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            thr_list.append(thresholds)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d' % (i))
            i = i+1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)    
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr, mean_tpr, color='b',label='Mean ROC ',lw=2, alpha=.8)
        
        ax.legend(loc=4 ,fontsize=15, ncol =1)
        return fpr_list , tpr_list ,thr_list