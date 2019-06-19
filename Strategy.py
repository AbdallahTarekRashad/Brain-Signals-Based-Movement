from abc import ABC, abstractmethod

class Strategy(ABC):
    '''
    this is interface class. the classifiers classes inherit and develop it
    '''
    @abstractmethod
    def build(self,x, y):
        pass

    
    @abstractmethod
    def reshape(self, x, y):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass