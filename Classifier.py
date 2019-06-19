class Classifier:
    def __init__(self, model):
        self.model = model

    def build(self, x, y):
        self.model.build(x, y)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        return score
