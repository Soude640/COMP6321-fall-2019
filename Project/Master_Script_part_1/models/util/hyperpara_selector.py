from sklearn.model_selection import RandomizedSearchCV


class hyperpara_selector:
    hs = None

    def __init__(self, model, param_distribution, rs=0, ni=100, cv=5):
        try:
            self.hs = RandomizedSearchCV(model, param_distribution, random_state=rs, n_iter=ni, cv=cv)
        except:
            print("RandomizedSearchCV: input parameter may be wrong")

    def train(self, X_train=None, y_train=None):
        try:
            return self.hs.fit(X_train, y_train)
        except:
            print("RandomizedSearchCV: x_train or y_train may be wrong")
