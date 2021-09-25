import numpy as np
import pandas as pd
import os

def loader_to_numpy(loader):
    """
    This function turns the training and testing data
    loader information into a numpy array
    """
    g_total, s_total = None, None
    for j, (g,s) in enumerate(loader):
        if g_total is None:
            g_total = g.numpy()
        else:
            g_total = np.concatenate([g_total, g.numpy()], axis=0)
        if s_total is None:
            s_total = s.numpy()
        else:
            s_total = np.concatenate([s_total, s.numpy()], axis=0)
    print('shape of g_total is', np.shape(g_total))
    return g_total, s_total

def MSE(pred, truth):
    return np.mean(np.square(pred - truth), axis=1)

class LR(object):
    """
    The linear regression model
    """
    def __init__(self, n_jobs=10):
        from sklearn.linear_model import LinearRegression
        self.n_jobs=n_jobs
        self.model = LinearRegression(n_jobs=10)
    def train_(self, train_loader, test_loader):
        train_x, train_y = loader_to_numpy(train_loader)
        test_x, test_y = loader_to_numpy(test_loader)
        self.model.fit(train_x, train_y)
        train_pred_y = self.model.predict(train_x)
        test_pred_y = self.model.predict(test_x)
        print('training MSE = {}, testing MSE = {}'.format(np.mean(MSE(train_pred_y, train_y)),
                                                           np.mean(MSE(test_pred_y, test_y))))
    def __call__(self, X):
        return self.model.predict(X)

class SVR(object):
    """
    The support vector regressor with linear kernel (as the number of points is too big)
    """
    def __init__(self, random_state=0, tol=1e-5):
        from sklearn.multioutput import MultiOutputRegressor            # Since sklearn SVR is only supporting 1d target, doing multioutput here
        from sklearn.svm import LinearSVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.random_state = random_state
        self.tol = tol
        self.model = MultiOutputRegressor(make_pipeline(StandardScaler(), 
                                   LinearSVR(random_state=random_state,
                                             tol=tol)))
    def train_(self, train_loader, test_loader):
        train_x, train_y = loader_to_numpy(train_loader)
        test_x, test_y = loader_to_numpy(test_loader)
        self.model.fit(train_x, train_y)
        train_pred_y = self.model.predict(train_x)
        test_pred_y = self.model.predict(test_x)
        print('training MSE = {}, testing MSE = {}'.format(np.mean(MSE(train_pred_y, train_y)),
                                                           np.mean(MSE(test_pred_y, test_y))))
    def __call__(self, X):
        return self.model.predict(X)



class RF(object):
    """
    The Random forest regressor 
    """
    def __init__(self, n_tree=100, max_depth=10, random_state=0, criterion='mse'):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(criterion=criterion,
                                           n_estimators=n_tree,
                                           max_depth=max_depth, 
                                           random_state=random_state)
    def train_(self, train_loader, test_loader):
        train_x, train_y = loader_to_numpy(train_loader)
        test_x, test_y = loader_to_numpy(test_loader)
        self.model.fit(train_x, train_y)
        train_pred_y = self.model.predict(train_x)
        test_pred_y = self.model.predict(test_x)
        print('training MSE = {}, testing MSE = {}'.format(np.mean(MSE(train_pred_y, train_y)),
                                                           np.mean(MSE(test_pred_y, test_y))))
    def __call__(self, X):
        return self.model.predict(X)

