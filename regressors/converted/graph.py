import mont3 as plt
import numpy as np


def make_curve(n=100, std=0.5, return_true=False):
    np.random.seed(0)
    x = np.linspace(0, 6 * np.pi, n).reshape(-1, 1)
    true = np.sin(x)
    y = true + np.random.normal(scale=0.5, size=[n, 1])
    if return_true:
        return x, y, true
    return x, y


def make_curve_for_bayes():
    np.random.seed(0)

    x = np.hstack([np.random.normal(1.5, 0.7, 20), np.random.normal(11, 1, 40)])
    x = np.sort(x)
    y = np.sin(x) + np.random.normal(scale=0.2, size=x.size)

    true_x = np.linspace(0, 4 * np.pi, 100)
    return (x, y), (true_x, np.sin(true_x))


x, y = make_curve()

fig = plt.figure(figsize=[14, 5])
fig[:, :3].scatter(x, y, color="r", marker=".")
fig[:].set(xlabel="[]", ylabel="[]", ylim=[-2, 2], xlim=[0, 20])
fig[:].grid(True)

# LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

trans = PolynomialFeatures(degree=7)
poly = Pipeline([('poly', trans), ('linear', LinearRegression())])
poly.fit(x, y.ravel())
fig[0, 0].plot(x, poly.predict(x), lw=2.3, color="b")
fig[0, 0].set(title="Polynominal")

# SVR
from sklearn.svm import SVR

eps = 1
svr = SVR(kernel="rbf", C=1000, epsilon=eps)
svr.fit(x, y.ravel())

pred = svr.predict(x)
fig[1, 0].plot(x, pred, lw=2.3, color="b")
fig[1, 0].scatter(x[svr.support_], y[svr.support_], marker="*", color="g", s=60)
fig[1, 0].fill_between(x.ravel(), pred - eps, pred + eps, color="gray", alpha=0.3)
fig[1, 0].set(title="SVR")

# MLPRegressor
from sklearn.neural_network import MLPRegressor

# relu
mlp = MLPRegressor(hidden_layer_sizes=(150, ),
                   alpha=0.1,
                   solver="lbfgs",
                   max_iter=2000,
                   random_state=0)
mlp.fit(x, y.ravel())
fig[0, 1].plot(x, mlp.predict(x), lw=2.3, color="b")
fig[0, 1].set(title="NN with relu")

# sigmoid
mlp = MLPRegressor(hidden_layer_sizes=(50, ),
                   activation="tanh",
                   alpha=0.1,
                   solver="lbfgs",
                   max_iter=2000,
                   random_state=0)
mlp.fit(x, y.ravel())
fig[1, 1].plot(x, mlp.predict(x), lw=2.3, color="b")
fig[1, 1].set(title="NN with tanh")

# RandomForestRegressor
from matplotlib import cm
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(min_samples_split=11, n_estimators=10, random_state=0)
rf.fit(x, y.ravel())

estimators = np.array(rf.estimators_).ravel()
for i, est in enumerate(estimators):
    c = cm.jet(i / rf.n_estimators)
    fig[0, 2].step(x, est.predict(x), alpha=0.4, where="mid", color=c)
fig[0, 2].step(x, rf.predict(x), where="mid", lw=2.3, color="b")
fig[0, 2].set(title="RandomForest")

# GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor

grad = GradientBoostingRegressor(min_samples_split=11,
                                 n_estimators=50,
                                 random_state=0)
grad.fit(x, y.ravel())

estimators = np.array(grad.estimators_).ravel()
for i, (est, s_pred) in enumerate(zip(estimators, grad.staged_predict(x))):
    c = cm.jet(i / grad.n_estimators)
    fig[1, 2].step(x, est.predict(x), alpha=0.25, where="mid", color=c)
fig[1, 2].step(x, grad.predict(x), lw=2.3, where="mid", color="b")
fig[1, 2].set(title="GradientBoosting")

# BayesianRidge
test_x = x.reshape(-1)
(x, y), _ = make_curve_for_bayes()
fig[:, -1].scatter(x, y, color="r", marker=".")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.pipeline import Pipeline


class RBFTransfomer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.loc = np.arange(20).reshape(-1, 1)

    def fit(self, X, y):
        return self

    def transform(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return rbf(X, self.loc)

    def __repr__(self):
        return self.__class__.__name__ + "()"


rbfbayes = Pipeline([("rbf", RBFTransfomer()),
                     ("bayes", BayesianRidge(lambda_2=1))])
rbfbayes.fit(x, y)
ymean, ystd = rbfbayes.predict(test_x.reshape(-1, 1), return_std=True)
fig[0, 3].plot(test_x, ymean, lw=2.3, color="b")
fig[0, 3].fill_between(test_x, ymean - ystd, ymean + ystd, color="pink", alpha=0.5)
fig[0, 3].set(title="BayesianRidge")

# GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

gp = GaussianProcessRegressor(alpha=1)
gp.fit(x.reshape(-1, 1), y.ravel())
ymean, ystd = gp.predict(test_x.reshape(-1, 1), return_std=True)
fig[1, 3].plot(test_x, ymean, lw=2.3, color="b")
fig[1, 3].fill_between(test_x, ymean - ystd, ymean + ystd, color="pink", alpha=0.5)
fig[1, 3].set(title="GaussianProcess")

fig.show()

# 3d * 2
