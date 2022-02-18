import matplotlib.pyplot as plt
import numpy as np

import japanize_matplotlib  # noqa isort: skip

# In[3]:


def make_curve():
    np.random.seed(0)

    x = np.hstack([np.random.normal(1.5, 0.7, 20), np.random.normal(11, 1, 40)])
    x = np.sort(x)
    y = np.sin(x) + np.random.normal(scale=0.2, size=x.size)

    true_x = np.linspace(0, 4 * np.pi, 100)
    return (x, y), (true_x, np.sin(true_x))


sample, true = make_curve()
plt.plot(*true, color="g")
plt.scatter(*sample, marker=".", color="r")
plt.grid(True)

# In[4]:


def show(x, y, model, test_x, pred):
    fig, (l, r) = plt.subplots(1, 2, figsize=[15, 4])

    for ax in [l, r]:
        ax.scatter(x, y, marker=".", color="r")

    ymean, ystd = model.predict(test_x.reshape(-1, 1), return_std=True)
    l.plot(test_x, ymean)
    l.fill_between(test_x, ymean - ystd, ymean + ystd, color="pink", alpha=0.5)
    r.plot(test_x, pred)
    r.fill_between(test_x, pred - ystd, pred + ystd, color="pink", alpha=0.5)
    return l, r


# In[5]:

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.pipeline import Pipeline


class RBFTransfomer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.loc = np.arange(15).reshape(-1, 1)

    def fit(self, X, y):
        return self

    def transform(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return rbf(X, self.loc)

    def __repr__(self):
        return self.__class__.__name__ + "()"


rbfbayes = Pipeline([("rbf", RBFTransfomer()),
                     ("bayes", BayesianRidge(lambda_2=10))])

# In[6]:

(x, y), (test_x, _) = make_curve()
rbfbayes.fit(x, y)

K = rbfbayes["rbf"].transform(test_x)
pred = rbfbayes["bayes"].coef_ @ K.T + rbfbayes["bayes"].intercept_

show(x, y, rbfbayes, test_x, pred)

# In[7]:

from scipy.stats import norm

ymean, ystd = rbfbayes.predict(test_x, return_std=True)

XX, YY = np.meshgrid(test_x, np.linspace(-3, 3, 100))
Z = norm.pdf(x=np.linspace(-3, 3, 100).reshape(-1, 1), loc=ymean, scale=ystd)
fig, ax = plt.subplots(subplot_kw={"projection": '3d'}, figsize=[10, 10])
ax.plot_surface(XX, YY, Z, linewidth=0, cmap="coolwarm")
ax.view_init(elev=50, azim=230)

# In[8]:

from sklearn.gaussian_process import GaussianProcessRegressor

(x, y), (test_x, _) = make_curve()

gp = GaussianProcessRegressor(alpha=1)
gp.fit(x.reshape(-1, 1), y.ravel())

K = gp.kernel_(test_x.reshape(-1, 1), gp.X_train_)
pred = K @ gp.alpha_

show(x, y, gp, test_x, pred)

# In[9]:

from scipy.stats import norm

XX, YY = np.meshgrid(test_x, np.linspace(-3, 3, 100))
Z = norm.pdf(x=np.linspace(-3, 3, 100).reshape(-1, 1), loc=ymean, scale=ystd)

fig, ax = plt.subplots(subplot_kw={"projection": '3d'}, figsize=[10, 10])
ax.plot_surface(XX, YY, Z, linewidth=0, cmap="coolwarm")
ax.view_init(elev=70, azim=190)

# In[ ]:
