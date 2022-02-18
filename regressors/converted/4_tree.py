import matplotlib.pyplot as plt
import numpy as np

import japanize_matplotlib  # noqa isort: skip

# In[3]:


def make_curve(n=100, std=0.5, return_true=False):
    np.random.seed(0)
    x = np.linspace(0, 6 * np.pi, n).reshape(-1, 1)
    true = np.sin(x)
    y = true + np.random.normal(scale=0.5, size=[n, 1])
    if return_true:
        return x, y, true
    return x, y


x, y, true = make_curve(return_true=True)
plt.plot(x, true, color="g")
plt.scatter(x, y, marker=".", color="r")
plt.grid(True)

# In[4]:


def show(x, y, model, pred):
    fig, (l, r) = plt.subplots(1, 2, figsize=[15, 4])
    l.scatter(x, y, marker=".", color="r")
    l.plot(x, model.predict(x))
    r.scatter(x, y, marker=".", color="r")
    r.plot(x, pred, color="k")
    return l, r


# In[5]:

from matplotlib import cm


def show_ensemble(model, cmap=cm.jet, alpha=0.3):
    fig, (l, r) = plt.subplots(1, 2, figsize=[15, 4])
    l.scatter(x, y, marker=".", color="r")
    r.scatter(x, y, marker=".", color="r")

    estimators = np.array(model.estimators_).ravel()
    for i, (est, s_pred) in enumerate(zip(estimators, model.staged_predict(x))):
        c = cmap(i / model.n_estimators)
        l.step(x, est.predict(x), alpha=alpha, where="mid", color=c)
        r.step(x, pred, alpha=alpha, where="mid", color=c)

    l.set(title="個々の弱学習木")
    r.set(title="更新中の強学習木")
    return l, r


# In[6]:

from sklearn.tree import DecisionTreeRegressor

x, y = make_curve()

tree = DecisionTreeRegressor(min_samples_split=11, random_state=0)
tree.fit(x, y.ravel())

plt.step(x, tree.predict(x), where="mid")
plt.scatter(x, y, marker=".", color="r")

# In[7]:

from sklearn.ensemble import RandomForestRegressor

x, y = make_curve()

rf = RandomForestRegressor(min_samples_split=11, n_estimators=10, random_state=0)
rf.fit(x, y.ravel())

fig, (l, r) = plt.subplots(1, 2, figsize=[15, 4])
l.scatter(x, y, marker=".", color="r")
r.scatter(x, y, marker=".", color="r")

estimators = np.array(rf.estimators_).ravel()
for i, est in enumerate(estimators):
    l.step(x, est.predict(x), alpha=0.3, where="mid", color="k")
r.step(x, rf.predict(x), where="mid")

l.set(title="個々の弱学習木")
r.set(title="平均した強学習木")

# In[8]:

from matplotlib import cm
from sklearn.ensemble import GradientBoostingRegressor

x, y = make_curve()

grad = GradientBoostingRegressor(min_samples_split=11,
                                 n_estimators=50,
                                 random_state=0)
grad.fit(x, y.ravel())

fig, (l, r) = plt.subplots(1, 2, figsize=[15, 4])
l.scatter(x, y, marker=".", color="r")
r.scatter(x, y, marker=".", color="r")

estimators = np.array(grad.estimators_).ravel()
for i, (est, s_pred) in enumerate(zip(estimators, grad.staged_predict(x))):
    c = cm.jet(i / grad.n_estimators)
    l.step(x, est.predict(x), alpha=0.3, where="mid", color=c)
    r.step(x, s_pred, alpha=0.3, where="mid", color=c)

l.set(title="個々の弱学習木")
r.set(title="更新中の強学習木")

# In[9]:

import sklearn.experimental.enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

x, y = make_curve()

hist = HistGradientBoostingRegressor(max_iter=50,
                                     min_samples_leaf=10,
                                     random_state=0)
hist.fit(x, y.ravel())

for i, pred in enumerate(hist.staged_predict(x)):
    plt.step(x,
             pred,
             alpha=0.3,
             where="mid",
             label=i,
             color=cm.jet(i / hist.n_iter_))
plt.scatter(x, y, marker=".", color="r")

# In[ ]:
