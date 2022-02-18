import matplotlib.pyplot as plt
import numpy as np

import japanize_matplotlib  # isort:skip # noqa
# In[11]:


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

# In[12]:


def show(x, y, model, pred):
    fig, (l, r) = plt.subplots(1, 2, figsize=[15, 4])
    l.scatter(x, y, marker=".", color="r")
    l.plot(x, model.predict(x))
    r.scatter(x, y, marker=".", color="r")
    r.plot(x, pred, color="k")
    return l, r


# In[13]:

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

trans = PolynomialFeatures(degree=7)
poly = Pipeline([('poly', trans), ('linear', LinearRegression())])

x, y = make_curve()
poly.fit(x, y.ravel())
Phi = trans.fit_transform(x)
W = poly.named_steps['linear'].coef_

show(x, y, model=poly, pred=W @ Phi.T)

# In[15]:

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import polynomial_kernel

kr = KernelRidge(kernel="poly", degree=7, coef0=x.mean(), alpha=100000)
kr.fit(x, y)
K = polynomial_kernel(x, x, degree=7, coef0=x.mean())

show(x, y, kr, pred=K @ kr.dual_coef_)

# In[16]:

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR

eps = 1
svr = SVR(kernel="rbf", C=1000, epsilon=eps)
x, y = make_curve()
svr.fit(x, y.ravel())

sv = x[svr.support_]
K = rbf_kernel(sv, x, gamma=1 / x.var())
pred = (svr.dual_coef_ @ K) + svr.intercept_
pred = pred.flatten()

l, _ = show(x, y, svr, pred=pred)
l.scatter(x[svr.support_], y[svr.support_], marker="*", color="b")
l.fill_between(x.ravel(), pred - eps, pred + eps, color="gray", alpha=0.3)
