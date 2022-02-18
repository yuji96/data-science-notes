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


def predict(x, model):

    def relu(X):
        return np.maximum(X, 0, out=X)

    act = x.reshape(-1, 1)
    for w, w0 in zip(model.coefs_, model.intercepts_):
        act = relu(act) @ w + w0

    return act


# In[6]:


def count_params(model):
    param_num = 0
    for c in model.coefs_:
        m, n = c.shape
        param_num += m * n
    for i in model.intercepts_:
        n, *_ = i.shape
        param_num += n
    print(f"パラメータの個数: {param_num}")


# In[7]:

from sklearn.neural_network import MLPRegressor

mlp1 = MLPRegressor(hidden_layer_sizes=(300, ),
                    alpha=0.1,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=0)
mlp1.fit(x, y.ravel())

pred = predict(x, mlp1)
show(x, y, mlp1, pred)
count_params(mlp1)

# In[8]:

from sklearn.neural_network import MLPRegressor

mlp2 = MLPRegressor(hidden_layer_sizes=(50, 15),
                    alpha=0.05,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=0)
mlp2.fit(x, y.ravel())

pred = predict(x, mlp2)
show(x, y, mlp2, pred)
count_params(mlp2)

# In[10]:

from sklearn.neural_network import MLPRegressor

mlp3 = MLPRegressor(hidden_layer_sizes=(20, 20, 20),
                    alpha=1e-5,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=0)
mlp3.fit(x, y.ravel())

pred = predict(x, mlp3)
show(x, y, mlp3, pred)
count_params(mlp3)

# In[ ]:
