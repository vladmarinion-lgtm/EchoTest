from reservoir_computing.reservoir import Reservoir
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle

def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))


with open("TSLexchange.pkl","rb") as f:
    artifact = pickle.load(f)

n = 2110
k = 24
warmup = 100

times = np.array(artifact["times"][0:n+k])

targetcheck = artifact["target"][n:n + k]
target = np.transpose(np.array([artifact["target"][0:n + k]]))

vars = artifact["vars"][0:n+k,:]


pca = PCA(n_components=80)

Echo = Reservoir(
    n_internal_units= 400,
    spectral_radius= 0.8,
    leak = 0.9,
    noise_level = 0.001,
    connectivity= 0.5

)


states_tr = Echo.get_states(vars[None,:n,:], n_drop=warmup , bidir=False)

states_tr = pca.fit_transform(states_tr[0])

print(np.shape(states_tr))


Ridge = Ridge(alpha = 1.2,fit_intercept=True ,max_iter=None,tol=10**-10,solver="auto")
Ridge.fit(states_tr,target[warmup:n])


states_te = Echo.get_states(vars[None,n-warmup:n+k,:], n_drop=warmup, bidir=False)
states_te = pca.transform(states_te[0])



pred = Ridge.predict(states_te)

print(weighted_rmse_score(targetcheck,pred,0.1))


plt.plot(np.concatenate((target[:n,0],pred)))
plt.plot(target)

plt.show()