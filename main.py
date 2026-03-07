from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("TSLexchange.pkl","rb") as f:
    artifact = pickle.load(f)

n = 2010
k = 25

times = np.array(artifact["times"][0:n+k])
#print(times[-1])
target = np.transpose(np.array([artifact["target"][0:n + k]]))
vars = artifact["vars"][0:n,:]


Echo = Reservoir(600,lr = 0.7,sr = 0.99)
Echo.initialize(x = vars[0,:])
readout = Ridge(ridge = 1e-7)

Net = Echo >> readout

Net.fit(vars,target[:n,:],warmup = 200)


testdata = artifact["vars"][n:n+k,:]
pred = Net.run(testdata)



plt.plot(np.concatenate((target[:n,0],pred[:,0])))
plt.plot(target)

plt.show()