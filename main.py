from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("TSLexchange.pkl","rb") as f:
    artifact = pickle.load(f)

#1100 merge incredibil
#1200 continua prea agresiv trend-ul
n = 1110
k = 25

times = np.array(artifact["times"][0:n+k])
#print(times[-1])
target = np.transpose(np.array([artifact["target"][0:n + k]]))
vars = artifact["vars"][0:n,:]

Echo = Reservoir(3500,lr = 1/n,sr = 1.5)
Echo.initialize(x = vars[0,:])
readout = Ridge(ridge = 1e-7)


states = Echo.run(vars)
print(np.shape(states))
readout.fit(states,target[:n,:])


testdata = artifact["vars"][n:n+k,:]
teststates = Echo.run(testdata)
print(np.shape(teststates))

pred = readout.run(teststates)
print(np.shape(pred))


plt.plot(np.concatenate((target[:n,0],pred[:,0])))
plt.plot(target)

plt.show()