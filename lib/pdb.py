import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys

h5fname = sys.argv[1]
if not h5fname:
    h5fname = 'temp.h5'
titleStr = sys.argv[2]
if not titleStr:
    title = ""
axesStr = sys.argv[3]

plt.title(titleStr)
dataset = h5py.File(h5fname, "r")

X = np.array(dataset["X"][:])
y = np.array(dataset["y"][:])
x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.array(dataset["Z"][:])
Z = Z.reshape(xx.shape)
if axesStr:
    axesStrArr = axesStr.split(",")
    axes = plt.gca()
    axes.set_xlim([float(axesStrArr[0]), float(axesStrArr[1])])
    axes.set_ylim([float(axesStrArr[2]), float(axesStrArr[3])])
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
plt.savefig('temp.png')