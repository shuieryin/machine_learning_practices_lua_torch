import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys

h5fname = sys.argv[1]
if not h5fname:
    h5fname = 'temp.h5'

dataset = h5py.File(h5fname, "r")
x = np.array(dataset["x"][:])
x_pad = np.array(dataset["x_pad"][:])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])

plt.savefig('temp.png')