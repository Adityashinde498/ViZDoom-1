import numpy as np
import matplotlib.pyplot as plt

binary_map = np.load("binary_map.npy")
plt.imshow(binary_map, cmap="gray")
plt.title("Loaded Binary Map")
plt.show()
