import numpy as np
import cv2
from scipy.linalg import svd

W = np.array([
			 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
			 [8., 4., 4., 0., 0., 0., 0., 0., 0.],
			 [8., 4., 8., 0., 0., 0., 8., 4., 8.],
			 [0., 0., 0., 0., 0., 0., 4., 4., 8.],
			 [0., 0., 0., 4., 8., 8., 4., 8., 8.],
			 [8., 8., 8., 8., 8., 8., 8., 8., 8.],
			 [8., 8., 4., 8., 8., 4., 0., 0., 0.],
			 [0., 0., 0., 4., 8., 4., 0., 0., 0.]
			 ])


U, s, VT = svd(W)
print("U = ")
print(np.around(U, decimals=3))
print("s = ")
print(np.around(s, decimals=3))
print("VT = ")
print(np.around(VT, decimals=3))




F = np.array([
			 [-0.374, -0.507, -0.293],
			 [0.44, 0.192, 0.325],
			 [-0.393, -0.076, 0.147]
			 ])

Kl = np.array([
			  [1, 0, 1],
			  [2, 1, 0],
			  [2, 3, 1]
			  ])

Kr = np.array([
			  [3, 0, 1],
			  [1, 0, 0],
			  [2, 1, 2]
			  ])


E = np.matmul(np.matmul(Kl, F), Kr)

print("E = ")
print(np.around(E, decimals=3))



