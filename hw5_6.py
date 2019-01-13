# Dhruvik Patel
# Homework 5 - Problem 6, Problem 7, Problem 8

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import svd
from mpl_toolkits import mplot3d

imgL = cv2.imread('left.jpg')
imgR = cv2.imread('right.jpg')

siftL = cv2.SIFT()
siftR = cv2.SIFT()
kpL, desL = siftL.detectAndCompute(imgL,None)
kpR, desR = siftR.detectAndCompute(imgR,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desL, desR, k=2)

good = []
ptsL = []
ptsR = []

for i,(m,n) in enumerate(matches):
	if m.distance < 0.8*n.distance:
		good.append(m)
		ptsR.append(kpR[m.trainIdx].pt)
		ptsL.append(kpL[m.queryIdx].pt)

# draw_params = dict(matchColor = (0,255,0),
# 				   singlePointColor = (255,0,0),
# 				   good = good,
# 				   flags = 0)

# img3 = cv2.drawMatchesKnn(imgL, kpL, imgR, kpR, matches,None, **draw_params)
# plt.imshow(img3,),plt.show()

ptsL = np.float32(ptsL)
ptsR = np.float32(ptsR)
F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.RANSAC, 3, 0.99)

print("F = ")
print F

K = np.array([
			  [1, 0, 0],
			  [0, 1, 0],
			  [0, 0, 1]
			  ])

E = np.matmul(np.matmul(K,F),K)

print("E = ")
print E

U, s, VT = svd(E)

Z = np.array([
			 [0, 1, 0],
			 [-1, 0, 0],
			 [0, 0, 0]
			 ])

W = np.array([
			 [0, -1, 0],
			 [1, 0, 0],
			 [0, 0, 1]
			 ])

S = np.matmul(np.matmul(U, Z), np.transpose(U))
R = np.matmul(np.matmul(U, W), np.transpose(VT))

t = np.matmul(U, np.array([0, 0, 1]))

M = np.matmul(K, np.hstack([R,np.array([[t[0]],[t[1]],[t[2]]])]))

print("M ")
print M

P1 = np.eye(4)

threeD = cv2.triangulatePoints(P1[:3], M[:3], np.transpose(ptsL), np.transpose(ptsR))
threeD /= threeD[3]

plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(threeD[0,:], threeD[1,:], threeD[2,:])

def press(event):
	if event.key == "n":
		plt.close(event.canvas.figure)

fig = plt.gcf()

fig.canvas.mpl_connect('key_press_event', press)

ax.view_init(59, 96)
plt.show()