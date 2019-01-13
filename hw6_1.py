# Dhruvik Patel
# dp811 - 163001797


import cv2
import numpy as np

def affinewarp(inimg, aparm):
	rows, cols = inimg.shape[:2]

	outimg = np.zeros([rows, cols], dtype=float)

	for x in range(1,cols):
		for y in range(1,rows):
			uu = aparm[0]*x + aparm[1]*y + aparm[2]
			vv = aparm[3]*x + aparm[4]*y + aparm[5]
			outimg[y][x] = bilinear(inimg, y+vv, x+uu)
	return outimg

def bilinear(inimg, x, y):
	rows, cols = inimg.shape[:2]

	x = max(x, 1)
	x = min(x, cols)
	y = max(y, 1)
	y = min(y, rows)

	x0 = max(np.floor(x).astype(int), 1)
	x1 = min((x0 + 1), cols)
	y0 = max(np.floor(y).astype(int), 1)
	y1 = min((y0 + 1), rows)

	valul = (inimg[x0][y0]).astype(int)
	valur = (inimg[x1][y0]).astype(int)
	valll = (inimg[x0][y1]).astype(int)
	vallr = (inimg[x1][y1]).astype(int)

	x0 = np.floor(x).astype(int)
	x1 = x0 + 1
	y0 = np.floor(y).astype(int)
	y1 = y0 + 1

	vala = ((x-x0)*valur + (x1-x)*valul).astype(int)
	valb = ((x-x0)*vallr + (x1-x)*valll).astype(int)

	value = ((y-y0)*valb + (y1-y)*vala).astype(int)
	return value

img = cv2.imread('Dhruvik.jpg',0)
aparm = np.array([0.02, 0.01, 10, 0.01, -0.02, 5])

outimg = affinewarp(img, aparm)

cv2.imwrite('outimg.jpg', outimg)

cv2.imshow('out', outimg)
cv2.waitKey(0)
cv2.destroyAllWindows()