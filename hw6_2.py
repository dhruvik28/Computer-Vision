# Dhruvik Patel
# dp811 - 163001797



import cv2
import numpy as np
from scipy import signal

initialImg = cv2.imread('right.jpg', 0)

finalImg = cv2.imread('outimg.jpg', 0)


# rows, cols = initialImg.shape[:2]

# ffimg = np.zeros([rows, cols])

# for x in range(0, cols-1):
# 	for y in range(0, rows-1):
# 		valueinital = initialImg[y, x]
# 		valuefinal = finalImg[y, x]
# 		value = valuefinal - valueinital
# 		if((value) < 0):
# 			ffimg[y, x] = ((value)) + 256
# 		else:
# 			ffimg[y, x] = value

# cv2.imshow('ffimg', ffimg)
# cv2.waitKey(0)

def cfaffine(iref,iinsp, aparm0, niterate, toplev, botlev):
	I = iref.copy().astype(float)
	F = iinsp.copy().astype(float)

	aparm = aparm0.astype(float)


	for lev in xrange(toplev):
		I = cv2.pyrDown(I)
		F = cv2.pyrDown(F)

		Iparm = affine(I, F, aparm, niterate)
		aparm[2] = Iparm[0][2]*2
		aparm[5] = Iparm[0][5]*2

	if(botlev > 0):
		for lev in range(botlev-1, 0):
			Iparm[2] = Iparm[0][2]*2
			Iparm[5] = Iparm[0][5]*2

	outimg = affinewarp(iref, [0,0,0,0,0,0])
	return [Iparm[0], outimg]

def affinewarp(inimg, aparm):
	rows, cols = inimg.shape[:2]

	outimg = np.zeros([rows, cols])

	for x in range(0,cols):
		for y in range(0,rows):
			uu = aparm[0]*x + aparm[1]*y + aparm[2]
			vv = aparm[3]*x + aparm[4]*y + aparm[5]
			outimg[y][x] = bilinear(inimg, y+vv, x+uu)
	return outimg

def bilinear(inimg, x, y):
	rows, cols = inimg.shape[:2]

	if(rows != cols):
		cols = rows -2

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

def affine(iref, iinsp, aparm0, niterate):
	gradx = grad(iref,'x')
	grady = grad(iref,'y')

	[sy,sx] = iref.shape[:2]
	A = np.zeros([sx*sy,6])
	B = np.zeros([sx*sy, 1])

	for i in range(1,sx):
		for j in range(1,sy):
			ix = gradx[j,i]
			iy = grady[j,i]
			A[(j-1)*sx+i][:] = [ix*i, ix*j, ix, iy*i, iy*j, iy]
	AtAinv = (np.matmul(np.transpose(A),A))
	aparm = aparm0

	for nn in range(1, niterate):
		for i in range(1, sx):
			for j in range(1, sy):
				uu = aparm[0]*i + aparm[1]*j + aparm[2]
				vv = aparm[3]*i + aparm[4]*j + aparm[5]
				B[(j-1)*sx+i] = iref[j,i] - bilinear(iinsp,j+vv,i+uu)

		incremental = np.matmul(AtAinv,(np.matmul(np.transpose(A),B)))
		aparm = aparm + np.transpose(incremental)
	return aparm

def grad(inimg, direction):

	inimg = inimg.copy().astype(float)
	if(direction == 'x'):
		outimg = signal.convolve2d(inimg, [[1, 0, -1],[0, 0, 0]])

	if(direction == 'y'):
		outimg = signal.convolve2d(inimg, np.transpose([[1, 0, -1],[0,0,0]]))

	inimg = inimg.astype(float)/2

	return inimg

aparm = np.array([0,0,0,0,0,0])
aparm, outimg = cfaffine(initialImg, finalImg,aparm,2,3,0)

cv2.imshow('outimg',outimg)
cv2.waitKey(0)

print(aparm)