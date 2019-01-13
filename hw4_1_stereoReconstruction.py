import matplotlib.pyplot as plt
import numpy as np
import cv2


wp1 = np.array([-2.77, -2.48, 6, 1], dtype=float)
pc11 = np.array([274.575, 203.277], dtype=float)
pc12 = np.array([246.209, 244.807], dtype=float)

ox = 200
oy = 200
sx = 1
sy = 1
f = 100

twodcamera11 = np.array([(pc11[0]-ox)*(-sx), (pc11[1]-oy)*(-sy)])
twodcamera12 = np.array([(pc12[0]-ox)*(-sx), (pc12[1]-oy)*(-sy)])

ptilda11 = np.array([[twodcamera11[0]], [twodcamera11[1]], [f]])
ptilda12 = np.array([[twodcamera12[0]], [twodcamera12[1]], [f]])

R1 = np.array([[0.707, 0.707, 0], [-0.707, 0.707, 0], [0, 0, 1]])
R2 = np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]])

R2transposed = np.transpose(R2)

R12 = np.array(np.dot(R1, R2transposed))

tempdot = np.dot(R12, ptilda12)
tempdott = np.transpose(tempdot)

ptilda11cross = np.transpose(ptilda11)

q = np.cross(ptilda11cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda11[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda11[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda11[2][0], -tempdot[2][0], normalqt[2][0]]
						])
constant = np.array([-1.74, -3.27, 0])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point11 = (solSystemOfEq[0]*ptilda11) + (solSystemOfEq[2]/2)*normalqt
point11 = np.array([[point11[0][0]], [point11[1][0]], [point11[2][0]], [1]])

t1 = np.array([[-3], [-0.5], [3]])

M1 = np.concatenate((R1, t1), axis=1)
M1 = np.array([
			  [M1[0][0], M1[0][1], M1[0][2], M1[0][3]],
			  [M1[1][0], M1[1][1], M1[1][2], M1[1][3]],
			  [M1[2][0], M1[2][0], M1[2][2], M1[2][3]],
			  [0, 0, 0, 1]
			  ])

M1inv = np.linalg.inv(M1)

point1w = np.dot(M1inv, point11)

print point1w