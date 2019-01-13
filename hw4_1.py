# Dhruvik Patel
# Homework 4 - Problem 1


import matplotlib.pyplot as plt
import numpy as np
import cv2

pt1 = np.array([-2.77 , -2.48 , 6, 1], dtype=float)
pt2 = np.array([-0.77 , -2.48 , 6, 1], dtype=float)
pt3 = np.array([-0.77 , -2.48 , 8, 1], dtype=float)
pt4 = np.array([-2.77 , -2.48 , 8, 1], dtype=float)
pt5 = np.array([-2.77 , -4.48 , 6, 1], dtype=float)
pt6 = np.array([-0.77 , -4.28 , 6, 1], dtype=float)
pt7 = np.array([-0.77 , -4.28 , 8, 1], dtype=float)
pt8 = np.array([-2.77 , -4.48 , 8, 1], dtype=float)
pt9 = np.array([-1.77 , -6.28 , 7, 1], dtype=float)

M_int = np.array([[-100, 0, 200], [-0, -100, 200], [0, 0, 1]])
M_est1 = np.array([[0.707, 0.707, 0, -3], [-0.707, 0.707, 0, -0.5], [0, 0, 1, 3]])
M_est2 = np.array([[0.866, -0.5, 0, -3], [0.5, 0.866, 0, -0.5], [0, 0, 1, 3]])

M = np.dot(M_int, M_est1)
M2 = np.dot(np.array(M_int), np.array(M_est2))

pt1t = np.array(np.matmul(M, pt1))
final1M = np.array(pt1t/pt1t[2])

pt1tM2 = np.array(np.matmul(M2, pt1))
final1M2 = np.array(pt1tM2/pt1tM2[2])

pt2t = np.array(np.matmul(M,pt2))
final2M = np.array(pt2t/pt2t[2])

pt2tM2 = np.array(np.matmul(M2, pt2))
final2M2 = np.array(pt2tM2/pt2tM2[2])

pt3t = np.array(np.matmul(M,pt3))
final3M = np.array(pt3t/pt3t[2])

pt3tM2 = np.array(np.matmul(M2, pt3))
final3M2 = np.array(pt3tM2/pt3tM2[2])

pt4t = np.array(np.matmul(M,pt4))
final4M = np.array(pt4t/pt4t[2])

pt4tM2 = np.array(np.matmul(M2, pt4))
final4M2 = np.array(pt4tM2/pt4tM2[2])

pt5t = np.array(np.matmul(M,pt5))
final5M = np.array(pt5t/pt5t[2])

pt5tM2 = np.array(np.matmul(M2, pt5))
final5M2 = np.array(pt5tM2/pt5tM2[2])

pt6t = np.array(np.matmul(M,pt6))
final6M = np.array(pt6t/pt6t[2])

pt6tM2 = np.array(np.matmul(M2, pt6))
final6M2 = np.array(pt6tM2/pt6tM2[2])

pt7t = np.array(np.matmul(M,pt7))
final7M = np.array(pt7t/pt7t[2])

pt7tM2 = np.array(np.matmul(M2, pt7))
final7M2 = np.array(pt7tM2/pt7tM2[2])

pt8t = np.array(np.matmul(M,pt8))
final8M = np.array(pt8t/pt8t[2])

pt8tM2 = np.array(np.matmul(M2, pt8))
final8M2 = np.array(pt8tM2/pt8tM2[2])

pt9t = np.array(np.matmul(M,pt9))
final9M = np.array(pt9t/pt9t[2])

pt9tM2 = np.array(np.matmul(M2, pt9))
final9M2 = np.array(pt9tM2/pt9tM2[2])

def drawmyobject( vertex1, vertex2):
	finalx = np.array([vertex1[0], vertex2[0]])
	finaly = np.array([vertex1[1], vertex2[1]])

	plt.plot(finalx, finaly)

	return;

drawmyobject(final1M, final2M)
drawmyobject(final1M, final4M)
drawmyobject(final2M, final3M)
drawmyobject(final3M, final4M)

drawmyobject(final5M, final6M)
drawmyobject(final5M, final8M)
drawmyobject(final6M, final7M)
drawmyobject(final7M, final8M)

drawmyobject(final1M, final5M)
drawmyobject(final2M, final6M)
drawmyobject(final3M, final7M)
drawmyobject(final4M, final8M)

drawmyobject(final5M, final9M)
drawmyobject(final6M, final9M)
drawmyobject(final7M, final9M)
drawmyobject(final8M, final9M)

def press(event):
	if event.key == "n":
		plt.close(event.canvas.figure)

fig = plt.gcf()

fig.canvas.mpl_connect('key_press_event', press)
plt.show()

plt.figure()

drawmyobject(final1M2, final2M2)
drawmyobject(final1M2, final4M2)
drawmyobject(final2M2, final3M2)
drawmyobject(final3M2, final4M2)

drawmyobject(final5M2, final6M2)
drawmyobject(final5M2, final8M2)
drawmyobject(final6M2, final7M2)
drawmyobject(final7M2, final8M2)

drawmyobject(final1M2, final5M2)
drawmyobject(final2M2, final6M2)
drawmyobject(final3M2, final7M2)
drawmyobject(final4M2, final8M2)

drawmyobject(final5M2, final9M2)
drawmyobject(final6M2, final9M2)
drawmyobject(final7M2, final9M2)
drawmyobject(final8M2, final9M2)

def done(eventKey):
	if eventKey.key == "n":
		plt.close(eventKey.canvas.figure)

fig1 = plt.gcf()

fig1.canvas.mpl_connect('key_press_event', done)
plt.show()

ox = 200
oy = 200
sx = 1
sy = 1
f = 100

twodcamera11 = np.array([(final1M[0]-ox)*(-sx), (final1M[1]-oy)*(-sy)])
twodcamera12 = np.array([(final1M2[0]-ox)*(-sx), (final1M2[1]-oy)*(-sy)])

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

######################################### point 2

twodcamera21 = np.array([(final2M[0]-ox)*(-sx), (final2M[1]-oy)*(-sy)])
twodcamera22 = np.array([(final2M2[0]-ox)*(-sx), (final2M2[1]-oy)*(-sy)])

ptilda21 = np.array([[twodcamera21[0]], [twodcamera21[1]], [f]])
ptilda22 = np.array([[twodcamera22[0]], [twodcamera22[1]], [f]])

tempdot = np.dot(R12, ptilda22)
tempdott = np.transpose(tempdot)

ptilda21cross = np.transpose(ptilda21)

q = np.cross(ptilda21cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda21[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda21[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda21[2][0], -tempdot[2][0], normalqt[2][0]]
						])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point21 = (solSystemOfEq[0]*ptilda21) + (solSystemOfEq[2]/2)*normalqt
point21 = np.array([[point21[0][0]], [point21[1][0]], [point21[2][0]], [1]])

point2w = np.dot(M1inv, point21)

######################################### point 3

twodcamera31 = np.array([(final3M[0]-ox)*(-sx), (final3M[1]-oy)*(-sy)])
twodcamera32 = np.array([(final3M2[0]-ox)*(-sx), (final3M2[1]-oy)*(-sy)])

ptilda31 = np.array([[twodcamera31[0]], [twodcamera31[1]], [f]])
ptilda32 = np.array([[twodcamera32[0]], [twodcamera32[1]], [f]])

tempdot = np.dot(R12, ptilda32)
tempdott = np.transpose(tempdot)

ptilda31cross = np.transpose(ptilda31)

q = np.cross(ptilda31cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda31[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda31[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda31[2][0], -tempdot[2][0], normalqt[2][0]]
						])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point31 = (solSystemOfEq[0]*ptilda31) + (solSystemOfEq[2]/2)*normalqt
point31 = np.array([[point31[0][0]], [point31[1][0]], [point31[2][0]], [1]])

point3w = np.dot(M1inv, point31)

######################################### point 4

twodcamera41 = np.array([(final4M[0]-ox)*(-sx), (final4M[1]-oy)*(-sy)])
twodcamera42 = np.array([(final4M2[0]-ox)*(-sx), (final4M2[1]-oy)*(-sy)])

ptilda41 = np.array([[twodcamera41[0]], [twodcamera41[1]], [f]])
ptilda42 = np.array([[twodcamera42[0]], [twodcamera42[1]], [f]])

tempdot = np.dot(R12, ptilda42)
tempdott = np.transpose(tempdot)

ptilda41cross = np.transpose(ptilda41)

q = np.cross(ptilda41cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda41[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda41[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda41[2][0], -tempdot[2][0], normalqt[2][0]]
						])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point41 = (solSystemOfEq[0]*ptilda41) + (solSystemOfEq[2]/2)*normalqt
point41 = np.array([[point41[0][0]], [point41[1][0]], [point41[2][0]], [1]])

point4w = np.dot(M1inv, point41)

######################################### point 5

twodcamera51 = np.array([(final5M[0]-ox)*(-sx), (final5M[1]-oy)*(-sy)])
twodcamera52 = np.array([(final5M2[0]-ox)*(-sx), (final5M2[1]-oy)*(-sy)])

ptilda51 = np.array([[twodcamera51[0]], [twodcamera51[1]], [f]])
ptilda52 = np.array([[twodcamera52[0]], [twodcamera52[1]], [f]])

tempdot = np.dot(R12, ptilda52)
tempdott = np.transpose(tempdot)

ptilda51cross = np.transpose(ptilda51)

q = np.cross(ptilda51cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda51[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda51[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda51[2][0], -tempdot[2][0], normalqt[2][0]]
						])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point51 = (solSystemOfEq[0]*ptilda51) + (solSystemOfEq[2]/2)*normalqt
point51 = np.array([[point51[0][0]], [point51[1][0]], [point51[2][0]], [1]])

point5w = np.dot(M1inv, point51)

######################################### point 6

twodcamera61 = np.array([(final6M[0]-ox)*(-sx), (final6M[1]-oy)*(-sy)])
twodcamera62 = np.array([(final6M2[0]-ox)*(-sx), (final6M2[1]-oy)*(-sy)])

ptilda61 = np.array([[twodcamera61[0]], [twodcamera61[1]], [f]])
ptilda62 = np.array([[twodcamera62[0]], [twodcamera62[1]], [f]])

tempdot = np.dot(R12, ptilda62)
tempdott = np.transpose(tempdot)

ptilda61cross = np.transpose(ptilda61)

q = np.cross(ptilda61cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda61[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda61[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda61[2][0], -tempdot[2][0], normalqt[2][0]]
						])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point61 = (solSystemOfEq[0]*ptilda61) + (solSystemOfEq[2]/2)*normalqt
point61 = np.array([[point61[0][0]], [point61[1][0]], [point61[2][0]], [1]])

point6w = np.dot(M1inv, point61)

######################################### point 7

twodcamera71 = np.array([(final7M[0]-ox)*(-sx), (final7M[1]-oy)*(-sy)])
twodcamera72 = np.array([(final7M2[0]-ox)*(-sx), (final7M2[1]-oy)*(-sy)])

ptilda71 = np.array([[twodcamera71[0]], [twodcamera71[1]], [f]])
ptilda72 = np.array([[twodcamera72[0]], [twodcamera72[1]], [f]])

tempdot = np.dot(R12, ptilda72)
tempdott = np.transpose(tempdot)

ptilda71cross = np.transpose(ptilda71)

q = np.cross(ptilda71cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda71[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda71[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda71[2][0], -tempdot[2][0], normalqt[2][0]]
						])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point71 = (solSystemOfEq[0]*ptilda71) + (solSystemOfEq[2]/2)*normalqt
point71 = np.array([[point71[0][0]], [point71[1][0]], [point71[2][0]], [1]])

point7w = np.dot(M1inv, point71)

######################################### point 8

twodcamera81 = np.array([(final8M[0]-ox)*(-sx), (final8M[1]-oy)*(-sy)])
twodcamera82 = np.array([(final8M2[0]-ox)*(-sx), (final8M2[1]-oy)*(-sy)])

ptilda81 = np.array([[twodcamera81[0]], [twodcamera81[1]], [f]])
ptilda82 = np.array([[twodcamera82[0]], [twodcamera82[1]], [f]])

tempdot = np.dot(R12, ptilda82)
tempdott = np.transpose(tempdot)

ptilda81cross = np.transpose(ptilda81)

q = np.cross(ptilda81cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda81[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda81[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda81[2][0], -tempdot[2][0], normalqt[2][0]]
						])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point81 = (solSystemOfEq[0]*ptilda81) + (solSystemOfEq[2]/2)*normalqt
point81 = np.array([[point81[0][0]], [point81[1][0]], [point81[2][0]], [1]])

point8w = np.dot(M1inv, point81)

######################################### point 9

twodcamera91 = np.array([(final9M[0]-ox)*(-sx), (final9M[1]-oy)*(-sy)])
twodcamera92 = np.array([(final9M2[0]-ox)*(-sx), (final9M2[1]-oy)*(-sy)])

ptilda91 = np.array([[twodcamera91[0]], [twodcamera91[1]], [f]])
ptilda92 = np.array([[twodcamera92[0]], [twodcamera92[1]], [f]])

tempdot = np.dot(R12, ptilda92)
tempdott = np.transpose(tempdot)

ptilda91cross = np.transpose(ptilda91)

q = np.cross(ptilda91cross, tempdott)
qt = np.transpose(q)

normalq = np.array(q/np.linalg.norm(qt))
normalqt = np.transpose(normalq)

systemOfEq = np.array([
						[ptilda91[0][0], -tempdot[0][0], normalqt[0][0]],
						[ptilda91[1][0], -tempdot[1][0], normalqt[1][0]],
						[ptilda91[2][0], -tempdot[2][0], normalqt[2][0]]
						])

solSystemOfEq = np.linalg.solve(systemOfEq, constant)

point91 = (solSystemOfEq[0]*ptilda91) + (solSystemOfEq[2]/2)*normalqt
point91 = np.array([[point91[0][0]], [point91[1][0]], [point91[2][0]], [1]])

point9w = np.dot(M1inv, point91)

pt1reprojection = np.array(np.matmul(M, point1w))
finalreprojection1M = np.array(pt1reprojection/pt1reprojection[2])

reprojectionError1 = np.array([
								final1M[0]-finalreprojection1M[0],
								final1M[1]-finalreprojection1M[1],
								final1M[2]-finalreprojection1M[2]
								])
print("Reprojection Error for point 1: ")
print reprojectionError1

pt2reprojection = np.array(np.matmul(M, point2w))
finalreprojection2M = np.array(pt2reprojection/pt2reprojection[2])

reprojectionError2 = np.array([
								final2M[0]-finalreprojection2M[0],
								final2M[1]-finalreprojection2M[1],
								final2M[2]-finalreprojection2M[2]
								])
print("Reprojection Error for point 2: ")
print reprojectionError2

pt3reprojection = np.array(np.matmul(M, point3w))
finalreprojection3M = np.array(pt3reprojection/pt3reprojection[2])

reprojectionError3 = np.array([
								final3M[0]-finalreprojection3M[0],
								final3M[1]-finalreprojection3M[1],
								final3M[2]-finalreprojection3M[2]
								])
print("Reprojection Error for point 3: ")
print reprojectionError3

pt4reprojection = np.array(np.matmul(M, point4w))
finalreprojection4M = np.array(pt4reprojection/pt4reprojection[2])

reprojectionError4 = np.array([
								final4M[0]-finalreprojection4M[0],
								final4M[1]-finalreprojection4M[1],
								final4M[2]-finalreprojection4M[2]
								])
print("Reprojection Error for point 4: ")
print reprojectionError4

pt5reprojection = np.array(np.matmul(M, point5w))
finalreprojection5M = np.array(pt5reprojection/pt5reprojection[2])

reprojectionError5 = np.array([
								final5M[0]-finalreprojection5M[0],
								final5M[1]-finalreprojection5M[1],
								final5M[2]-finalreprojection5M[2]
								])
print("Reprojection Error for point 5: ")
print reprojectionError5

pt6reprojection = np.array(np.matmul(M, point6w))
finalreprojection6M = np.array(pt6reprojection/pt6reprojection[2])

reprojectionError6 = np.array([
								final6M[0]-finalreprojection6M[0],
								final6M[1]-finalreprojection6M[1],
								final6M[2]-finalreprojection6M[2]
								])
print("Reprojection Error for point 6: ")
print reprojectionError6

pt7reprojection = np.array(np.matmul(M, point7w))
finalreprojection7M = np.array(pt7reprojection/pt7reprojection[2])

reprojectionError7 = np.array([
								final7M[0]-finalreprojection7M[0],
								final7M[1]-finalreprojection7M[1],
								final7M[2]-finalreprojection7M[2]
								])
print("Reprojection Error for point 7: ")
print reprojectionError7

pt8reprojection = np.array(np.matmul(M, point8w))
finalreprojection8M = np.array(pt8reprojection/pt8reprojection[2])

reprojectionError8 = np.array([
								final8M[0]-finalreprojection8M[0],
								final8M[1]-finalreprojection8M[1],
								final8M[2]-finalreprojection8M[2]
								])
print("Reprojection Error for point 8: ")
print reprojectionError8

pt9reprojection = np.array(np.matmul(M, point9w))
finalreprojection9M = np.array(pt9reprojection/pt9reprojection[2])

reprojectionError9 = np.array([
								final9M[0]-finalreprojection9M[0],
								final9M[1]-finalreprojection9M[1],
								final9M[2]-finalreprojection9M[2]
								])
print("Reprojection Error for point 9: ")
print reprojectionError9

plt.figure()

drawmyobject(finalreprojection1M, finalreprojection2M)
drawmyobject(finalreprojection1M, finalreprojection4M)
drawmyobject(finalreprojection2M, finalreprojection3M)
drawmyobject(finalreprojection3M, finalreprojection4M)

drawmyobject(finalreprojection5M, finalreprojection6M)
drawmyobject(finalreprojection5M, finalreprojection8M)
drawmyobject(finalreprojection6M, finalreprojection7M)
drawmyobject(finalreprojection7M, finalreprojection8M)

drawmyobject(finalreprojection1M, finalreprojection5M)
drawmyobject(finalreprojection2M, finalreprojection6M)
drawmyobject(finalreprojection3M, finalreprojection7M)
drawmyobject(finalreprojection4M, finalreprojection8M)

drawmyobject(finalreprojection5M, finalreprojection9M)
drawmyobject(finalreprojection6M, finalreprojection9M)
drawmyobject(finalreprojection7M, finalreprojection9M)
drawmyobject(finalreprojection8M, finalreprojection9M)

def press(event):
	if event.key == "n":
		plt.close(event.canvas.figure)

fig = plt.gcf()

fig.canvas.mpl_connect('key_press_event', press)
plt.show()
