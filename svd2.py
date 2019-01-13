import numpy as np
import cv2
from scipy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


ptl1 = np.array([-2.77 , -2.48 , 6], dtype=float)
ptl2 = np.array([-0.77 , -2.48 , 6], dtype=float)
ptl3 = np.array([-0.77 , -2.48 , 8], dtype=float)
ptl4 = np.array([-2.77 , -2.48 , 8], dtype=float)
ptl5 = np.array([-2.77 , -4.48 , 6], dtype=float)
ptl6 = np.array([-0.77 , -4.28 , 6], dtype=float)
ptl7 = np.array([-0.77 , -4.28 , 8], dtype=float)
ptl8 = np.array([-2.77 , -4.48 , 8], dtype=float)
ptl9 = np.array([-1.77 , -6.28 , 7], dtype=float)

ptr1 = np.array([-4.77, -4.48, 8], dtype=float)
ptr2 = np.array([-2.77, -4.48, 8], dtype=float)
ptr3 = np.array([-2.77, -4.48, 10], dtype=float)
ptr4 = np.array([-4.77, -4.48, 10], dtype=float)
ptr5 = np.array([-4.77, -6.48, 8], dtype=float)
ptr6 = np.array([-2.77, -6.48, 8], dtype=float)
ptr7 = np.array([-2.77, -6.48, 10], dtype=float)
ptr8 = np.array([-4.77, -6.48, 10], dtype=float)
ptr9 = np.array([-3.77, -6.48, 9], dtype=float)

W = np.array([
			 [(ptl1[0]*ptr1[0]), (ptl1[0]*ptr1[1]), (ptl1[0]*ptr1[2]), (ptl1[1]*ptr1[0]), (ptl1[1]*ptr1[1]), (ptl1[1]*ptr1[2]), (ptl1[2]*ptr1[0]), (ptl1[2]*ptr1[1]), (ptl1[2]*ptr1[2])],
			 [(ptl2[0]*ptr2[0]), (ptl2[0]*ptr2[1]), (ptl2[0]*ptr2[2]), (ptl2[1]*ptr2[0]), (ptl2[1]*ptr2[1]), (ptl2[1]*ptr2[2]), (ptl2[2]*ptr2[0]), (ptl2[2]*ptr2[1]), (ptl2[2]*ptr2[2])],
			 [(ptl3[0]*ptr3[0]), (ptl3[0]*ptr3[1]), (ptl3[0]*ptr3[2]), (ptl3[1]*ptr3[0]), (ptl3[1]*ptr3[1]), (ptl3[1]*ptr3[2]), (ptl3[2]*ptr3[0]), (ptl3[2]*ptr3[1]), (ptl3[2]*ptr3[2])],
			 [(ptl4[0]*ptr4[0]), (ptl4[0]*ptr4[1]), (ptl4[0]*ptr4[2]), (ptl4[1]*ptr4[0]), (ptl4[1]*ptr4[1]), (ptl4[1]*ptr4[2]), (ptl4[2]*ptr4[0]), (ptl4[2]*ptr4[1]), (ptl4[2]*ptr4[2])],
			 [(ptl5[0]*ptr5[0]), (ptl5[0]*ptr5[1]), (ptl5[0]*ptr5[2]), (ptl5[1]*ptr5[0]), (ptl5[1]*ptr5[1]), (ptl5[1]*ptr5[2]), (ptl5[2]*ptr5[0]), (ptl5[2]*ptr5[1]), (ptl5[2]*ptr5[2])],
			 [(ptl6[0]*ptr6[0]), (ptl6[0]*ptr6[1]), (ptl6[0]*ptr6[2]), (ptl6[1]*ptr6[0]), (ptl6[1]*ptr6[1]), (ptl6[1]*ptr6[2]), (ptl6[2]*ptr6[0]), (ptl6[2]*ptr6[1]), (ptl6[2]*ptr6[2])],
			 [(ptl7[0]*ptr7[0]), (ptl7[0]*ptr7[1]), (ptl7[0]*ptr7[2]), (ptl7[1]*ptr7[0]), (ptl7[1]*ptr7[1]), (ptl7[1]*ptr7[2]), (ptl7[2]*ptr7[0]), (ptl7[2]*ptr7[1]), (ptl7[2]*ptr7[2])],
			 [(ptl8[0]*ptr8[0]), (ptl8[0]*ptr8[1]), (ptl8[0]*ptr8[2]), (ptl8[1]*ptr8[0]), (ptl8[1]*ptr8[1]), (ptl8[1]*ptr8[2]), (ptl8[2]*ptr8[0]), (ptl8[2]*ptr8[1]), (ptl8[2]*ptr8[2])],
			 [(ptl9[0]*ptr9[0]), (ptl9[0]*ptr9[1]), (ptl9[0]*ptr9[2]), (ptl9[1]*ptr9[0]), (ptl9[1]*ptr9[1]), (ptl9[1]*ptr9[2]), (ptl9[2]*ptr9[0]), (ptl9[2]*ptr9[1]), (ptl9[2]*ptr9[2])],
			 ])

U, s, VT = svd(W)

F = VT[:,8]
F = np.resize(F, (3,3))
print(np.around(F, decimals=3))

Kl = np.array([
			  [-1, 0, 0],
			  [0, -1, 0],
			  [0, 0, 1]
			  ])

Kr = np.array([
			  [1, 0, 10],
			  [0, 1, 10],
			  [0, 0, 1]
			  ])

E = np.matmul(np.matmul(Kl, F),np.transpose(Kr))

print(np.around(E, decimals=3))

ptl1t = np.array(np.matmul(E, ptl1))
finall1 = np.array(ptl1t/ptl1t[2])

ptl2t = np.array(np.matmul(E, ptl2))
finall2 = np.array(ptl2t/ptl2t[2])

ptl3t = np.array(np.matmul(E, ptl3))
finall3 = np.array(ptl3t/ptl3t[2])

ptl4t = np.array(np.matmul(E, ptl4))
finall4 = np.array(ptl4t/ptl4t[2])

ptl5t = np.array(np.matmul(E, ptl5))
finall5 = np.array(ptl5t/ptl5t[2])

ptl6t = np.array(np.matmul(E, ptl6))
finall6 = np.array(ptl6t/ptl6t[2])

ptl7t = np.array(np.matmul(E, ptl7))
finall7 = np.array(ptl7t/ptl7t[2])

ptl8t = np.array(np.matmul(E, ptl8))
finall8 = np.array(ptl8t/ptl8t[2])

ptl9t = np.array(np.matmul(E, ptl9))
finall9 = np.array(ptl9t/ptl9t[2])

ptr1t = np.array(np.matmul(E, ptr1))
finalr1 = np.array(ptr1t/ptr1t[2])

ptr2t = np.array(np.matmul(E, ptr2))
finalr2 = np.array(ptr2t/ptr2t[2])

ptr3t = np.array(np.matmul(E, ptr3))
finalr3 = np.array(ptr3t/ptr3t[2])

ptr4t = np.array(np.matmul(E, ptr4))
finalr4 = np.array(ptr4t/ptr4t[2])

ptr5t = np.array(np.matmul(E, ptr5))
finalr5 = np.array(ptr5t/ptr5t[2])

ptr6t = np.array(np.matmul(E, ptr6))
finalr6 = np.array(ptr6t/ptr6t[2])

ptr7t = np.array(np.matmul(E, ptr7))
finalr7 = np.array(ptr7t/ptr7t[2])

ptr8t = np.array(np.matmul(E, ptr8))
finalr8 = np.array(ptr8t/ptr8t[2])

ptr9t = np.array(np.matmul(E, ptr9))
finalr9 = np.array(ptr9t/ptr9t[2])

def drawmyobject( vertex1, vertex2):
	finalx = np.array([vertex1[0], vertex2[0]])
	finaly = np.array([vertex1[1], vertex2[1]])

	plt.plot(finalx, finaly)

	return;

drawmyobject(finall1, finall2)
drawmyobject(finall1, finall4)
drawmyobject(finall2, finall3)
drawmyobject(finall3, finall4)

drawmyobject(finall5, finall6)
drawmyobject(finall5, finall8)
drawmyobject(finall6, finall7)
drawmyobject(finall7, finall8)

drawmyobject(finall1, finall5)
drawmyobject(finall2, finall6)
drawmyobject(finall3, finall7)
drawmyobject(finall4, finall8)

drawmyobject(finall5, finall9)
drawmyobject(finall6, finall9)
drawmyobject(finall7, finall9)
drawmyobject(finall8, finall9)

def press(event):
	if event.key == "n":
		plt.close(event.canvas.figure)

fig = plt.gcf()

fig.canvas.mpl_connect('key_press_event', press)
plt.show()

plt.figure()

drawmyobject(finalr1, finalr2)
drawmyobject(finalr1, finalr4)
drawmyobject(finalr2, finalr3)
drawmyobject(finalr3, finalr4)

drawmyobject(finalr5, finalr6)
drawmyobject(finalr5, finalr8)
drawmyobject(finalr6, finalr7)
drawmyobject(finalr7, finalr8)

drawmyobject(finalr1, finalr5)
drawmyobject(finalr2, finalr6)
drawmyobject(finalr3, finalr7)
drawmyobject(finalr4, finalr8)

drawmyobject(finalr5, finalr9)
drawmyobject(finalr6, finalr9)
drawmyobject(finalr7, finalr9)
drawmyobject(finalr8, finalr9)

def press(event):
	if event.key == "n":
		plt.close(event.canvas.figure)

fig = plt.gcf()

fig.canvas.mpl_connect('key_press_event', press)
plt.show()

El = np.matmul(Kl, E)
Er = np.matmul(Kr, E)

ones = np.array([[1], [1], [1]])
El = np.hstack([El, ones])
Er = np.hstack([Er, ones])

ptl = np.array([finall1, finall2, finall3, finall4, finall5, finall6, finall7, finall8, finall9])
ptr = np.array([finalr1, finalr2, finalr3, finalr4, finalr5, finalr6, finalr7, finalr8, finalr9])
ptl = np.transpose(ptl)
ptr = np.transpose(ptr)

print(El)
print(Er)

threeD = cv2.triangulatePoints(El[:3], Er[:3], ptl[:2], ptr[:2])
threeD /= threeD[3]
print(np.around(threeD, decimals=3))

plt.figure()

ax = plt.axes(projection='3d')
def drawmyobject3d( ax, vertex1, vertex2):

	finalx = np.array([vertex1[0], vertex2[0]])
	finaly = np.array([vertex1[1], vertex2[1]])
	finalz = np.array([vertex1[2], vertex2[2]])

	ax.plot3D(finalx, finaly, finalz, )

	return;

drawmyobject3d(ax, threeD[:, 0], threeD[:, 1])
drawmyobject3d(ax, threeD[:, 0], threeD[:, 3])
drawmyobject3d(ax, threeD[:, 1], threeD[:, 2])
drawmyobject3d(ax, threeD[:, 2], threeD[:, 3])

drawmyobject3d(ax, threeD[:, 4], threeD[:, 5])
drawmyobject3d(ax, threeD[:, 4], threeD[:, 7])
drawmyobject3d(ax, threeD[:, 5], threeD[:, 6])
drawmyobject3d(ax, threeD[:, 6], threeD[:, 7])

drawmyobject3d(ax, threeD[:, 0], threeD[:, 4])
drawmyobject3d(ax, threeD[:, 1], threeD[:, 5])
drawmyobject3d(ax, threeD[:, 2], threeD[:, 6])
drawmyobject3d(ax, threeD[:, 3], threeD[:, 7])

drawmyobject3d(ax, threeD[:, 4], threeD[:, 8])
drawmyobject3d(ax, threeD[:, 5], threeD[:, 8])
drawmyobject3d(ax, threeD[:, 6], threeD[:, 8])
drawmyobject3d(ax, threeD[:, 7], threeD[:, 8])

def press(event):
	if event.key == "n":
		plt.close(event.canvas.figure)

fig = plt.gcf()

fig.canvas.mpl_connect('key_press_event', press)
plt.show()

