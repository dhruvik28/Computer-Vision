from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2

K = np.array(
			[
			[100, 0,200],
			[0,100,200],
			[0,0,1]
			], dtype=float)

R = np.array([
			 [-0.707,0.707,0],
			 [-0.707,-0.707,0],
			 [0,0,1]
			 ], dtype=float)

t = np.array(
			 [
			 [3],
			 [0.5],
			 [3]
			 ], dtype=float)

M = np.array([
			 [-0.707,0.707,0,3],
			 [-0.707,-0.707,0,0.5],
			 [0,0,1,3],
			 ], dtype=float)


M = np.matmul(K,M)

pt1 = np.array([-7 , 4 , 0, 1], dtype=float)
pt2 = np.array([-12 , 4 , 0, 1], dtype=float)
pt3 = np.array([-7 , 8 , 0, 1], dtype=float)
pt4 = np.array([-12 , 8 , 0, 1], dtype=float)
pt5 = np.array([-7 , 4 , 3, 1], dtype=float)
pt6 = np.array([-7 , 8 , 3, 1], dtype=float)
pt7 = np.array([-12 , 4 , 3, 1], dtype=float)
pt8 = np.array([-12 , 8 , 3, 1], dtype=float)
pt9 = np.array([-10 , 6 , 8, 1], dtype=float)

pts1 = np.array(np.matmul(M,pt1), dtype=float)
final1 = np.array(pts1/pts1[2], dtype=float)

print final1

pts2 = np.array(np.matmul(M,pt2), dtype=float)
final2 = np.array(pts2/pts2[2], dtype=float)

print final2

pts3 = np.array(np.matmul(M,pt3), dtype=float)
final3 = np.array(pts3/pts3[2], dtype=float)

print final3

pts4 = np.array(np.matmul(M,pt4), dtype=float)
final4 = np.array(pts4/pts4[2], dtype=float)

print final4

pts5 = np.array(np.matmul(M,pt5), dtype=float)
final5 = np.array(pts5/pts5[2], dtype=float)

print final5

pts6 = np.array(np.matmul(M,pt6), dtype=float)
final6 = np.array(pts6/pts6[2], dtype=float)

print final6

pts7 = np.array(np.matmul(M,pt7), dtype=float)
final7 = np.array(pts7/pts7[2], dtype=float)

print final7

pts8 = np.array(np.matmul(M,pt8), dtype=float)
final8 = np.array(pts8/pts8[2], dtype=float)

print final8

pts9 = np.array(np.matmul(M,pt9), dtype=float)
final9 = np.array(pts9/pts9[2], dtype=float)

print final9

finalx = np.array(
					[
					[final1[0]],
					[final2[0]],
					[final3[0]],
					[final4[0]],
					[final5[0]],
					[final6[0]],
					[final7[0]],
					[final8[0]],
					[final9[0]]
					], dtype=float)

finaly = np.array(
				 [
				 [final1[1]],
				 [final2[1]],
				 [final3[1]],
				 [final4[1]],
				 [final5[1]],
				 [final6[1]],
				 [final7[1]],
				 [final8[1]],
				 [final9[1]]
				 ], dtype=float)

plt.figure()
#ax = fig.add_subplot(111, projection='3d')

plt.plot(finalx, finaly, 'ro')

while True:
		plt.show()
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

cv2.destroyAllWindows()


#plt.savefig('Hw2_4.png')
