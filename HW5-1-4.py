import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

K = np.array([[-100, 0, 200],
	 		  [0, -100, 200],
	 		  [0, 0, 1]]);

Mextleft = np.array([[0.707, 0.707, 0, -3],
					 [-0.707, 0.707, 0, -0.5],
					 [0, 0, 1, 3]]);

Mextright = np.array([[0.866, -0.5, 0, -3],
			 		  [0.5, 0.866, 0, -0.5],
					  [0, 0, 1, 3]]);

pts = np.array([[2, 0, 0],
 	   			[3, 0, 0],
 	   			[3, 1, 0],
 	   			[2, 1, 0],
 	   			[2, 0, 1],
 	   			[3, 0, 1],
 	   			[3, 1, 1],
 	   			[2, 1, 1],
 	   			[2.5, 0.5, 2]]);

NN = 9
leftpix = np.array(np.zeros((NN,3)));
rightpix = np.array(np.zeros((NN,3)));

M_left = np.array(np.dot(K, Mextleft));
M_right = np.array(np.dot(K, Mextright));

for i in range(NN):
	pixels = np.dot(M_left,np.transpose(np.array([pts[i,0], pts[i,1], pts[i,2], 1])));
	leftpix[i,:] = np.array(pixels/pixels[2]);
	pixels = np.dot(M_right,np.transpose(np.array([pts[i,0], pts[i,1], pts[i,2], 1])));
	rightpix[i,:] = np.array(pixels/pixels[2]);

def drawmyobject(pix):
	x = [pix[0],pix[0],pix[1],pix[2],pix[4],pix[4],pix[5],pix[6],pix[0],pix[1],pix[2],pix[3],pix[4],pix[5],pix[6],pix[7]]
	y = [pix[1],pix[3],pix[2],pix[3],pix[5],pix[7],pix[6],pix[7],pix[4],pix[5],pix[6],pix[7],pix[8],pix[8],pix[8],pix[8]]
	for i in range(16):
		v1 = x[i]
		v2 = y[i]
		fx = np.array([v1[0], v2[0]])
		fy = np.array([v1[1], v2[1]])
		plt.plot(fx, fy, 'r')
	return;

def nextimg(event):
	if event.key == "n":
		plt.close(event.canvas.figure)

A = np.zeros((9,9))

for j in range(9):
	m1 = np.dot(leftpix[j,0],rightpix[j,0])
	m2 = np.dot(leftpix[j,0],rightpix[j,1])
	m3 = np.dot(leftpix[j,0],rightpix[j,2])
	m4 = np.dot(leftpix[j,1],rightpix[j,0])
	m5 = np.dot(leftpix[j,1],rightpix[j,1])
	m6 = np.dot(leftpix[j,1],rightpix[j,2])
	m7 = np.dot(leftpix[j,2],rightpix[j,0])
	m8 = np.dot(leftpix[j,2],rightpix[j,1])
	m9 = np.dot(leftpix[j,2],rightpix[j,2])
	tt = np.array([[m1,m2,m3],
				   [m4,m5,m6],
				   [m7,m8,m9]])
	A[j,:] = [tt[0,0], tt[0,1], tt[0,2], tt[1,0], tt[1,1], tt[1,2], tt[2,0], tt[2,1], tt[2,2]]

U, S, V = np.linalg.svd(A)
lastcol = V[8,:]

F = np.array([[lastcol[3],lastcol[4],lastcol[2]],
	 		 [lastcol[0],lastcol[1],lastcol[5]],
	 		 [lastcol[6],lastcol[7],lastcol[8]]]);

print '1. Fundamental Matrix:'
print F
print ' '

temp = np.dot(np.transpose(K),F)
E = np.dot(temp,K)

print '2. Essential Matrix:'
print E
print ' '

rightray = np.dot(np.linalg.inv(K),np.transpose(np.array(rightpix)))
leftray = np.dot(np.linalg.inv(K),np.transpose(np.array(leftpix)))

Trw = np.array([[0.866, -0.5, 0, -3],
			 	[0.5, 0.866, 0, -0.5],
				[0, 0, 1, 3], 
				[0, 0, 0, 1]]);

Tlw = np.array([[0.707, 0.707, 0, -3],
				[-0.707, 0.707, 0, -0.5],
				[0, 0, 1, 3], 
				[0, 0, 0, 1]]);

Twr = np.linalg.inv(Trw);
Twl = np.linalg.inv(Tlw); 
Tlr = np.dot(Tlw,Twr);
Rlr = np.array(Tlr[0:3,0:3]);
tlr = np.array(Tlr[0:3,3]);

def triangulate_midpoint(pl,pr, Rlr,tlr):
	plt = pl;
	prt = pr;

	q = np.cross(plt,np.dot(Rlr,prt));
	q = q/np.linalg.norm(q) 

	temp = np.dot(-Rlr,prt)

	A = [[plt[0], temp[0], q[0]], 
		 [plt[1], temp[1], q[1]], 
		 [plt[2], temp[2], q[2]]];

	solveit = np.dot(np.linalg.inv(A),tlr);
	a = solveit[0];
	b = solveit[1];
	c = solveit[2];

	outpoint = np.dot(a,plt) + np.dot(np.dot(c,0.5),q);

	return outpoint

def reconstruct3d(leftray,rightray,Rlr,tlr,Twl):
	reconpts = np.zeros((9,4))
	for i in range(9):
		wrt_left = triangulate_midpoint(leftray[:,i],rightray[:,i],Rlr,tlr);
		temp = np.array([[wrt_left[0]],[wrt_left[1]],[wrt_left[2]],[1]])
		three_d_point = np.array(np.dot(Twl,temp));
		reconpts[i,:] = np.array(np.transpose(three_d_point));

	return reconpts[:,:]

reconpts1 = reconstruct3d(leftray,rightray,Rlr,tlr,Twl);

def drawmy3dobject(pts):
	ax = plt.axes(projection='3d')

	x = [pts[0],pts[0],pts[1],pts[2],pts[4],pts[4],pts[5],pts[6],pts[0],pts[1],pts[2],pts[3],pts[4],pts[5],pts[6],pts[7]]
	y = [pts[1],pts[3],pts[2],pts[3],pts[5],pts[7],pts[6],pts[7],pts[4],pts[5],pts[6],pts[7],pts[8],pts[8],pts[8],pts[8]]
	for i in range(16):
		v1 = x[i]
		v2 = y[i]
		fx = np.array([v1[0], v2[0]])
		fy = np.array([v1[1], v2[1]])
		fz = np.array([v1[2], v2[2]])
		plt.plot(fx, fy, fz, 'r')

	# Data for three-dimensional scattered points
	# zdata = pts[:,2]
	# xdata = pts[:,0]
	# ydata = pts[:,1]
	# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
	return

# plt.figure()
# drawmy3dobject(pts)
# fig = plt.gcf()
# plt.show()

W = [ [0, -1, 0], [1, 0, 0], [0, 0, 1]];
Z = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]];
U,S,V = np.linalg.svd(E);
V = np.array([[V[0,0],V[0,1],V[0,2]],
	 		  [V[1,0],V[1,1],V[1,2]],
	 		  [-V[2,0],-V[2,1],-V[2,2]]])

Up = np.transpose(U)
Wp = np.transpose(W)

tempS1 = np.dot(-U,Z)
S1 = np.dot(tempS1,Up)

tempS2 = np.dot(U,Z)
S2 = np.dot(tempS2,Up)

tempR1 = np.dot(U,Wp)
R1 = U.dot(Wp).dot(V)

tempR2 = np.dot(U,W)
R2 = np.dot(tempR2,V)

foundit = 0;

if(foundit == 0):
	S = S1; R=R1;
	tlr = np.transpose([S[2,1], S[0,2], -S[0,1]]);
	reconpts = reconstruct3d(leftray,rightray,R,tlr,np.eye(4))
	if (reconpts[:,2].min()>0): 
		foundit = 1;

if(foundit == 0):
	S = S2; R=R1;
	tlr = np.transpose([S[2,1], S[0,2], -S[0,1]]);
	reconpts = reconstruct3d(leftray,rightray,R,tlr,np.eye(4))
	if (reconpts[:,2].min()>0): 
		foundit = 1;

if(foundit == 0):
	S = S1; R=R2;
	tlr = np.transpose([S[2,1], S[0,2], -S[0,1]]);
	reconpts = reconstruct3d(leftray,rightray,R,tlr,np.eye(4))
	if (reconpts[:,2].min()>0): 
		foundit = 1;

if(foundit == 0):
	S = S2; R=R2;
	tlr = np.transpose([S[2,1], S[0,2], -S[0,1]]);
	reconpts = reconstruct3d(leftray,rightray,R,tlr,np.eye(4))
	if (reconpts[:,2].min()>0): 
		foundit = 1;

plt.figure()
drawmyobject(leftpix)
plt.title('3a. Left Image');
fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', nextimg)
plt.show()

plt.figure()
drawmyobject(rightpix)
plt.title('3b. Right Image');
fig1 = plt.gcf()
fig1.canvas.mpl_connect('key_press_event', nextimg)
plt.show()

plt.figure()
drawmy3dobject(reconpts)
plt.title('3c. Reconstructed Polygonal Model Using Essential Matrix');
fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', nextimg)
plt.show()

plt.figure()
drawmyobject(leftpix)
plt.title('4a. Left Image');
fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', nextimg)
plt.show()

plt.figure()
drawmyobject(rightpix)
plt.title('4b. Right Image');
fig1 = plt.gcf()
fig1.canvas.mpl_connect('key_press_event', nextimg)
plt.show()

plt.figure()
drawmy3dobject(reconpts1)
plt.title('4c. Reconstructed Polygonal Model Using Camera Matrix');
fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', nextimg)
plt.show()