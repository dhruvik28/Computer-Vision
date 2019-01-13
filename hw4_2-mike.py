import numpy as np
import math

degree0 = input("Enter a degree of freedom about the z0 axis: ")
z0 = int(degree0)

degree1 = input("Enter a degree of freedom about the x1 axis: ")
x1 = int(degree1)

degree2 = input("Enter a degree of freedom about the x2 axis: ")
x2 = int(degree2)

degree3 = input("Enter a degree of freedom about the z3 axis: ")
z3 = int(degree3)

z0Cos = math.cos(math.radians(z0))
z0Sin = math.sin(math.radians(z0))

x1Cos = math.cos(math.radians(x1))
x1Sin = math.sin(math.radians(x1))

x2Cos = math.cos(math.radians(x2))
x2Sin = math.sin(math.radians(x2))

z3Cos = math.cos(math.radians(z3))
z3Sin = math.sin(math.radians(z3))

x_z0 = np.array([z0Cos,z0Sin,0,0])
y_z0 = np.array([-z0Sin,z0Cos,0,0])
z_z0 = np.array([0,0,1,0])
t_z0 = np.array([0,0,0,1])
R_z0 = np.vstack([x_z0,y_z0,z_z0,t_z0])

x_x1 = np.array([1,0,0,0])
y_x1 = np.array([0,x1Cos,x1Sin,0])
z_x1 = np.array([0,-x1Sin,x1Cos,0])
t_x1 = np.array([0,0,10,1])
R_x1 = np.vstack([x_x1,y_x1,z_x1,t_x1])

x_x2 = np.array([1,0,0,0])
y_x2 = np.array([0,x1Cos,x1Sin,0])
z_x2 = np.array([0,-x1Sin,x1Cos,0])
t_x2 = np.array([0,0,20,1])
R_x2 = np.vstack([x_x2,y_x2,z_x2,t_x2])

x_z3 = np.array([z0Cos,z0Sin,0,0])
y_z3 = np.array([-z0Sin,z0Cos,0,0])
z_z3 = np.array([0,0,1,0])
t_z3 = np.array([0,0,15,1])
R_z3 = np.vstack([x_z3,y_z3,z_z3,t_z3])

J01 = np.dot(np.transpose(R_x1), np.transpose(R_z0))
print('0_T_1:\n')
T01 = np.around(J01,decimals=3)
print(T01)
print('\n')

J12 = np.dot(np.transpose(R_x2), np.transpose(R_x1))
print('1_T_2:\n')
T12 = np.around(J12,decimals=3)
print(T12)
print('\n')

J23 = np.dot(np.transpose(R_z3), np.transpose(R_x2))
print('2_T_3:\n')
T23 = np.around(J23,decimals=3)
print(T23)
print('\n')

J03_a = np.dot(J23,np.transpose(R_x1))
J03_b = np.dot(J03_a,np.transpose(R_z0))
print('0_T_3:\n')
T03 = np.around(J03_b,decimals=3)
print(T03)
print('\n')




#yCos = math.cos(y)
#ySin = math.sin(y)


#Rotx = [[1,0,0], [0,xCos,-xSin], [0,xSin,xCos]]

#Roty = [[yCos,0,ySin], [0,1,0], [-ySin,0,yCos]]

#Rotz = [[zCos,-zSin,0], [zSin,zCos,0], [0,0,1]]