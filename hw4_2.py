# Dhruvik Patel
# Homework 4 - Problem 2

import cv2
import numpy as np
import math

inputz0 = int(input("What is the degree of freedom about the z0 axis? "))
sinz0 = np.sin((inputz0)*np.pi / 180)
cosz0 = np.cos((inputz0)*np.pi / 180)

inputx1 = int(input("What is the degree of freedom about the x1 axis? "))
sinx1 = math.sin((inputx1)*np.pi / 180)
cosx1 = math.cos((inputx1)*np.pi / 180)

inputx2 = int(input("What is the degree of freedom about the x2 axis? "))
sinx2 = math.sin((inputx2)*np.pi / 180)
cosx2 = math.cos((inputx2)*np.pi / 180)

inputz3 = int(input("What is the degree of freedom about the z3 axis? "))
sinz3 = math.sin((inputz3)*np.pi / 180)
cosz3 = math.cos((inputz3)*np.pi / 180)

aug = np.array([0, 0, 0, 1])

Rz0 = np.array([[cosz0, -sinz0, 0], [sinz0, cosz0, 0], [0, 0, 1]])
tz0 = np.array([[0], [0], [0]])
Tz0 = np.hstack([Rz0, tz0])
Tz0 = np.around(np.vstack([Tz0, aug]), decimals=3)

Rx1 = np.array([[1, 0, 0], [0, cosx1, -sinx1], [0, sinx1, cosx1]])
tx1 = np.array([[0], [0], [10]])
Tx1 = np.hstack([Rx1, tx1])
Tx1 = np.around(np.vstack([Tx1, aug]), decimals=3)

Rx2 = np.array([[1, 0, 0], [0, cosx2, -sinx2], [0, sinx2, cosx2]])
tx2 = np.array([[0], [0], [20]])
Tx2 = np.hstack([Rx2, tx2])
Tx2 = np.around(np.vstack([Tx2, aug]), decimals=3)

Rz3 = np.array([[cosz3, -sinz3, 0], [sinz3, cosz3, 0], [0, 0, 1]])
tz3 = np.array([[0], [0], [15]])
Tz3 = np.hstack([Rz3, tz3])
Tz3 = np.around(np.vstack([Tz3, aug]), decimals=3)

T01 = np.dot(Tx1, Tz0)
print('0T1:')
print T01
print('\n')

T12 = np.dot(Tx2, Tx1)
print('1T2:')
print T12
print('\n')

T23 = np.dot(Tz3, Tx2)
print('2T3:')
print T23
print('\n')

T03First = np.dot(T12, T23)
T03 = np.around(np.dot(T01, T03First), decimals=3)
print('0T3:')
print T03

