# Dhruvik Patel
# Homework 4 - Problem 3

import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

objpoints = []
imgpoints = [] 
images = glob.glob('*.jpg')
for imgName in images:
    img = cv.imread(imgName)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7,9), None)
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, (7,9), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

cv.destroyAllWindows()

np.set_printoptions(suppress=True)

print 'K:'
print mtx

rot =[]
rot, jac = cv.Rodrigues(rvecs[0], jacobian=None)

Mwrot = np.dot(mtx, rot)
M = np.around(np.hstack([Mwrot, tvecs[0]]), decimals=3)
print 'M:'
print M


