import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread('hw_5_left.jpg')
imgR = cv2.imread('hw_5_right.jpg')
# grayL= cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
# grayR= cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

siftL = cv2.SIFT()
siftR = cv2.SIFT()
kpL, desL = siftL.detectAndCompute(imgL,None)
kpR, desR = siftR.detectAndCompute(imgR,None)

# imgL=cv2.drawKeypoints(grayL,kpL)
# imgR=cv2.drawKeypoints(grayR,kpR)

# cv2.imwrite('sift_keypointsL.jpg',imgL)
# cv2.imwrite('sift_keypointsR.jpg',imgR)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desL, desR, k=2)

good = []
ptsL = []
ptsR = []

for i,(m,n) in enumerate(matches):
	if m.distance < 0.8*n.distance:
		good.append(m)
		ptsR.append(kpR[m.trainIdx].pt)
		ptsL.append(kpL[m.queryIdx].pt)

draw_params = dict(matchColor = (0,255,0),
				   singlePointColor = (255,0,0),
				   good = good,
				   flags = 0)

# img3 = cv2.drawMatchesKnn(imgL, kpL, imgR, kpR, matches,None, **draw_params)
# img3 = cv2.drawMatches(grayL,kpL,grayR,kpR,good)
# plt.imshow(img3,),plt.show()

ptsL = np.float32(ptsL)
ptsR = np.float32(ptsR)
F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_8POINT)

ptsL = ptsL[mask.ravel()==1]
ptsR = ptsR[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c, channels = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),5,color,-1)
        cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

linesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1,1,2), 2, F)
linesL = linesL.reshape(-1, 3)

img5, img6 = drawlines(imgL, imgR, linesL, ptsL, ptsR)

linesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1,1,2), 1, F)
linesR = linesR.reshape(-1, 3)
img3, img4 = drawlines(imgR, imgL, linesR, ptsR, ptsL)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()


