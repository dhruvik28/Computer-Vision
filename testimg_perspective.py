import cv2
import numpy as np

if __name__ == '__main__' :

	im_src = cv2.imread('testimg.jpg')
	size = im_src.shape

	pts_src = np.array(
						[
						[0,0],
						[size[1] - 1, 0],
						[size[1] - 1, size[0] - 1],
						[0, size[0] - 1]
						], dtype=float
						);

	im_dst = cv2.imread('TimesSquare.jpg')

	pts_dst = np.array([[603, 420], [594, 506], [703, 522], [708, 444]])

	h, status = cv2.findHomography(pts_src, pts_dst)

	im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

	cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16);

	im_dst = im_dst + im_temp

	while True:

		cv2.imshow("Image", im_dst)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("c"):
			break

	cv2.destroyAllWindows()

