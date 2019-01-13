import cv2
import numpy as np

if __name__ == '__main__' :

	im_src = cv2.imread('ts.jpg')
	size = (600,500,3)
	im_dst = np.zeros(size, np.uint8)

	pts_dst = np.array(
						[
						[0,0],
						[size[0] - 1, 0],
						[size[0] - 1, size[1] - 1],
						[0, size[1] - 1]
						], dtype=float
						)


	pts_src = np.array([[363, 321], [399, 50], [549, 241], [523,438]])

	h, status = cv2.findHomography(pts_src, pts_dst)


	im_dst = cv2.warpPerspective(im_src, h, size[0:2])

	while True:

		cv2.imshow("Image", im_dst)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("c"):
			break

	cv2.destroyAllWindows()