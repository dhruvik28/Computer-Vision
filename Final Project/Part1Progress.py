from __future__ import print_function

import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    print('loading images...')
    img1 = cv.imread('image1.jpg')
    img2 = cv.imread('image2.jpg')
    img3 = cv.imread('image3.jpg')
    img4 = cv.imread('image4.jpg')

    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp12 = stereo.compute(img1, img2).astype(np.float32) / 16.0
    disp34 = stereo.compute(img3, img4).astype(np.float32) / 16.0
    disp13 = stereo.compute(img1, img3).astype(np.float32) / 16.0
    disp14 = stereo.compute(img1, img4).astype(np.float32) / 16.0
    disp23 = stereo.compute(img2, img3).astype(np.float32) / 16.0
    disp24 = stereo.compute(img2, img4).astype(np.float32) / 16.0


    print('generating 3d point cloud...',)
    h, w = img1.shape[:2]
    f = 0.8*w
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h],
                    [0, 0, 0,     -f],
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp12, Q)
    colors = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    mask = disp12 > disp12.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    plt.imshow(disp12)
    plt.show()
    cv.waitKey()
    cv.destroyAllWindows()

    print('generating 3d point cloud...',)
    h, w = img1.shape[:2]
    f = 0.8*w
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h],
                    [0, 0, 0,     -f],
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp34, Q)
    colors = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    mask = disp34 > disp34.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out34.ply'
    write_ply('out34.ply', out_points, out_colors)
    print('%s saved' % 'out34.ply')

    plt.imshow(disp34)
    plt.show()
    cv.waitKey()
    cv.destroyAllWindows()    