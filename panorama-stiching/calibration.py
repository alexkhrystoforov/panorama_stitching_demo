import cv2
import numpy as np
import glob
import os


def get_matrix(folder_name, patternSize):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # вместо этих координат задать размеры

    objp = np.zeros((patternSize[1] * patternSize[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(folder_name + '/*.jpeg')

    binary = None
    i = 1
    for fname in images:
        img = cv2. imread(fname)

        _, thr = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(thr, cv2.COLOR_RGB2GRAY)
        gray = cv2.erode(gray, kernel=np.ones((3, 3), np.uint8))

        # Find the chess board corners
        print(i)
        ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
        cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
        cv2.imshow('binary', gray)
        cv2.waitKey(0)

        if ret:
            i+=1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, patternSize, corners2, ret)
            # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # cv2.imshow('img',img)
            # cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
            # cv2.imshow('binary', gray)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(mtx)

    return mtx, dist


def undistort(folder_name, images_number):
    # mtx, dist = get_matrix(folder_name='datasets_for_calibrating/chessboard9x6', patternSize=(9, 6))
    mtx, dist = get_matrix(folder_name='datasets_for_calibrating/chessboard19x14', patternSize=(19, 14))

    images = [cv2.imread(folder_name + f'/frame{i}.jpeg') for i in range(images_number)]
    im_num = 0

    for img in images:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # print(dst.shape)
        # print(img.shape)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        # print(dst.shape)
        # print(img.shape)
        dim = (dst.shape[1], dst.shape[0])
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        stack = np.concatenate((img, dst), axis=0)
        cv2.namedWindow('undistortion', cv2.WINDOW_NORMAL)
        cv2.imshow('undistortion', stack)

        if not os.path.exists(folder_name + '_with_undistortion19x14/'):
            os.mkdir(folder_name + '_with_undistortion19x14/')
            print('folder is created')

        cv2.imwrite(f'{folder_name}_with_undistortion19x14/frame{im_num}.png', dst)
        # cv2.imwrite(f'{folder_name}_with_undistortion9x6/frame{im_num+5}.png', dst)

        im_num += 1
        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # stack = np.concatenate((img, dst), axis=0)
        # cv2.namedWindow('undistort using remapping', cv2.WINDOW_NORMAL)
        # cv2.imshow('undistort using remapping', stack)

        cv2.waitKey(0)


# if __name__ == '__main__':
    # undistort(folder_name='datasets/view', images_number=6)
    # undistort(folder_name='datasets/building', images_number=4)
    # undistort(folder_name='datasets/room', images_number=4)
    # undistort(folder_name='datasets/road', images_number=4)
get_matrix(folder_name='datasets_for_calibrating/chessboard19x14', patternSize=(19, 14))
get_matrix(folder_name='datasets_for_calibrating/chessboard9x6', patternSize=(9, 6))

