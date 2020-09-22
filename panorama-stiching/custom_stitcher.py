import cv2
import numpy as np
# from .utils import *
import os


def find_best_matches(matches):
    """
    Filter matches by distance
    Args:
         matches: list
    Returns:
        best_matches: list
    """
    best_matches = []
    for m in matches:
        if m.distance < 100:
            best_matches.append(m)

    return best_matches


class Matcher:

    def __init__(self, sift, img0, img1):
        self.kp1, self.des1 = sift.detectAndCompute(img0, None)
        self.kp2, self.des2 = sift.detectAndCompute(img1, None)
        self.norm_hamming = cv2.NORM_HAMMING

    def BF_matcher(self):
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(self.des2, self.des1)
        matches = sorted(matches, key=lambda x: x.distance)
        best_matches = find_best_matches(matches)

        return best_matches


def stitch(images):

    MIN_MATCH_COUNT = 5

    sift = cv2.SIFT_create()
    matcher = Matcher(sift, images[0], images[1])
    best_matches = matcher.BF_matcher()

    if len(best_matches) > MIN_MATCH_COUNT:

        # coordinates of KP
        src_pts = np.float32([matcher.kp2[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([matcher.kp1[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w, d = images[0].shape

        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  # shape 4 1 2

        max_extent = np.max(dst, axis=0)[0].astype(np.int)[::-1]
        sz_out = (max(max_extent[1], images[0].shape[1]), max(max_extent[0], images[0].shape[0]))

        warp = cv2.warpPerspective(images[1], M, dsize=sz_out)
        print('warp', warp.shape)

        result = warp.copy()
        result[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]
        print('result shape', result.shape)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', result)
        cv2.waitKey(0)

        return result


def stitching_demo(folder_name, images_number, mode=None):

    if mode == 'understortion':

        imgs = [cv2.imread(folder_name + f'/frame{i}.jpeg') for i in range(images_number)]
        img_with_undistorion = [cv2.imread(folder_name + f'_with_undistortion/frame{i}.jpeg') for i in range(images_number)]
        res1 = stitch([imgs[0], img_with_undistorion[1]])
        res2 = stitch([imgs[1], img_with_undistorion[2]])
        res3 = stitch([imgs[2], img_with_undistorion[3]])

        if not os.path.exists('my_stitcher_results/' + folder_name + '_with_undistortion_result/'):
            os.mkdir('my_stitcher_results/' + folder_name + '_with_undistortion_result/')
            print('folder is created')

        if images_number == 6:

            res4 = stitch([imgs[3], img_with_undistorion[4]])
            res5 = stitch([imgs[4], img_with_undistorion[5]])
            cv2.imwrite(f'my_stitcher_results/{folder_name}_with_undistortion_result/frame{4}.png', res4)
            cv2.imwrite(f'my_stitcher_results/{folder_name}_with_undistortion_result/frame{5}.png', res5)

        cv2.imwrite(f'my_stitcher_results/{folder_name}_with_undistortion_result/frame{1}.png', res1)
        cv2.imwrite(f'my_stitcher_results/{folder_name}_with_undistortion_result/frame{2}.png', res2)
        cv2.imwrite(f'my_stitcher_results/{folder_name}_with_undistortion_result/frame{3}.png', res3)

    else:
        imgs = [cv2.imread(folder_name + f'/frame{i}.jpeg') for i in range(images_number)]

        res1 = stitch(imgs[:2])
        res2 = stitch(imgs[1:3])
        res3 = stitch(imgs[2:4])

        if not os.path.exists('my_stitcher_results/' + folder_name + '_result/'):
            os.mkdir('my_stitcher_results/' + folder_name + '_result/')
            print('folder is created')

        if images_number == 6:

            res4 = stitch([imgs[3], imgs[4]])
            res5 = stitch([imgs[4], imgs[5]])
            cv2.imwrite(f'my_stitcher_results/{folder_name}_result/frame{44}.png', res4)
            cv2.imwrite(f'my_stitcher_results/{folder_name}_result/frame{55}.png', res5)

        cv2.imwrite(f'my_stitcher_results/{folder_name}_result/frame{11}.png', res1)
        cv2.imwrite(f'my_stitcher_results/{folder_name}_result/frame{22}.png', res2)
        cv2.imwrite(f'my_stitcher_results/{folder_name}_result/frame{33}.png', res3)


if __name__ == '__main__':
    # stitching_demo('datasets/view', images_number=6)
    # stitching_demo('datasets/view', images_number=6, mode='understortion')
    # stitching_demo('datasets/building', images_number=4)
    stitching_demo('datasets/road', images_number=4, mode='understortion')
