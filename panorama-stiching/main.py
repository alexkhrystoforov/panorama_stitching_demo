import cv2
import numpy as np
# from .utils import *


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
        # bf = cv2.BFMatcher(self.norm_hamming, crossCheck=True)
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(self.des2, self.des1)
        matches = sorted(matches, key=lambda x: x.distance)
        best_matches = find_best_matches(matches)

        return best_matches
    #
    # def FLANN_matcher(self):
    #     FLANN_INDEX_LSH = 6
    #     index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                         table_number=6,
    #                         key_size=12,
    #                         multi_probe_level=1)
    #     search_params = dict(checks=50)
    #
    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
    #     matches = flann.knnMatch(self.des2, self.des1, k=2)
    #     best_matches = lowe_ratio(matches, ratio_thresh=0.8)
    #     return best_matches

    # def BF_matcher_knn(self, k=2):
    #     bf = cv2.BFMatcher(self.norm_hamming, crossCheck=False)  # crosscheck - define if you have 1 match
    #     matches = bf.knnMatch(self.des2, self.des1, k)
    #     matches = sorted(matches, key=lambda x: x[0].distance)
    #     # Lowe's ratio test
    #     best_matches = [[m] for (m, n) in matches if m.distance < 0.75 * n.distance]
    #
    #     return best_matches


def stitch(images):

    MIN_MATCH_COUNT = 5
    orb = cv2.ORB_create(nfeatures=1000)
    sift = cv2.SIFT_create()
    # matcher = Matcher(orb, images[0], images[1])
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


def stitching_demo(folder_name, images_number):
    imgs = [cv2.imread(folder_name + f'/frame{i}.jpeg') for i in range(images_number)]

    res1 = stitch(imgs[:2])
    res2 = stitch(imgs[1:3])
    res3 = stitch(imgs[2:4])

    if folder_name == 'view':
        res4 = stitch(imgs[3:5])
        res5 = stitch(imgs[4:6])
        cv2.imwrite(f'my_stitcher_results/{folder_name}4.jpg', res4)
        cv2.imwrite(f'my_stitcher_results/{folder_name}5.jpg', res5)

    cv2.imwrite(f'my_stitcher_results/{folder_name}1.jpg', res1)
    cv2.imwrite(f'my_stitcher_results/{folder_name}2.jpg', res2)
    cv2.imwrite(f'my_stitcher_results/{folder_name}3.jpg', res3)


if __name__ == '__main__':
    stitching_demo('view', images_number=6)
    # stitching_demo('room', images_number=4)
