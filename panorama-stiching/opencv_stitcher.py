import cv2
import os, os.path


def stitch(folder_name):

    path, dirs, files = next(os.walk(folder_name))
    file_count = len(files)

    imgs = [cv2.imread(folder_name + f'/frame{i}.jpeg') for i in range(file_count-1)]
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    _, result = stitcher.stitch(imgs)

    # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    # cv2.imshow('res', result)
    # cv2.waitKey(0)

    cv2.imwrite(f'opencv_stitcher_results/{path[9:]}.png', result)


if __name__ == '__main__':
    stitch('datasets/view')
    stitch('datasets/room')
    stitch('datasets/building')
    stitch('datasets/road')
