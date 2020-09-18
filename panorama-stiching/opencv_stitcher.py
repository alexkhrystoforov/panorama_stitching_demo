import cv2


def stitcher(folder_name, images_number):
    if images_number == 4:
        imgs = [cv2.imread(folder_name + f'/frame{i}.jpeg') for i in range(4)]

        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        _, result1 = stitcher.stitch(imgs[:2])
        _, result2 = stitcher.stitch(imgs[2:])

        print('res1 shape', result1.shape)
        print('res2 shape', result2.shape)

        dim = None

        if folder_name == 'room':
            dim = (2027, 1012)
        elif folder_name == 'road' or folder_name == 'building':
            dim = (1250, 1017)

        result1 = cv2.resize(result1, dim, interpolation=cv2.INTER_AREA)
        result2 = cv2.resize(result2, dim, interpolation=cv2.INTER_AREA)

        images = [result1, result2]

        _, final = stitcher.stitch(images)
        print('final shape', final.shape)

        cv2.namedWindow('res1', cv2.WINDOW_NORMAL)
        cv2.imshow('res1', result1)

        cv2.namedWindow('res2', cv2.WINDOW_NORMAL)
        cv2.imshow('res2', result2)

        cv2.namedWindow('final', cv2.WINDOW_NORMAL)
        cv2.imshow('final', final)

        cv2.imwrite(f'opencv_stitcher_results/{folder_name}.jpg', final)

        cv2.waitKey(0)

    elif images_number == 6:
        imgs = [cv2.imread(folder_name + f'/frame{i}.jpeg') for i in range(6)]

        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        _, result1 = stitcher.stitch(imgs[:2])
        _, result2 = stitcher.stitch(imgs[1:3])

        _, result3 = stitcher.stitch(imgs[3:5])
        _, result4 = stitcher.stitch(imgs[4:6])

        # print(result1.shape)
        # print(result2.shape)
        # print(result3.shape)
        # print(result4.shape)

        dim = (2027, 1012)
        result1 = cv2.resize(result1, dim, interpolation=cv2.INTER_AREA)
        result2 = cv2.resize(result2, dim, interpolation=cv2.INTER_AREA)

        result3 = cv2.resize(result3, dim, interpolation=cv2.INTER_AREA)
        result4 = cv2.resize(result4, dim, interpolation=cv2.INTER_AREA)

        images1 = [result1, result2]

        _, result_1 = stitcher.stitch(images1)

        images2 = [result3, result4]

        st, result_2 = stitcher.stitch(images2)

        dim = (1600, 1017)
        result_1 = cv2.resize(result_1, dim, interpolation=cv2.INTER_AREA)
        result_2 = cv2.resize(result_2, dim, interpolation=cv2.INTER_AREA)

        # print(result_1.shape)
        # print(result_2.shape)

        images_final = [result_1, result_2]

        st, final = stitcher.stitch(images_final)
        # dim = (3000, 1017)
        # final = cv2.resize(final, dim, interpolation=cv2.INTER_AREA)

        cv2.namedWindow('res1', cv2.WINDOW_NORMAL)
        cv2.imshow('res1', result1)

        cv2.namedWindow('res2', cv2.WINDOW_NORMAL)
        cv2.imshow('res2', result2)

        cv2.namedWindow('res3', cv2.WINDOW_NORMAL)
        cv2.imshow('res3', result3)

        cv2.namedWindow('res4', cv2.WINDOW_NORMAL)
        cv2.imshow('res4', result4)

        cv2.namedWindow('prefinal1', cv2.WINDOW_NORMAL)
        cv2.imshow('prefinal1', result_1)

        cv2.namedWindow('prefinal2', cv2.WINDOW_NORMAL)
        cv2.imshow('prefinal2', result_2)

        cv2.namedWindow('final', cv2.WINDOW_NORMAL)
        cv2.imshow('final', final)

        cv2.imwrite(f'opencv_stitcher_results/{folder_name}1.jpg', result_1)
        cv2.imwrite(f'opencv_stitcher_results/{folder_name}2.jpg', result_2)
        cv2.imwrite(f'opencv_stitcher_results/{folder_name}_final.jpg', final)

        cv2.waitKey(0)


if __name__ == '__main__':
    # stitcher('view', images_number=6)
    stitcher('room', images_number=4)