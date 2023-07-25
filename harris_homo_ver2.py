import numpy as np
import cv2
import os
from utils import aruco_display, harris, get_homography_transform, best_match, get_image_homography
import matplotlib.pyplot as plt
from glob import glob

# ArUco Marker 의 정보를 cv2 에서 불러옵니다.

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    

import argparse

parser = argparse.ArgumentParser(description='Arguments for Displacement Measurement')

# img path (폴더 지정해주기)
parser.add_argument(
    '--img_path', type=str, default="harris",
    help='Directory of Images for Displacement Measurement'
)
# img 파일 형식 지정
parser.add_argument(
    '--img_ext', type=str, default="jpg",
    help='Image File Extension'
)

# 타겟 크기 지정 (단위: mm)
parser.add_argument(
    '--p_length', type=int, default=60,
    help='target size to milimeter'
)



def main():
    
    args = parser.parse_args()

    img_path = args.img_path
    img_ext = args.img_ext
    p_length = args.p_length

    disp_img_list = glob(os.path.join(img_path, "*." + img_ext))
    
    aruco_type = "DICT_4X4_1000"
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(aruco_type))
    arucoParams = cv2.aruco.DetectorParameters()
    
    # 기준 사진으로 기준 점 제작
    disp_img = cv2.imread(disp_img_list[0])
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(disp_img)
    detected_markers, topRight, bottomRight, bottomLeft, topLeft = aruco_display(corners, ids, rejected, disp_img)
    corners = np.array([topLeft, bottomLeft, topRight, bottomRight])

    # 호모그래피 행렬 저장
    h_matrix = get_homography_transform(corners, p_length)

    wc_list = []

    for img_path in disp_img_list[0:]:
        answer_corners = np.array([
            [10, 10],
            [20, 10],
            [30, 10],
            [50, 10],
            [10, 20],
            [20, 20],
            [30, 20],
            [40, 20],
            [20, 30],
            [30, 30],
            [40, 30],
            [40, 40],
            [50, 40],
            [30, 50],
            [40, 50]
        ], dtype=np.float32)
        target_image = cv2.imread(img_path)
        target_points, target_ids, target_rejected = detector.detectMarkers(target_image)
        detected_img, target_TR, target_BR, target_BL, target_TL = aruco_display(target_points, target_ids, target_rejected, target_image)
        dest_cn = np.array([target_TL, target_BL, target_TR, target_BR])
        sorted_points = sorted(dest_cn, key=lambda x: x[0])
        sorted_points2 = sorted(dest_cn, key=lambda y: y[1])

        dest_list = dest_cn.tolist()
        # ROI 영역 설정
        x_min = sorted_points[0][0]
        x_max = sorted_points[3][0]
        y_min = sorted_points2[0][1]
        y_max = sorted_points2[3][1]
        # print(f"x_min: {x_min}")
        roi = target_image[y_min:y_max, x_min:x_max]
        filt_cn = harris(roi)
        filt_cn = filt_cn + np.array([[y_min, x_min]])
        i_matrix = get_image_homography(dest_cn, p_length)

        A = []
        for corner in answer_corners:
            x, y = corner[0], corner[1]
            corn_vec = np.array([[x], [y], [1]])
            trans_corn = np.dot(i_matrix, corn_vec)
            trans_corn = np.divide(trans_corn, trans_corn[2])
            A.append(trans_corn)
        A = [np.squeeze(arr)[:2].astype(int) for arr in A]

        designated_points = []
        for corner in A:
            designated_point = best_match(corner, filt_cn)
            designated_points.append(designated_point)
        num_points = len(designated_points)  #19개
        # print(f"num_point 개 수 : {num_points}")

        total_x = 0
        total_y = 0

        for point in designated_points:
            x = point[1]
            y = point[0]

            total_x += x
            total_y += y

        for point in dest_list:
            x = point[0]
            y = point[1]

            total_x += x
            total_y += y


        average_x = total_x / (num_points + 4)
        average_y = total_y / (num_points + 4)
        average_x = int(average_x)
        average_y = int(average_y)

        average_vec = np.array([[average_x], [average_y], [1]])
        worldcoordinate_point = np.dot(h_matrix, average_vec)
        worldcoordinate_point = np.divide(worldcoordinate_point, worldcoordinate_point[2])
        # print(f"worldcoordinate : {worldcoordinate_point}")
        worldcoordinate_point = worldcoordinate_point[:2]

        wc_list.append(worldcoordinate_point)

    print(f"displacement_list : {wc_list}")
    # x_list=[]
    # y_list=[]
    # for i in displacement_list:
    #     pass
    #     print(i)
    #     print(type(i))
    #     x_list.append(float(i[0]))
    #     y_list.append(float(i[1]))

    # print(f"x: {x_list}, y: {y_list}") 
    
    # a = len(displacement_list)
    # b = list(range(a))


    # plt.xlabel("Image")
    # plt.ylabel("displacement")    
    # plt.plot(b, x_list, "bo--", label="x", marker="8")
    # plt.plot(b, y_list, "ro--", label="y", marker="^")
    # plt.title("Displacement")
    # for i, v in enumerate(b):
    #     plt.text(v, x_list[i], x_list[i],
    #              color='blue',
    #              horizontalalignment='center',
    #              verticalalignment = 'top')
    #     plt.text(v, y_list[i], y_list[i],
    #              color='red',
    #              horizontalalignment='center',
    #              verticalalignment='bottom')                      
    # plt.legend()
    # plt.show()

    
if __name__ == "__main__" :
    main()