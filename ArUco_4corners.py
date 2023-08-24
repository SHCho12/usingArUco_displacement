import numpy as np
import cv2
import os
from utils import aruco_display,get_homography_transform
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm


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
    '--img_path', type=str, default="high_55m_fl_3000_2",
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


    disp_img = cv2.imread(disp_img_list[0])
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(disp_img)
    detected_markers, topRight, bottomRight, bottomLeft, topLeft = aruco_display(corners, ids, rejected, disp_img)
    corners = np.array([topLeft, bottomLeft, topRight, bottomRight])

    # 호모그래피 행렬 저장
    h_matrix = get_homography_transform(corners, p_length)
    
    wc_list = []
    for img_path in tqdm(disp_img_list[0:]):
        target_image = cv2.imread(img_path)
        target_points, target_ids, target_rejected = detector.detectMarkers(target_image)
        detected_markers_2,target_TR, target_BR, target_TL, target_BL = aruco_display(target_points, target_ids, target_rejected, target_image)
        
        dest_cn = np.array([target_TL, target_BL, target_TR, target_BR])
        total_x = 0
        total_y = 0

        for point in dest_cn:
            x = point[0]
            y = point[1]

            total_x += x
            total_y += y


        average_x = total_x / 4
        average_y = total_y / 4
        average_x = int(average_x)
        average_y = int(average_y)

        average_vec = np.array([[average_x], [average_y], [1]])
        worldcoordinate_point = np.dot(h_matrix, average_vec)
        worldcoordinate_point = np.divide(worldcoordinate_point, worldcoordinate_point[2])
        # print(f"worldcoordinate : {worldcoordinate_point}")
        worldcoordinate_point = worldcoordinate_point[:2]


        wc_list.append(worldcoordinate_point)

    print(f"displacement_list : {wc_list}")

    reference = wc_list[0]
    x_deviation = [0] + [d[0][0] - reference[0][0] for d in wc_list[1:]]
    y_deviation = [0] + [d[1][0] - reference[1][0] for d in wc_list[1:]]

    # 그래프 생성
    plt.figure(figsize=(10, 6))

    # x 편차 그래프 생성
    plt.subplot(1, 2, 1)
    plt.plot(x_deviation, marker='o', color='blue')
    plt.title('X Displacement from Reference')
    plt.xlabel('Index')
    plt.ylabel('X Displacement')

    # 각 점에 값을 표시
    for i, txt in enumerate(x_deviation):
        plt.annotate(f"{txt:.2f}", (i, x_deviation[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # y 편차 그래프 생성
    plt.subplot(1, 2, 2)
    plt.plot(y_deviation, marker='o', color='orange')
    plt.title('Y Displacement from Reference')
    plt.xlabel('Index')
    plt.ylabel('Y Displacement')

    # 각 점에 값을 표시
    for i, txt in enumerate(y_deviation):
        plt.annotate(f"{txt:.2f}", (i, y_deviation[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.show()
        
                    
if __name__ == "__main__" :
    main()