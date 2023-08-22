import numpy as np
import cv2
import os
from utils import aruco_display, homography_harris, harris, get_homography_transform, get_image_homography
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
    '--img_path', type=str, default="non_focus_1",
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
    # print(f"corners : {corners}")
    sorted_corners = sorted(corners, key=lambda x: (x[0], x[1]))
    sorted_corners2 = sorted(corners, key=lambda y: (y[0], y[1]))
    x_mn = sorted_corners[0][0]
    x_mx = sorted_corners[3][0]
    y_mn = sorted_corners2[0][1]
    y_mx = sorted_corners2[3][1]

    m_roi = disp_img[y_mn-20:y_mx+20, x_mn-20:x_mx+20]
    m_filt_cn = harris(m_roi)

    # 호모그래피 행렬 저장
    h_matrix = get_homography_transform(corners, p_length)
    i_matrix = get_image_homography(corners, p_length)
    # world coordinate -> image coordinate

    #
    datum_list = []
    for point in corners:
        corn_vec = np.array([[point[0]], [point[1]], [1]])
        trans_corn = np.dot(h_matrix, corn_vec)
        trans_corn = trans_corn/trans_corn[2]
        datum_list.append(trans_corn)
    # print(f"num1: {len(datum_list)}")
    for point in m_filt_cn:
        m_filt_vec = np.array([[point[0]], [point[1]], [1]])
        trans_m_filt = np.dot(h_matrix, m_filt_vec)
        trans_m_filt = trans_m_filt/trans_m_filt[2]
        datum_list.append(trans_m_filt)
    # print(f"num1: {len(datum_list)}")
    total_x = 0
    total_y = 0
    for point in datum_list:
        x = point[0]
        y = point[1]

        total_x += x
        total_y += y

    num_datum = len(datum_list)
    # print(f"num : {num_datum}")
    ave_x = total_x/num_datum
    ave_y = total_y/num_datum
    datum_point = np.array([[ave_x], [ave_y], [1]])
    # print(f"datum_point : {datum_point}")
    # h_matrix_list = []
    displacement_list = []

    for img_path in disp_img_list[0:]:
        target_image = cv2.imread(img_path)
        target_points, target_ids, target_rejected = detector.detectMarkers(target_image)
        detected_img, target_TR, target_BR, target_BL, target_TL = aruco_display(target_points, target_ids, target_rejected, target_image)
    

        dest_cn = np.array([target_TL, target_BL, target_TR, target_BR])

        sorted_points = sorted(dest_cn, key=lambda x: x[0])
        sorted_points2 = sorted(dest_cn, key=lambda y: y[1])
        # ROI 영역 설정
        x_min = sorted_points[0][0]
        x_max = sorted_points[3][0]
        y_min = sorted_points2[0][1]
        y_max = sorted_points2[3][1]
        # print(f"x_min: {x_min}")
        roi = target_image[y_min-20:y_max+20, x_min-20:x_max+20]
        filt_cn = harris(roi)

        
        displacement  = homography_harris(datum_point, h_matrix, dest_cn, filt_cn) 
        displacement_list.append(displacement)
       
    
    x_list=[]
    y_list=[]
    for i in displacement_list:
        pass
        print(i)
        print(type(i))
        x_list.append(float(i[0]))
        y_list.append(float(i[1]))

    print(f"x: {x_list}, y: {y_list}") 
    
    a = len(displacement_list)
    b = list(range(a))


    plt.xlabel("Image")
    plt.ylabel("displacement")    
    plt.plot(b, x_list, "bo--", label="x", marker="8")
    plt.plot(b, y_list, "ro--", label="y", marker="^")
    plt.title("Displacement")
    for i, v in enumerate(b):
        plt.text(v, x_list[i], x_list[i],
                 color='blue',
                 horizontalalignment='center',
                 verticalalignment = 'top')
        plt.text(v, y_list[i], y_list[i],
                 color='red',
                 horizontalalignment='center',
                 verticalalignment='bottom')                      
    plt.legend()
    plt.show()

    
if __name__ == "__main__" :
    main()