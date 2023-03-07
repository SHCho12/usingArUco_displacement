import numpy as np
import cv2
import os
from utils import aruco_display, get_displacements, get_homography_transform
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

# class cornersdp :

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

parser.add_argument(
    '--img_path', type=str, default="multiple_exp_98",
    help='Directory of Images for Displacement Measurement'
)
parser.add_argument(
    '--img_ext', type=str, default="jpg",
    help='Image File Extension'
)
parser.add_argument(
    '--p_length', type=int, default=50,
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
    corners, ids, rejected = cv2.aruco.detectMarkers(disp_img, arucoDict, parameters=arucoParams)
    detected_markers,topRight, bottomRight, topLeft, bottomLeft = aruco_display(corners, ids, rejected, disp_img)
    corners = np.array([topLeft, bottomLeft, topRight, bottomRight])

    h_matrix = get_homography_transform(corners, p_length)

    print(f"호모그래피 행렬 : ", h_matrix)

    displacement_list = []
    for img_path in disp_img_list[0:]:
        target_image = cv2.imread(img_path)
        target_points, target_ids, target_rejected = cv2.aruco.detectMarkers(target_image, arucoDict, parameters=arucoParams)
        detected_markers_2,target_TR, target_BR, target_TL, target_BL = aruco_display(target_points, target_ids, target_rejected, target_image)
    

        dest_cn = np.array([target_TL, target_BL, target_TR, target_BR])
        print(f"destcn :",dest_cn)
        displacement = get_displacements(h_matrix, dest_cn, p_length)
        displacement = np.round(displacement, 3) 
        displacement_list.append(displacement)
    
    print(f"계측된 변위 : {displacement_list}")
    print(f"수 :  {len(displacement_list)}")
    
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