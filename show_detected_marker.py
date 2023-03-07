import numpy as np
import cv2
import os
from utils import  draw_markers
import matplotlib.pyplot as plt
from glob import glob

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

# img path (폴더 지정해주기)
parser.add_argument(
    '--img_path', type=str, default="multiple_exp_97",
    help='Directory of Images for Displacement Measurement'
)
# img 파일 형식 지정
parser.add_argument(
    '--img_ext', type=str, default="jpg",
    help='Image File Extension'
)

# 타겟 크기 지정
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
	
    for img_path in disp_img_list[0:]:
        target_image = cv2.imread(img_path)
        corners, ids, rejected = cv2.aruco.detectMarkers(target_image, arucoDict, parameters=arucoParams)
        detected_markers = draw_markers(corners, ids, rejected, target_image)
        
        detected_markers = cv2.resize(detected_markers, (2000, 1500), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Image", detected_markers)
        cv2.waitKey(0)
    
                

if __name__ == "__main__" :
    main()
