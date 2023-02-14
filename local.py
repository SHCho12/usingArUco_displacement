import numpy as np
import cv2
import os
from utils import aruco_display, homography_transformation, get_homography_transformation
# import matplotlib.pyplot as plt
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

parser.add_argument(
    '--img_path', type=str, default="image",
    help='Directory of Images for Displacement Measurement'
)
parser.add_argument(
    '--img_ext', type=str, default="jpg",
    help='Image File Extension'
)
parser.add_argument(
    '--target_length', type=int, default=50,
    help='Image File Extension'
)



def main():
    
    args = parser.parse_args()
    
    p_length = args.target_length
    img_path = args.img_path
    img_ext = args.img_ext

    disp_img_list = glob(os.path.join(img_path, "*." + img_ext))
    
    aruco_type = "DICT_4X4_1000"
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(aruco_type))
    arucoParams = cv2.aruco.DetectorParameters()


    disp_img = cv2.imread(disp_img_list[0])
    corners, ids, rejected = cv2.aruco.detectMarkers(disp_img, arucoDict, parameters=arucoParams)
    detected_markers,TR, BR, TL, BL = aruco_display(corners, ids, rejected, disp_img)
    corners = np.array([TL, BL, TR, BR])

    print(f"corners: ", corners)
if __name__ == "__main__" :
    main()