import numpy as np
import cv2
import os
from utils import aruco_display, homography_transformation
import matplotlib.pyplot as plt
import glob

# class cornersdp :


# argparse 

width = 1000
height = 1000
aruco_type = "DICT_4X4_1000"
path = "\image"
disp_img_list = glob.glob(f"{path}/*.jpg")

def main(self):
    
    """
    NOTE: 알고리즘
    """
    
    
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
    

    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(aruco_type))
    arucoParams = cv2.aruco.DetectorParameters()



    # print(f"path {os.getcwd()}")
    # curd = os.getcwd()
    # disp_img_path = os.path.join(curd, "./image")
    # print(disp_img_path)
    # disp_img_list=[]



    # glob를 사용하면 간편 

    # for (root, directories, files) in os.walk(disp_img_path):
    #     for file in files:
    #         if '.jpg' in file:
    #             file_path = os.path.join(root, file)
    #             print(file_path)
    #             self.disp_img_list.append(file_path)

    # print(f"img_list: {self.disp_img_list}")        
    

    # 변수명을 initialized 암시하는 그런 변수명들 
    disp_img = cv2.imread(disp_img_list[0])
    corners, ids, rejected = cv2.aruco.detectMarkers(disp_img, arucoDict, parameters=arucoParams)
    detected_markers,topRight, bottomRight, topLeft, bottomLeft = aruco_display(corners, ids, rejected, disp_img)
    
    disp_img_1 = cv2.resize(disp_img, (width, height), interpolation=cv2.INTER_CUBIC)
    print(f"{topRight}, {topLeft}, {bottomLeft}, {bottomRight}")
    

    corners = np.array([topLeft, bottomLeft, topRight, bottomRight])
    # cv2.imshow("Marker_detect", detected_markers)
    cv2.imshow("aruco", disp_img_1)
    cv2.waitKey(0)

    # h_matrix=homography_transformation(corners,5.0, 5.0 )
    # print(f"{h_matrix}")
    for img_list in disp_img_list:
        disp_img_2 = cv2.imread(self.disp_img_list[1])
        corners_2, ids_2, rejected_2 = cv2.aruco.detectMarkers(disp_img_2, arucoDict, parameters=arucoParams)
        detected_markers_2,topRight_2, bottomRight_2, topLeft_2, bottomLeft_2 = aruco_display(corners_2, ids_2, rejected_2, disp_img_2)
        disp_img_2_1 = cv2.resize(disp_img_2, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("aruco", disp_img_2_1)
        cv2.waitKey(0)


        dest_cn = np.array([topLeft_2, bottomLeft_2, topRight_2, bottomRight_2])
        h_matrix, displacement = homography_transformation(corners, dest_cn, 50 ) 
        displacement_list.append(displacement)
            

    
    print(f"호모그래피 행렬: {h_matrix}")
    print(f"계측된 변위 : {displacement}")

    
if __name__ == "__main__" :
    main(1)