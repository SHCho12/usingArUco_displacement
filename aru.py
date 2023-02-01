import numpy as np
import cv2
import os
from utils import aruco_display, homography_transformation
import matplotlib.pyplot as plt


class cornersdp :
    def __init__(self):
        
        """
        NOTE: 알고리즘
        """
        
        
        self.ARUCO_DICT = {
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
        aruco_type = "DICT_4X4_1000"

        arucoDict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT.get(aruco_type))
        arucoParams = cv2.aruco.DetectorParameters()


        print(f"path {os.getcwd()}")
        curd = os.getcwd()
        disp_img_path = os.path.join(curd, "./image")
        print(disp_img_path)
        self.disp_img_list=[]

        for (root, directories, files) in os.walk(disp_img_path):
            for file in files:
                if '.JPG' in file:
                    file_path = os.path.join(root, file)
                    print(file_path)
                    self.disp_img_list.append(file_path)

        print(f"img_list: {self.disp_img_list}")        
    
        disp_img = cv2.imread(self.disp_img_list[3])
        corners, ids, rejected = cv2.aruco.detectMarkers(disp_img, arucoDict, parameters=arucoParams)
        detected_markers,topRight, bottomRight, topLeft, bottomLeft = aruco_display(corners, ids, rejected, disp_img)
        width=1000
        height = 1000
        disp_img = cv2.resize(disp_img, (width, height), interpolation=cv2.INTER_CUBIC)
        print(f"{topRight}, {topLeft}, {bottomLeft}, {bottomRight}")

        corners = np.array([topLeft, bottomLeft, topRight, bottomRight])
        #cv2.imshow("Marker_detect", detected_markers)
        cv2.imshow("aruco", disp_img)
        cv2.waitKey(0)

        h_matrix=homography_transformation(corners,5.0, 5.0 )
        print(f"{h_matrix}")
        """
        NOTE: 함수
        """
    

   
    

    
if __name__ == "__main__" :
    print(f"main")
    cornersdp()