import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from itertools import combinations 
# from .utils import findProjectiveTransform, imfindcircles, find_valid_dest_circles, adjust_gamma, adaptiveThreshold_3ch


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


def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0] + topRight[0] + bottomLeft[0]) / 4.0)
			cY = int((topLeft[1] + bottomRight[1] + topRight[1] + bottomLeft[1]) / 4.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			print(f"cx, cy: {cX, cY}")
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image

aruco_type = "DICT_4X4_250"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(aruco_type))
# arucoDict= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
arucoParams = cv2.aruco.DetectorParameters()

print(f"path {os.getcwd()}")
curd = os.getcwd()
disp_img_path = os.path.join(curd, "module\image")
print(disp_img_path)
disp_img_list=[]

for (root, directories, files) in os.walk(disp_img_path):
    for file in files:
        if '.JPG' in file:
            file_path = os.path.join(root, file)
            print(file_path)
            disp_img_list.append(file_path)

print(f"img_list: {disp_img_list}")


# disp_img = cv2.imread("")

#disp_img = cv2.imread(disp_img_list[0])
#plt.imshow(disp_img)
#plt.show()

for i in disp_img_list:
	
	
    disp_img = cv2.imread(i)
    print(i)

    corners, ids, rejected = cv2.aruco.detectMarkers(disp_img, arucoDict, parameters=arucoParams)
    detected_markers = aruco_display(corners, ids, rejected, disp_img)

	
	
    #cv2.imshow("Marker_detect", detected_markers)
    plt.imshow(disp_img)
    plt.show()
    print("hello world")