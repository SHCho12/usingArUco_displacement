import cv2
import numpy as np
import copy
from utils import aruco_display, harris, get_image_homography, find_best_match

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

aruco_type = "DICT_4X4_1000"
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(aruco_type))
arucoParams = cv2.aruco.DetectorParameters()


image = cv2.imread(r"DH0.JPG")

imageview=copy.deepcopy(image)
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
corners, ids, rejected = detector.detectMarkers(imageview)
detected_markers, topRight, bottomRight, bottomLeft, topLeft = aruco_display(corners, ids, rejected, image)
corners = np.array([topLeft, bottomLeft, topRight, bottomRight])


print(f"corners : {corners}")
i_matrix = get_image_homography(corners, 60)

answer_corners = np.array([
    [0, 0],
    [60, 0],
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
    [40, 50],
    [0, 60],
    [60, 60]
], dtype=np.float32)

A = []
for corner in answer_corners:
    x, y = corner[0], corner[1]
    corn_vec = np.array([[x], [y], [1]])
    trans_corn = np.dot(i_matrix, corn_vec)
    trans_corn = np.divide(trans_corn, trans_corn[2])
    A.append(trans_corn)
A = [np.squeeze(arr)[:2] for arr in A]
# print(f"A : {A}")    

for corner in A:
    x, y = int(corner[0]), int(corner[1])
    cv2.circle(imageview, (x, y), 10, (0, 255, 0), -1)
    cv2.putText(imageview, f"({x}, {y})", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


resized_image = cv2.resize(imageview, (0, 0), fx=0.4, fy=0.4)
# 결과 이미지 출력
cv2.imshow('Corners', resized_image)
cv2.waitKey(0)

cv2.destroyAllWindows()