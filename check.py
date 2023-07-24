import cv2
import numpy as np
import copy
from utils import aruco_display, harris, best_match, get_image_homography, find_best_match

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


# 이미지 경로
image = cv2.imread(r"DH0.JPG")

# 이미지 크기 조정 (원본 크기 유지)
imageview = copy.deepcopy(image)
# resized_image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
corners, ids, rejected = detector.detectMarkers(imageview)

detected_markers, topRight, bottomRight, bottomLeft, topLeft = aruco_display(corners, ids, rejected, image)
corners = np.array([topLeft, bottomLeft, topRight, bottomRight])
sorted_corners = sorted(corners, key=lambda x: (x[0], x[1]))
sorted_corners2 = sorted(corners, key=lambda y: (y[0], y[1]))
x_mn = sorted_corners[0][0]
x_mx = sorted_corners[3][0]
y_mn = sorted_corners2[0][1]
y_mx = sorted_corners2[3][1]

roi = imageview[y_mn-20:y_mx+20, x_mn-20:x_mx+20]
m_filt_cn = harris(roi)
m_filt_cn_adjusted = m_filt_cn + np.array([[y_mn-20, x_mn-20]])
# print(f"len : {len(m_filt_cn)}")

# 정답 좌표
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


i_matrix = get_image_homography(corners, 60)


A = []
for corner in answer_corners:
    x, y = corner[1], corner[0]
    corn_vec = np.array([[x], [y], [1]])
    trans_corn = np.dot(i_matrix, corn_vec)
    trans_corn = np.divide(trans_corn, trans_corn[2])
    A.append(trans_corn)

A = [np.squeeze(arr)[:2].astype(int) for arr in A]
A1 = copy.deepcopy(A)

x, y = A1[0][0], A1[0][1]
x1,y1 = A1[1][0], A1[1][1]
x2,y2 = A1[2][0], A1[2][1]
x3, y3 = A1[3][0], A1[3][1]

min_distance = float('inf')
min_distance1 = float('inf')
min_distance2 = float('inf')
min_distance3 = float('inf')
for corner in m_filt_cn_adjusted:
    h_x, h_y = corner[1], corner[0]
    distance = np.sqrt((x - h_x) ** 2 + (y - h_y) ** 2)
    if distance < min_distance:
                min_distance = distance
                designated_point = corner

for corner in m_filt_cn_adjusted:
    h_x1, h_y1 = corner[1], corner[0]
    distance1 = np.sqrt((x1 - h_x1) ** 2 + (y1 - h_y1) ** 2)
    if distance1 < min_distance1:
                min_distance1 = distance1
                designated_point1 = corner

for corner in m_filt_cn_adjusted:
    h_x2, h_y2 = corner[1], corner[0]
    distance2 = np.sqrt((x2 - h_x2) ** 2 + (y2 - h_y2) ** 2)
    if distance2 < min_distance2:
                min_distance2 = distance2
                designated_point2 = corner

for corner in m_filt_cn_adjusted:
    h_x3, h_y3 = corner[1], corner[0]
    distance3 = np.sqrt((x3 - h_x3) ** 2 + (y3 - h_y3) ** 2)
    if distance3 < min_distance3:
                min_distance3 = distance3
                designated_point3 = corner

print(f"A[0] : {A[0]}")
print(f"designated_point0 : {designated_point}")
print(f"A[1] : {A[1]}")
print(f"designated_point1 : {designated_point1}")
print(f"A[2] : {A[2]}")
print(f"designated_point2 : {designated_point2}")
print(f"A[3] : {A[3]}")
print(f"designated_point3 : {designated_point3}")
