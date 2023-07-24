import cv2
import numpy as np
import copy
from utils import aruco_display, harris

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
image = cv2.imread(r"DSCN0006.JPG")

# 이미지 크기 조정 (원본 크기 유지)
imageview = copy.deepcopy(image)
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

roi = imageview[y_mn-50:y_mx+50, x_mn-50:x_mx+50]
m_filt_cn = harris(roi)
print(f"len : {len(m_filt_cn)}")
# 원본 이미지에 코너 시각화
for corner in m_filt_cn:
    print(corner)
    x, y = corner[1] + x_mn-50  , corner[0] + y_mn-50
    cv2.circle(imageview, (x, y), 10, (0, 255, 0), -1)
    cv2.putText(imageview, f"({x}, {y})", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

resized_image = cv2.resize(imageview, (0, 0), fx=0.3, fy=0.3)
# 결과 이미지 출력
cv2.imshow('Corners', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
