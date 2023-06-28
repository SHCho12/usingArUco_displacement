import numpy as np
import cv2
import os
from utils import aruco_display,homography_transformation
import matplotlib.pyplot as plt
from glob import glob

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
    '--img_path', type=str, default="homo",
    help='Directory of Images for Displacement Measurement'
)
# img 파일 형식 지정
parser.add_argument(
    '--img_ext', type=str, default="jpg",
    help='Image File Extension'
)

# 타겟 크기 지정 (단위: mm)
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

    # 변수 초기화
    ref_point = []
    roi_defined = False
    frame = None
    avg_flow = None

    def select_roi(event, x, y, flags, param):
        ref_point = []
        roi_defined = False
        frame = None
        
        # global ref_point, roi_defined, frame

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]
            roi_defined = False

        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            roi_defined = True
            cv2.rectangle(frame, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("Image", frame)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", select_roi)

    # 첫 번째 이미지 로드
    # frame = cv2.imread(disp_img_list[0])

    # 초기 ROI 창 크기 설정
    roi_window_width, roi_window_height = 400, 300
    cv2.resizeWindow("Image", roi_window_width, roi_window_height)

    while True:
        cv2.imshow("Image", disp_img)
        key = cv2.waitKey(1) & 0xFF
        
        # 'r' 키를 누르면 ROI 재설정
        if key == ord("r"):
            ref_point = []
            roi_defined = False
            disp_img
        
        # 'q' 키를 누르면 종료
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()        

    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in disp_img if len(img.shape) == 3]

    # 초기화
    prev_gray = gray_images
    flow_accumulator = np.zeros_like(disp_img, dtype=np.float32)  # float32로 초기화

    # 이미지 간의 optical flow 계산
    for gray in gray_images[1:]:
        # optical flow 계산
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # optical flow 누적
        flow_accumulator[..., :2] += flow
        
        # 이전 프레임 업데이트
        prev_gray = gray

        # ROI 영역 적용
        if roi_defined:
            roi_x1, roi_y1 = ref_point[0]
            roi_x2, roi_y2 = ref_point[1]
            roi_flow = flow_accumulator[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # ROI에서의 변위 계산 (평균 변위)
            avg_flow = np.mean(roi_flow, axis=(0, 1))
            
            # 결과 출력
            # print("ROI 변위 (평균 변위):", avg_flow)

    result = avg_flow[:2]
    result_0 = np.array([[0], [0], [1]])
    result_1 = np.array([[result[0]], [result[1]], [1]])

    h_matrix_list = []
    displacement_list = []
    for img_path in disp_img_list[0:]:
        target_image = cv2.imread(img_path)
        target_points, target_ids, target_rejected = cv2.aruco.detectMarkers(target_image, arucoDict, parameters=arucoParams)
        detected_markers_2,target_TR, target_BR, target_TL, target_BL = aruco_display(target_points, target_ids, target_rejected, target_image)
    

        dest_cn = np.array([target_TL, target_BL, target_TR, target_BR])
        print(f"destcn :",dest_cn)
        h_matrix, displacement = homography_transformation(corners, dest_cn, p_length) 
        displacement_list.append(displacement)
        h_matrix_list.append(h_matrix)

        optical_result_0 = np.dot(h_matrix, result_0)
        optical_result_0 = optical_result_0/optical_result_0[2]
        optical_result_1 = np.dot(h_matrix, result_1)
        optical_result_1 = optical_result_1/optical_result_1[2]

        disp = optical_result_1 - optical_result_0

    print(f"opticalflow calc: {disp}")

    # optical flow 시각화
    magnitude, angle = cv2.cartToPolar(flow_accumulator[..., 0], flow_accumulator[..., 1])
    hsv = np.zeros((disp_img_list[0].shape[0], disp_img_list[0].shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 결과 출력
    cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)  # 조절 가능한 창 크기 설정
    cv2.namedWindow("Image with ROI", cv2.WINDOW_NORMAL)  # 조절 가능한 창 크기 설정
    cv2.resizeWindow("Optical Flow", 800, 600)  # 출력 창 크기 설정
    cv2.resizeWindow("Image with ROI", 800, 600)  # 출력 창 크기 설정
    cv2.imshow("Optical Flow", flow_visualization)
    cv2.imshow("Image with ROI", disp_img_list[0])
    cv2.setMouseCallback("Image with ROI", select_roi)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()