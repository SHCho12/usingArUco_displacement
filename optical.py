import cv2

# 웹캠 캡쳐 객체 생성
cap = cv2.VideoCapture(0)

# 이전 프레임 저장 변수
prev_frame = None

# 비디오 녹화 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# 반복문 시작
while True:
    # 현재 프레임 읽기
    ret, frame = cap.read()

    # 프레임이 정상적으로 읽어졌다면
    if ret:
        # 프레임 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 이전 프레임이 존재한다면
        if prev_frame is not None:
            # optical flow 계산
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # flow 시각화
            h, w = flow.shape[:2]
            flow_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            step = 16
            for y in range(0, h, step):
                for x in range(0, w, step):
                    fx, fy = flow[y, x]
                    cv2.line(flow_img, (x, y), (int(x+fx), int(y+fy)), (0, 255, 0), 1)

            # flow 출력
            cv2.imshow('Optical Flow', flow_img)

            # 녹화된 영상 저장
            out.write(flow_img)

        # 이전 프레임을 현재 프레임으로 갱신
        prev_frame = gray

    # 키 입력 대기
    key = cv2.waitKey(1)

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break

# 비디오 녹화 객체 해제
out.release()

# 캡쳐 객체 해제
cap.release()

# 윈도우 제거
cv2.destroyAllWindows()
