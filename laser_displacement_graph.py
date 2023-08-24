import matplotlib.pyplot as plt
import numpy as np

def create_graph(data):
    x = list(range(len(data)))  # x 축 데이터 (0부터 n-1까지의 값)
    y = data  # y 축 데이터 (입력한 숫자 데이터)
    
    plt.figure(figsize=(5, 6))  # 그래프 크기 설정
    plt.plot(y, marker='o', color='Red')  # 선 그래프 그리기 (마커는 'o'로 설정)
    
    plt.title("Laser Displacement Measurement")  # 그래프 제목 설정
    plt.xlabel("Index")  # x 축 레이블 설정
    plt.ylabel("Y Displacement")  # y 축 레이블 설정
    
    for i, txt in enumerate(y):
        plt.annotate(f"{txt:.2f}", (i, y[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.show()

# 사용자로부터 숫자 데이터 개수 입력 받기
n = int(input("숫자 데이터 개수를 입력하세요: "))
data = []

# 숫자 데이터 입력 받기
for i in range(n):
    value = float(input(f"{i+1}번째 데이터를 입력하세요: "))
    data.append(value)

# 첫 번째 값으로부터의 변위 계산
first_value = data[0]
data = np.array(data) - first_value

# 그래프 생성 함수 호출
create_graph(data)
