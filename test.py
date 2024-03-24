import numpy as np

# 리스트 정의
values = [75.0, 461.5, 0.0, 4441.5, 6371.0, 8903.5, 6380.0, 4394.0, 6358.0, 4722.5, 5682.5, 7779.5, 5631.0, 6073.5, 82.0, 5149.0, 4775.0, 0.5, 0.0, 0.0, 4548.5, 7061.0, 21017.0, 4839.0, 15690.5, 5140.5, 6616.0, 5686.5, 6016.0, 5445.5, 0.0, 6.0, 5244.5]

# Q1과 Q3 계산
Q1 = np.percentile(values, 25)
Q3 = np.percentile(values, 75)

# IQR 계산
IQR = Q3 - Q1

# IQR 범위에 해당하는 값들 필터링
iqr_values = [x for x in values if Q1 <= x <= Q3]

# IQR 부분의 평균값 계산
iqr_mean = np.mean(iqr_values)

print(f"IQR 부분의 평균값: {iqr_mean}")
