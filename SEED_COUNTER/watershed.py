import numpy as np
import cv2
from matplotlib import pyplot as plt

# 이미지 로드
img = cv2.imread('SEED_COUNTER\seed_min_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 시각화
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
ax = axes.ravel()

ax[0].set_title("Original Image")
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].axis('off')

# 1. 임계값 적용하여 마스크 생성
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 2. 노이즈 제거를 위한 모폴로지 연산
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

# 3. 배경 영역 확실히 하기
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 4. 전경 영역 확실히 하기
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# 5. 분수령 찾기를 위한 준비
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 마커 레이블링
ret, markers = cv2.connectedComponents(sure_fg)

# 모든 마커에 1을 더해 배경이 0이 되도록 함
markers = markers+1

# 알 수 없는 영역을 0으로 마킹
markers[unknown==255] = 0

# 워터셰드 적용
markers = cv2.watershed(img, markers)
# 마커별로 색을 지정하기 위한 랜덤 색 생성
label_hue = np.uint8(179*markers/np.max(markers))
blank_ch = 255*np.ones_like(label_hue)
img = cv2.merge([label_hue, blank_ch, blank_ch])

# HSV에서 BGR로 변환하여 색상을 시각적으로 인식 가능하게 함
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

# 배경을 검은색으로 설정
img[label_hue==0] = 0



ax[1].set_title("Threshold")
ax[1].imshow(thresh)
ax[1].axis('off')

ax[2].set_title("Dist Transform")
ax[2].imshow(dist_transform)
ax[2].axis('off')

ax[3].set_title("Sure Background")
ax[3].imshow(sure_bg)
ax[3].axis('off')

ax[4].set_title("Sure Foreground")
ax[4].imshow(sure_fg)
ax[4].axis('off')

ax[5].set_title("Watershed Result")
ax[5].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[5].axis('off')

plt.tight_layout()
plt.show()
