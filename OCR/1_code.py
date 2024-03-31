import cv2
import numpy as np
import pandas as pd
import pyautogui
import pytesseract as ptr
from PIL import Image

# 바탕화면 스크린샷 캡처
screenshot = pyautogui.screenshot()
screenshot = np.array(screenshot)
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
ptr.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# 이미지 로드 및 OCR을 통한 텍스트 인식
img = Image.open('OCR\\대한민국헌법(제00010호).png')
data = ptr.image_to_data(img, lang="kor", output_type=ptr.Output.DICT)

# 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(data)
# DataFrame을 Excel 파일로 저장
df.to_excel('OCR\\output.xlsx', index=False)
# 초기화 부분
result = []
prev_left = 0
current_text = ""

# data 딕셔너리에서 'conf' 키에 해당하는 리스트를 순회
for i in range(len(data['conf'])):
    # conf 값이 -1인 경우, 현재 문자열을 결과 리스트에 추가하고 새로운 문자열로 시작
    if data['conf'][i] == -1:
        if current_text:  # 현재 문자열이 비어있지 않은 경우에만 추가
            result.append(current_text)
            current_text = ""
    elif data['conf'][i] < 55:
        pass
    else:
        if i == 0 or data['left'][i] - prev_left >= 30:
            # 첫 번째 단어이거나 현재 'left' 값과 이전 'left' 값의 차이가 30 이상이면 띄어쓰기 후 텍스트 추가
            current_text += " " + data['text'][i]
        else:
            # 그렇지 않으면 바로 텍스트 추가 (붙여쓰기)
            current_text += data['text'][i]
        # 현재 'left' 값을 'prev_left'에 저장
        prev_left = data['left'][i]

# 마지막으로 남은 문자열을 결과 리스트에 추가
if current_text:

    result.append(current_text)

for line in result:
    print(line)