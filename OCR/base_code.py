import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image

# 바탕화면 스크린샷 캡처
screenshot = pyautogui.screenshot()
screenshot = np.array(screenshot)
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

# 스크린샷 저장
cv2.imwrite('OCR/desktop_screenshot.png', screenshot)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# 이미지 로드 및 OCR을 통한 텍스트 인식
img = Image.open('OCR/desktop_screenshot.png')
text = pytesseract.image_to_string(img)

# 지정한 텍스트가 있는지 확인 (영어)
search_text = 'OCR'
print(text)
if search_text in text:
    print(f"'{search_text}'를 찾았습니다!")
else:
    print(f"'{search_text}'를 찾을 수 없습니다.")
