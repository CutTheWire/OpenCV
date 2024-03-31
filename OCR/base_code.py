import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image

# 바탕화면 스크린샷 캡처
screenshot = pyautogui.screenshot()
screenshot = np.array(screenshot)
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# 이미지 로드 및 OCR을 통한 텍스트 인식
img = Image.open('OCR\대한민국헌법(제00010호).png')
text = pytesseract.image_to_string(img, lang="eng+kor")

# 지정한 텍스트가 있는지 확인 (영어)
search_text = '민주'
if search_text in text:
    print(f"'{search_text}'를 찾았습니다!")
else:
    print(f"'{search_text}'를 찾을 수 없습니다.")
