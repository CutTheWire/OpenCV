import numpy as np
import cv2
from matplotlib import pyplot as plt

class ImageSegmenter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(self.image_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.kernel = np.ones((3,3), np.uint8)

    def apply_threshold(self):
        """이미지에 임계값을 적용하여 이진 이미지 생성"""
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def remove_noise(self, thresh):
        """노이즈 제거"""
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations = 2)
        return opening

    def get_background_foreground(self, opening):
        """배경 및 전경 영역 확실하게 설정"""
        sure_bg = cv2.dilate(opening, self.kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        return sure_bg, sure_fg, unknown

    def apply_watershed(self, sure_fg, unknown):
        """워터셰드 알고리즘 적용"""
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        markers = cv2.watershed(self.img, markers)
        self.img[markers == -1] = [255,0,0]
        return self.img

if __name__ == "__main__":
    # 사용 예시
    image_path = 'seed_count_image.jpg'
    IS = ImageSegmenter(image_path)
    thresh = IS.apply_threshold()
    opening = IS.remove_noise(thresh)
    sure_bg, sure_fg, unknown = IS.get_background_foreground(opening)
    image = IS.apply_watershed(sure_fg, unknown)
    plt.imshow(image)
    plt.show()