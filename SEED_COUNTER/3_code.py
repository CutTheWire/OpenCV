import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import Union

class ImageSegmenter:
    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.img = cv2.imread(self.image_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.kernel = np.ones((3,3), np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.result_image = self.img.copy()  # result_image 속성 추가

    def apply_threshold(self) -> np.ndarray:
        """
        이미지에 임계값을 적용하여 이진 이미지를 생성합니다.

        Returns:
        - thresh: 임계값 적용 후의 이진화된 이미지입니다.
        """
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)        
        return thresh

    def remove_noise(self, thresh: np.ndarray) -> np.ndarray:
        """
        노이즈를 제거합니다.
        
        Args:
        - thresh: 이진화된 이미지입니다.
        
        Returns:
        - opening: 노이즈가 제거된 이미지입니다.
        """
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations = 2)
        return opening

    def get_background_foreground(self, opening) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        배경 및 전경 영역을 확실하게 설정합니다.
        
        Args:
        - opening: 노이즈가 제거된 이미지입니다.
        
        Returns:
        - sure_bg: 배경이 확실한 영역입니다.
        - sure_fg: 전경이 확실한 영역입니다.
        - unknown: 배경과 전경을 구분할 수 없는 영역입니다.
        """
        sure_bg = cv2.dilate(opening, self.kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        plt.imshow(dist_transform)
        plt.show()
        _, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        return sure_bg, sure_fg, unknown

    def apply_watershed(self, sure_fg: np.ndarray, unknown: np.ndarray) -> np.ndarray:
        """
        워터셰드 알고리즘을 적용합니다.
        
        Args:
        - sure_fg: 전경이 확실한 영역입니다.
        - unknown: 배경과 전경을 구분할 수 없는 영역입니다.
        
        Returns:
        - self.img: 워터셰드 알고리즘 적용 후의 이미지입니다.
        """
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        markers = cv2.watershed(self.img, markers)
        self.img[markers == -1] = [255,0,0]
        return self.img
    
    def find_contours(self, sure_fg: np.ndarray) -> np.ndarray:
        """
        이진화된 이미지에서 윤곽선을 찾습니다.

        Returns:
        - contours: 찾아진 윤곽선 목록
        """
        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def binarize(self) -> np.ndarray:
        """
        이미지를 그레이스케일로 변환한 후 이진화를 수행합니다.

        Returns:
        - cv2.bitwise_not(thresh): 이진화된 이미지
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(thresh)
    
    def annotate_contours(self, image: np.uint8, contours: np.ndarray) -> np.ndarray:
        """
        윤곽선을 이미지에 그리고, 각 윤곽선의 중심에 번호를 부여합니다.

        Args:
        - image: 윤곽선을 그릴 이미지
        - contours: 윤곽선 목록

        Returns:
        - image: 윤곽선과 번호가 추가된 이미지
        """
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(image, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
        return image

if __name__ == "__main__":
    image_path = 'SEED_COUNTER\seed_count_image.jpg'
    IS = ImageSegmenter(image_path)
    thresh = IS.apply_threshold()
    opening = IS.remove_noise(thresh)
    sure_bg, sure_fg, unknown = IS.get_background_foreground(opening)

    image = IS.apply_watershed(sure_fg, unknown)
    contours = IS.find_contours(sure_fg)
    cv2.putText(image, f"seed count: {len(contours)}", (30, 70), IS.font, 2, (0, 0, 255), 2)
    annotated_image = IS.annotate_contours(image, contours)  # 수정된 메소드 호출
    plt.imshow(annotated_image)
    plt.show()
