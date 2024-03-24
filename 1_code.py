import cv2
import numpy as np
import copy

class SeedC:
    def __init__(self, input_image_path: str) -> None:
        self.image = cv2.imread(input_image_path)
        self.result_image = copy.deepcopy(self.image)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.threshold  = 120

    def binarize(self) -> np.ndarray:
        gray = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(thresh)

    def find_contours(self) -> np.ndarray:
        thresh = self.binarize()
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def annotate_contours(self, contours: np.ndarray) -> np.ndarray:
        self.result_image[self.binarize() == 0] = [255, 255, 255]
        cv2.drawContours(self.result_image, contours, -1, (0, 255, 0), 2)
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(self.result_image, str(i + 1), (cX, cY), self.font, 1, (0, 0, 255), 2)
        return self.result_image

if __name__ == "__main__":
    SC = SeedC('./seed_count_image.jpg')
    contours = SC.find_contours()
    result_image = SC.annotate_contours(contours)

    # 결과 이미지를 표시합니다.
    cv2.putText(result_image, f"seed count: {len(SC.find_contours())}", (10, 30), SC.font, 1, (0, 0, 255), 2)
    cv2.imshow('Contours', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()