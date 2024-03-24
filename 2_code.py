import cv2
import numpy as np
import copy

class SeedC:
    def __init__(self, input_image_path: str) -> None:
        """
        SeedC 클래스의 생성자입니다. 이미지 경로를 입력받아 이미지를 로드하고 초기화합니다.

        Args:
        - input_image_path: 이미지 파일의 경로
        """
        self.image = cv2.imread(input_image_path)
        self.result_image = copy.deepcopy(self.image)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.threshold  = 120
        self.areas = []

    def binarize(self) -> np.ndarray:
        """
        이미지를 그레이스케일로 변환한 후 이진화를 수행합니다.

        Returns:
        - 이진화된 이미지
        """
        gray = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(thresh)

    def find_contours(self) -> np.ndarray:
        """
        이진화된 이미지에서 윤곽선을 찾습니다.

        Returns:
        - 찾아진 윤곽선 목록
        """
        thresh = self.binarize()
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def Area_contours(self, contours: np.ndarray) -> None:
        """
        윤곽선의 넓이를 계산하여 리스트에 할당합니다.

        Args:
        - contours: 윤곽선 목록
        """
        self.areas = [cv2.contourArea(c) for c in contours] 

    def annotate_contours(self, contours: np.ndarray) -> np.ndarray:
        """
        윤곽선을 이미지에 그리고, 각 윤곽선의 중심에 번호를 부여합니다.

        Args:
        - contours: 윤곽선 목록

        Returns:
        - 윤곽선과 번호가 추가된 이미지
        """
        self.result_image[self.binarize() == 0] = [255, 255, 255]
        cv2.drawContours(self.result_image, contours, -1, (0, 255, 0), 2)
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(self.result_image, str(i + 1), (cX, cY), self.font, 1, (0, 0, 255), 2)
        return self.result_image
    
    def filter_contours_by_size(self, contours: np.ndarray, min_area: int, max_area: int) -> list:
        """
        주어진 윤곽선 목록에서 면적이 min_area와 max_area 사이인 윤곽선만을 필터링합니다.

        Args:
        - contours: np.ndarray, 윤곽선 목록
        - min_area: int, 필터링할 최소 면적 기준값
        - max_area: int, 필터링할 최대 면적 기준값

        Returns:
        - list: min_area와 max_area 사이의 면적을 가진 윤곽선 목록
        """
        filtered_contours = [contour for contour in contours if min_area <= cv2.contourArea(contour) <= max_area]
        return filtered_contours
    
    def median_value_contours(self) -> float:
        """
        중앙값을 찾습니다.

        Returns:
        - median_value: 중앙값
        """
        median_value = np.median(self.areas)
        return median_value

    def IQR_value_contours(self) -> float:
        """
        IQR범위의 평균값을 찾습니다.

        Returns:
        - iqr_mean: IQR범위의 평균값
        """
        Q1 = np.percentile(self.areas, 25)
        Q3 = np.percentile(self.areas, 75)
        # IQR 범위에 해당하는 값들 필터링
        iqr_values = [x for x in self.areas if Q1 <= x <= Q3]
        # IQR 부분의 평균값 계산
        iqr_mean = np.mean(iqr_values)
        return iqr_mean
    
if __name__ == "__main__":
    SC = SeedC('./seed_count_image.jpg')
    contours = SC.find_contours()
    SC.Area_contours(contours)
    value = SC.median_value_contours()
    filter_contours = SC.filter_contours_by_size(contours, value*0.3, value*1.7)
    result_image = SC.annotate_contours(filter_contours)
    
    cv2.putText(result_image, f"seed count: {len(filter_contours)}", (10, 30), SC.font, 1, (0, 0, 255), 2)
    cv2.imshow('Contours', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
