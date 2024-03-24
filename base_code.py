import cv2
import numpy as np
import copy

image = cv2.imread('./seed_count_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

thresh = cv2.bitwise_not(thresh)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result_image = copy.deepcopy(image)
result_image [thresh == 0] = [255, 255, 255]
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

contour_count = len(contours)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(result_image, f"seed count: {contour_count}", (10, 30), font, 1, (0, 0, 255), 2)

cv2.imshow('Contours', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()