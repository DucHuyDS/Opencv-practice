from utils import biggest_area, connect_point, reorder, find_answer, show_answers
import time
import cv2
import numpy as np
import pytesseract
from PIL import Image


def nothing(x):
    pass


img = cv2.imread('img\\answer.jpg')

width, height = 800, 200

img = cv2.resize(img, (width, height))

questions = 20
choices = 4

answer_correct= [3, 4, 1, 3, 3, 1, 2, 3, 4, 1, 4, 4, 2, 1, 3, 2, 4, 1, 2, 3]
# cv2.namedWindow('image') # make a window with name 'image'
# cv2.createTrackbar('thres_1', 'image', 0, 255, nothing) #lower threshold trackbar for window 'image
# cv2.createTrackbar('thres_2', 'image', 0, 255, nothing) #upper threshold trackbar for window 'image

while True:
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thres_1 = cv2.getTrackbarPos('thres_1', 'image')
    # thres_2 = cv2.getTrackbarPos('thres_2', 'image')

    img_canny = cv2.Canny(img_cvt, 70, 130)
    img_blur = cv2.GaussianBlur(img_canny, (5, 5), 1)

    contours, hierarchy = cv2.findContours(img_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area = biggest_area(contours)
    if area.size !=0:
        area = reorder(area)

        img_canny = np.stack((img_canny,)*3, axis=-1)
        img_blur = np.stack((img_blur,)*3, axis=-1)

        test = img.copy()
        cv2.drawContours(test, area, -1, (0, 255, 0), 10)

        pts1 = np.float32(area)
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        new_img = img.copy()
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(new_img, matrix, (width, height))


        img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 199, 5)
        

        answers = find_answer(img_thresh, questions)

        img_thresh = np.stack((img_thresh,)*3, axis=-1)
      
        show_answers(result, questions, answers, answer_correct)


        img_conca = cv2.vconcat([img, result])

    cv2.imshow("Original", img_conca)
    # print(answers)
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break