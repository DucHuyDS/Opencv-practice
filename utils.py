import cv2
import numpy as np



def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]   #top-left
    myPointsNew[2] = myPoints[np.argmax(add)]   #bottom-right

    diff = np.diff(myPoints, axis=1)

    myPointsNew[1] = myPoints[np.argmin(diff)]   #top-right
    myPointsNew[3] = myPoints[np.argmax(diff)]   #bottom-left

    return myPointsNew

def biggest_area(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
    # print(biggest)
    return biggest

def connect_point(img, points, color, thickness):
    cv2.line(img, points[0][0],points[1][0], color, thickness)
    cv2.line(img, points[1][0],points[2][0], color, thickness)
    cv2.line(img, points[2][0],points[3][0], color, thickness)
    cv2.line(img, points[3][0],points[0][0], color, thickness)


def find_answer(img, questions):
    list_answer = []
    columns = np.hsplit(img, questions)
    for column in columns:
        rows = np.vsplit(column,5)
        # print(rows)
        count = np.count_nonzero(rows, axis=-1)
        index_count = np.sum(count, axis=-1)
        list_answer.append(np.argmin(index_count))

    return list_answer


def show_answers(img, questions, answers, answers_correct):
    row = img.shape[0] // 5
    column = img.shape[1] // questions

    for i in range(0, questions):
        
        center_x = (i*column) + column // 2
        center_y = answers[i]*row + row//2
        if answers[i] == answers_correct[i]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            center_x_correct = (i*column) + column // 2
            center_y_correct = answers_correct[i]*row + row//2

            cv2.circle(img, (int(center_x_correct), int(center_y_correct)), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (int(center_x), int(center_y)), 15, color, cv2.FILLED)