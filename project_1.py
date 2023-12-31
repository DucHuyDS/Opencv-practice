import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = HandDetector(detectionCon=0.8)


colorR = (255,0,255)




class DragRect:
    def __init__(self, position, size=[200, 200]):
        self.x, self.y = position[0], position[1]
        self.w, self.h = size[0], size[1]
        self.position_prev = position
        self.first = False
    def update(self, cursor8, is_dragging):
        if self.first == False:
            self.position_prev = cursor8
            self.first = True
        if self.check_position(cursor8):
            if not is_dragging:
                self.position_prev = cursor8
            dx, dy = cursor8[0] - self.position_prev[0], cursor8[1] - self.position_prev[1]
            self.x += dx
            self.y += dy
            self.position_prev = cursor8

    def check_position(self, cursor8):
        if self.x <= cursor8[0] <= self.x + self.w and self.y <= cursor8[1] <= self.y + self.h:
            return True
        return False

reclist = []

w, h = 250, 250

j=0
for i in range(6):
    if i % 5 == 0:
        j = 0
    reclist.append(DragRect([j*w + 50 , 100 + (i//5)*h ]))
    j +=1

length_reclist = len(reclist)
is_dragging = False
optimizer_insert = 0


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands = detector.findHands(img, draw=False)
    # lmlist, _= detector.find
    if hands:
        lmList = hands[0]['lmList']
        cursor4, cursor8 = lmList[4], lmList[8]
        cv2.circle(img, (cursor4[0], cursor4[1]), 10, colorR, cv2.FILLED)
        cv2.circle(img, (cursor8[0], cursor8[1]), 10, colorR, cv2.FILLED)
        distance = detector.findDistance(cursor4[:2], cursor8[:2])

        if distance[0] < 50:
            for i in range(length_reclist):
                if reclist[i].check_position(cursor8):
                    reclist[i].update(cursor8, is_dragging)

                    if optimizer_insert == 0:
                        reclist.insert(0, reclist[i])
                        reclist.pop(i+1)
                        optimizer_insert = i

                    break
            is_dragging = True
        else:
            optimizer_insert = 0
            is_dragging = False


    img_new = img.copy()       
    for rec in reclist:
        cv2.rectangle(img_new, (rec.x, rec.y), (rec.x + rec.w, rec.y + rec.h), colorR,  cv2.FILLED)
        cv2.rectangle(img_new, (rec.x, rec.y), (rec.x + rec.w, rec.y + rec.h), (0,255,0),  thickness=2)
   
    alpha = 0.5
    out = cv2.addWeighted(img_new, alpha, img, 1 - alpha, 0) 
    cv2.imshow("Image", out)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release() 
cv2.destroyAllWindows()
