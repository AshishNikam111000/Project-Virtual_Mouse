import mediapipe as mp
import cv2, math
from numpy.lib import math


class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def FindPos(self, img, handnumber = 0, draw = True):
        lmList = []
        xlist = []
        ylist = []
        bbox = (0, 0, 0, 0)

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handnumber]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                xlist.append(cx)
                ylist.append(cy)

                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)

            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = (xmin-10, ymin-10, xmax+10, ymax+10)

        return lmList, bbox

    def FingersUp(self, img, lmlist, tipIds):
        fingers = []
        if lmlist[tipIds[0]][1] < lmlist[tipIds[0]-1][1]:
            fingers.append(1)
        else: fingers.append(0)
        
        for id in range(1, 5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else: fingers.append(0)

        return fingers

    def findDistance(self, img, lmlist, t1, t2):
        cx , cy = (lmlist[t1][1] + lmlist[t2][1])//2 , (lmlist[t1][2] + lmlist[t2][2])//2
        length = int(math.hypot(lmlist[t1][1] - lmlist[t2][1] , lmlist[t1][2] - lmlist[t2][2]))
        cv2.line(img, (lmlist[t1][1], lmlist[t1][2]), (lmlist[t2][1], lmlist[t2][2]), (255, 0, 255), 5)
        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return length, img , (cx, cy)

        