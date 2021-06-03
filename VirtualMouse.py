import cv2, time, autopy
import mediapipe as mp
import HandModule as hm
import numpy as np

#####################
wcam, hcam = 640, 480
frameR = 100
smoothening = 10
#####################

ptime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

wScr, hScr = autopy.screen.size()
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)


detector = hm.HandDetector(maxHands=1 ,detectionCon=0.85)
tipIds = [4, 8, 12, 16, 20]

while True:
    bbox = 0
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Detecting and Finding Hand Landmarks
    img = detector.FindHands(img)
    lmlist, bbox = detector.FindPos(img, draw=False)
    cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0), 3)

    # Check which fingers are up
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:] 
        x2, y2 = lmlist[12][1:]

        fingers = detector.FingersUp(img, lmlist, tipIds)
        cv2.rectangle(img, (frameR, frameR), (wcam-frameR, hcam-frameR), (255, 0, 255), 3)

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wcam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hcam-frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1: 
            length, img, center = detector.findDistance(img, lmlist, 8, 12)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

            if length<45:
                cv2.circle(img, center, 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()


    ctime = time.time()
    fps = int(1/(ctime - ptime))
    ptime = ctime
    
    cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.imshow("Mouse", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
