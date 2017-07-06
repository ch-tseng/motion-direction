# import the necessary packages
from collections import deque
import numpy as np
import time
import imutils
import cv2

trackLength = 32
directionPoints = 10

pts = deque(maxlen=trackLength)
direction = ""
(dX, dY) = (0, 0)
contoursNow = False  #Is there any objects found?
noContoursFoundAccu = 0  #Count how many frames continously no contours found

camera = cv2.VideoCapture(0)

def grepObject(t0, t1):
    global pts, direction, dX, dY, noContoursFoundAccu, contoursNow


    #grey1 = cv2.cvtColor(t0, cv2.COLOR_BGR2GRAY)
    #grey2 = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)
    
    c1 = cv2.cvtColor(t0, cv2.COLOR_BGR2LAB)
    c2 = cv2.cvtColor(t1, cv2.COLOR_BGR2LAB)
    _, grey1, _  = cv2.split(c1)
    _, grey2, _ = cv2.split(c2)
    
    d = cv2.absdiff(grey1, grey2)
    d = cv2.GaussianBlur(d,(15,15),0)

    #cv2.imshow("Gray", d)
    #cv2.imwrite("gray-"+str(int(time.time()))+".jpg", d)

    ret, mask = cv2.threshold( d, 8, 255, cv2.THRESH_BINARY )
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=10)
    cv2.imshow("Mask", mask)
    #cv2.imwrite("mask-"+str(int(time.time()))+".jpg", mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    areas = [cv2.contourArea(c) for c in cnts]

    if(len(areas)>0):
        contoursNow = True
        noContoursFoundAccu = 0

        max_index = np.argmax(areas)
        cnt=cnts[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        print("area={}".format(areas[max_index]))  
        if(areas[max_index]>10000):
            cv2.rectangle(t1,(x,y),(x+w,int(y+w*1.35)), (0,255,0),2)
            #M = cv2.moments(cnt)
            #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center = (int( x+(w/2)), y+30)
    else:
        contoursNow = False
        noContoursFoundAccu += 1

    pts.append(center)

    for ii in np.arange(1, len(pts)):
        i = len(pts) - ii

        if(len(pts)>directionPoints):
            #print ("i={}, len(pts)={}, pts[-10]={}".format(i, len(pts), pts[-directionPoints]))

            if pts[i - 1] is None or pts[i] is None:
                continue            
            elif pts[-directionPoints] is not None:
                dX = pts[-directionPoints][0] - pts[i][0]
                dY = pts[-directionPoints][1] - pts[i][1]
                (dirX, dirY) = ("", "")

                if(np.abs(dX)) > 0:
                    if(np.sign(dX) == 1): dirX = "East"
                    else: dirX = "West"

                if np.abs(dY) > 0:
                    if(np.sign(dY) == 1): dirY = "South"
                    else: dirY = "North"

                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)

                else:
                    direction = dirX if dirX != "" else dirY


        thickness = int(np.sqrt(trackLength/(float(ii + 1))) * 7.2)
        #thickness = int( (trackLength / int( (i + 1))^3)**(1/2) )
        cv2.line(t1, pts[i - 1], pts[i], (0, 250, 253), thickness)

    cv2.putText(t1, "West", (10, t1.shape[0]/2), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 0, 0), 2)
    cv2.putText(t1, "East", (t1.shape[1]-40, t1.shape[0]/2), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 0, 0), 2)
    cv2.putText(t1, "North", (t1.shape[1]/2, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 0, 0), 2)
    cv2.putText(t1, "South", (t1.shape[1]/2, t1.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 0, 0), 2)



    cv2.putText(t1, direction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
        1.55, (0, 0, 255), 3)
    cv2.putText(t1, "dx: {}, dy: {}".format(dX, dY),
        (10, t1.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    return t1

def getCameraRead():
    (grabbed, frame) = camera.read()
    return cv2.flip(frame, 1)

for i in range(1,30):
    frame1 = getCameraRead()

while True:

    #(grabbed, frame1) = camera.read()
    frame2 = getCameraRead()
    frame = grepObject(frame1, frame2)

    cv2.imshow("Frame", frame)
    #cv2.imwrite("frame-"+str(int(time.time()))+".jpg", frame)
    key = cv2.waitKey(1) & 0xFF

    if(noContoursFoundAccu>30 and contoursNow == False):
        frame1 = getCameraRead()
        noContoursFoundAccu = 0

camera.release()
cv2.destroyAllWindows()
