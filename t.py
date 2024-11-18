import numpy as np
import cv2 as cv


class Slam:
    def __init__(self):
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        self.last = None

    def process_frame(self, frame):
        # preprocess
        frame = cv.resize(frame, (1920 // 2, 1080 // 2))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detection
        corners = cv.goodFeaturesToTrack(frame, 3000, 0.01, 3)
        kps = [cv.KeyPoint(corner[0][0], corner[0][1], 20.0) for corner in corners]

        # extracting
        kps, des = self.orb.compute(frame, kps)

        # matching
        good = None
        if self.last is not None:
            matches = self.matcher.knnMatch(des, self.last['des'], k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        # drawing
        out_frame = cv.drawKeypoints(frame, kps, None, color=(0, 255, 0))
        if good is not None:
            for m in good:
                ptf2pt = lambda pt: (int(pt[0]), int(pt[1]))
                out_frame = cv.line(out_frame, ptf2pt(kps[m.queryIdx].pt), ptf2pt(self.last['kps'][m.trainIdx].pt), (255,0,0), 1)

        # updating last
        self.last = { "kps": kps, "des": des }
        
        return out_frame

cap = cv.VideoCapture('road.mp4')

slam = Slam()

while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = slam.process_frame(frame)
    cv.imshow('video', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
