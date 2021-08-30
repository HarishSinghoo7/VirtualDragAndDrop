import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

class CreateRectangle:
    colorR = (255, 0, 255)

    def __init__(self, center_pos, size=[200, 200]):
        self.centerPos = center_pos
        self.size = size

    def draw(self, img):
        cx, cy = self.centerPos
        w, h = self.size

        # Creating Rectangle ( img, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), self.colorR, cv2.FILLED)

class DragRectangle:
    def __init__(self, rect_obj: CreateRectangle):
        self.centerPos = rect_obj.centerPos
        self.size = rect_obj.size
        self.colorR = rect_obj.colorR

    def drag(self, img, detector):
        # drawing finger positions
        lmList, _ = detector.findPosition(img)
        colorR = self.colorR
        if lmList:
            cx, cy = self.centerPos
            w, h = self.size

            # find distance b/w index finger tip landmark (8) and middle finger tip landmark (12)
            l, _, _ = detector.findDistance(8, 12, img)
            # Finger positions from media pipeline docs

            if l < 50:
                cursor = lmList[8]
                # cursor[0] is x-axis cursor[1] is y-axis
                if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
                    colorR = 0, 255, 0
                    self.centerPos = cursor
        return self.centerPos, self.size, colorR


# detectionCon (Detection Confidence default 0.5)
detector = HandDetector(detectionCon=0.8)
rects = []
for i in range(5):
    rects.append(CreateRectangle([250 * i + 100, 100], [200, 200]))

while True:
    success, img = cap.read()

    # Flip the image
    img = cv2.flip(img, 1)

    # finding hands on captured frame
    img = detector.findHands(img)
    for rect in rects:
        rect.centerPos, rect.size, rect.colorR = DragRectangle(rect).drag(img, detector)
        rect.draw(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
