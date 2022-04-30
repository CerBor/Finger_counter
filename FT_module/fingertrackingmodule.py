import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=0, detectionCon=0.75, trackingCon=0.75):
        self.mpHands = mp.solutions.hands  # говорим, что хотим распознавать руки
        self.hands = self.mpHands.Hands(mode, maxHands, complexity, detectionCon, trackingCon)  # характеристики для распознавания
        self.mpDraw = mp.solutions.drawing_utils # инициализация утилиты для рисования
        self.fingertips = [4, 8, 12, 16, 20] # кончики пальцев
        self.pointPosition = {}
    
    def findHands(self, img, draw=False):
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        self.result = self.hands.process(RGB_image)  # ищем руки на изображении
        img.flags.writeable = True
        if draw:
            multiLandMarks = self.result.multi_hand_landmarks  # извлекаем коллекцию (список) найденных рук
            upCount = 0
            if multiLandMarks:
                for handLms in multiLandMarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

    def findFingersPosition(self, img, draw=False):
        mhl = self.result.multi_hand_landmarks
        if mhl:
            for idx, handLms in enumerate(mhl):
                xList = []
                yList = []

                self.pointPosition[idx] = []
                for lm in handLms.landmark:
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    self.pointPosition[idx].append((x, y))
                    if draw:
                        xList.append(x)
                        yList.append(y)

                if len(xList) > 0 and len(yList) > 0:
                    offset = 20
                    xmin, xmax = (min(xList), max(xList))
                    ymin, ymax = (min(yList), max(yList))
                    cv2.rectangle(img, (xmin - offset, ymin - offset), (xmax + offset, ymax + offset), (0, 255 // 1.5, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while cap.isOpened():  # пока камера "работает"
        success, image = cap.read()  # получение кадра с камеры
        if not success:  # если не удалось получить кадр
            print('Не удалось получить кадр с web-камеры')
            continue  # возвращаемся к ближайшему циклу
        image = cv2.flip(image, 1)  # зеркально отражаем изображение
        prevTime = time.time()
        detector.findHands(image, True)
        detector.findFingersPosition(image, True)
        curTime = time.time()

        fps = round(1 / (curTime - prevTime), 1)
        h, w, c = image.shape
        cv2.putText(image, f"FPS: {fps}", (w // 2, h // 6), cv2.FONT_ITALIC, 2, (255, 255, 255), 2)

        cv2.imshow('web-cam', image)

        if cv2.waitKey(1) & 0xFF == 27:  # Ожидаем нажатие ESC 
            break

main()
