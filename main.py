import cv2
import mediapipe as mp
# import win32api

cap = cv2.VideoCapture(0) # Подключение к web-камере
mp_Hands = mp.solutions.hands # говорим, что хотим, а хотим мы распозновать руки
hands = mp_Hands.Hands(max_num_hands = 10) # характеристика для распознования
mpDraw = mp.solutions.drawing_utils # инициализация утилиты для рисования
fingersCoord = [(8, 6), (12, 10), (16, 14), (20, 18)] # ключевые точки всех пальцев, кроме большого
thumbCoord = (4, 2) # ключевая точка для большого пальца

while cap.isOpened(): # пока камера "работает"
    success, image = cap.read() # получение кадра с камеры
    if not success: # если не удалось получить кадр
        # win32api.MessageBox(0, 'Ошибка!', 'Не удалось получить кадр с web-камеры'
        print('Не удалось получить кадр с web-камеры')
        continue
    image = cv2.flip(image, 1) # зеркально отражаем изображение
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # меняем кодировку изображения
    result = hands.process(RGB_image) # ищем руки на изображении
    multiLandMarks = result.multi_hand_landmarks # извлекаем коллекцию (список) найденных рук
    upCount = 0 # сколько поднятых пальцев
    downCount = 0 # сколько опущенных пальцев
    h, w, c = image.shape
    if multiLandMarks: # если коллекция не пустая
        for idx, handLms in enumerate(multiLandMarks):
            label = result.multi_handedness[idx].classification[0].label
            rus = "Левая" if label == "Left" else "Правая" if label == "Right" else "Неизвестно"
            print (rus)

            mpDraw.draw_landmarks(image, handLms, mp_Hands.HAND_CONNECTIONS)
            fingersList = []
            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                fingersList.append((x, y))
            
            for coord in fingersCoord:
                if fingersList[coord[0]][1] < fingersList[coord[1]][1]:
                    upCount += 1
                else:
                    downCount += 1

            if fingersList[thumbCoord[0]][0] < fingersList[thumbCoord[1]][0] or fingersList[thumbCoord[0]][0] > fingersList[thumbCoord[1]][0]:
                upCount += 1
            else:
                downCount += 1
    
    cv2.putText(image, "Up fingers: " + str(upCount), (100, 150), cv2.FONT_ITALIC, (w + h) // 560, (0, 255, 0), (w + h) // 560)
    cv2.putText(image, "Down fingers: " + str(downCount), (100, 250), cv2.FONT_ITALIC, (w + h) // 560, (0, 255, 0), (w + h) // 560)
    cv2.imshow('web-cam', image) # показываем изображение

    if cv2.waitKey(1) & 0xFF == 27: # Ожидаем нажатие ESC
        break

cap.release()