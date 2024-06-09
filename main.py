import numpy as np
import pygetwindow as gw
import mss
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import threading
import winsound

# Параметры для анализа движения
MIN_CONTOUR_AREA = 65
FRAME_DIFF_THRESHOLD = 39
BLUR_SIZE = (21, 21)
DILATE_ITERATIONS = 2

# Параметры для уведомлений
ALERT_SOUND_FREQUENCY = 1000  # Частота звукового сигнала
ALERT_SOUND_DURATION = 1000  # Длительность звукового сигнала в миллисекундах


def get_window_image(window_title):
    try:
        # Найдите окно по названию
        window = gw.getWindowsWithTitle(window_title)[0]

        # Получаем координаты окна
        left, top, right, bottom = window.left, window.top, window.right, window.bottom
        width = right - left
        height = bottom - top

        # Захват экрана с помощью mss
        with mss.mss() as sct:
            monitor = {"top": top+40, "left": left+20, "width": width-40, "height": height-50}
            img = sct.grab(monitor)
            img = Image.frombytes('RGB', (img.width, img.height), img.rgb)

        # Возвращаем изображение
        return np.array(img)

    except Exception as e:
        print(f"Ошибка захвата изображения окна: {e}")
        return None


# Функция для проигрывания звукового сигнала
def play_alert_sound():
    winsound.Beep(ALERT_SOUND_FREQUENCY, ALERT_SOUND_DURATION)


# Укажите название окна, которое хотите захватывать
window_title = "DayZ"
windows = gw.getWindowsWithTitle(window_title)
if not windows:
    print("Окно игры не найдено!")
    exit()

# Инициализация переменных
prev_frame = None
motion_detected = False


# Создаем функцию обновления изображения
def update_frame(frame):
    global prev_frame, motion_detected

    # Захватываем изображение из указанного окна
    img = get_window_image(window_title)
    if img is None:
        return

    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применяем размытие для снижения уровня шума
    gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

    # Если это первый кадр, инициализируем prev_frame
    if prev_frame is None:
        prev_frame = gray
        return im,

    # Вычисляем разницу между текущим и предыдущим кадром
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, FRAME_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # Увеличиваем область различий
    thresh = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)

    # Находим контуры изменений
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем прямоугольники вокруг обнаруженных контуров
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    # Если обнаружено движение и уведомление не было отправлено, воспроизводим звуковой сигнал
    if motion_detected:
        motion_detected = False
        threading.Thread(target=play_alert_sound).start()

    # Обновляем предыдущий кадр
    prev_frame = gray

    # Обновляем изображение в анимации
    im.set_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return im,


# Создаем объект анимации
fig = plt.figure()
plt.axis('off')  # Отключаем оси координат
im = plt.imshow(get_window_image(window_title))

# Запускаем анимацию с явным указанием параметра cache_frame_data=False
ani = animation.FuncAnimation(fig, update_frame, interval=50, blit=True, cache_frame_data=False)


# Отображаем анимацию
plt.show()
