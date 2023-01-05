# данный файл является модификацией исходного
# потому что библиотеки в исходном проекте старые и при запуске вылезает много ошибок
# После различных исправлений скрипт запускается, НО его точность страдает
from pylab import *
from matplotlib.pyplot import imshow
import numpy as np
import os
import cv2
import warnings

warnings.simplefilter("ignore")


def dist(p1, p2):           # расстояние между точками
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def pers_trans(img, coords):    # преобразование для вырезания
    pts1 = np.float32(coords)
    a = coords[0]
    b = coords[1]
    d = coords[3]
    dist1 = dist(a, b)
    dist2 = dist(a, d)
    if dist1 > dist2:
        pts2 = np.float32([[dist1, dist2], [0, dist2], [0, 0], [dist1, 0]])
        shape = (int(dist1), int(dist2))
    else:
        pts2 = np.float32([[dist2, 0], [dist2, dist1], [0, dist1], [0, 0]])
        shape = (int(dist2), int(dist1))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, shape)
    return dst


def draw_rec(cnt):          # определение координат прямоугольника вокруг контура
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def find(lista, ekstr, ktory):           # поиск крайних вершин
    index = 0
    if ekstr == 'max':
        for i in range(len(lista)):
            if lista[index][ktory] < lista[i][ktory]:
                index = i
    else:
        for i in range(len(lista)):
            if lista[index][ktory] > lista[i][ktory]:
                index = i

    return lista[index]


def corners(points):            # определение крайних вершин
    a = find(points, 'max', 0)
    b = find(points, 'max', 1)
    c = find(points, 'min', 0)
    d = find(points, 'min', 1)
    return [a, b, c, d]


def fun(img, canny_first, canny_second, coef_blur, count_morph):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # предварительная обработка изображения для обнаружения контуров

    imgray = cv2.medianBlur(imgray, coef_blur)
    imgray = cv2.Canny(imgray, canny_first, canny_second)

    kernel = np.ones((5, 5), np.uint8)
    imgray = cv2.dilate(imgray, kernel, iterations=count_morph)
    imgray = cv2.erode(imgray, kernel, iterations=count_morph)

    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def crop(img, canny_first, canny_second, coef_blur, count_morph, size):
    contours, hierarchy = fun(img, canny_first, canny_second, coef_blur, count_morph)
    hier = hierarchy[0]                          # основная функция для вырезания флага из фотографии

    points = []
    boxes = []

    for i in range(len(hier)):
        if len(contours[i]) > size and hier[i][3] != -1:
            cnt = contours[i]
            rec = draw_rec(cnt)
            boxes.append(rec)
            for p in rec:
                points.append(p)

    coords = corners(points)
    img = pers_trans(img, coords)
    return img, boxes


def adjust_gamma(image, gamma=1.0):         # гамма-коррекция
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def match(flag, n):           # поиск наилучшего соответствия
    rows_flag, cols_flag = flag.shape[:2]
    fit = []
    for i in range(1, n+1):
        print('baza ' + str(i) + ': ')
        base = cv2.imread(os.getcwd() + '//flags/base/' + str(i) + '.png')
        base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        rows_base, cols_base = base.shape[:2]
        if rows_flag < rows_base:
            x = rows_flag
        else:
            x = rows_base
        if cols_flag < cols_base:
            y = cols_flag
        else:
            y = cols_base
        flag = float64(cv2.resize(flag, (y, x)))
        base = float64(cv2.resize(base, (y, x)))

        fit.append(sum(abs((flag-base))))
        print(fit[-1])

    best_fit = min(fit)
    print('\nbest: ')
    print(best_fit)
    idx = fit.index(best_fit)
    idx = idx + 1
    return idx


# MAIN
fig = figure(figsize=(20, 10))

# Тут пишем путь до нашего изображения
random = cv2.imread(os.getcwd() + '//flags/random/6.jpg')
random = cv2.cvtColor(random, cv2.COLOR_RGB2BGR)

subplot(1, 2, 1)

#error
result, box = crop(random, 100, 200, 3, 1, 100)

result = adjust_gamma(result, 0.7)

imshow(random)

idx = match(result, 11)
base = cv2.imread(os.getcwd() + '//flags/base/' + str(idx) + '.png')
base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

subplot(1, 2, 2)
imshow(base)

fig.savefig('flags.pdf')
