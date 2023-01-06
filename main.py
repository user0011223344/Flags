# данный файл является модификацией исходного
# потому что библиотеки в исходном проекте старые и при запуске вылезает много ошибок
# После различных исправлений скрипт запускается, НО его точность страдает
from pylab import *
from matplotlib.pyplot import imshow
import numpy as np
import os
import cv2
import warnings
import argparse

warnings.simplefilter("ignore")


# Change me
INPUT_FILENAME = "/flags/input_img/6.jpg"

OUTPUT_PDF_FILENAME = "flags.pdf"



def dist(p1, p2):           # расстояние между точками
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def perspective_transformation(img, corner_coords):    # преобразование для вырезания
    pts1 = np.float32(corner_coords)
    a = corner_coords[0]
    b = corner_coords[1]
    d = corner_coords[3]
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


def get_rectangle_coords(cnt):          # определение координат прямоугольника вокруг контура
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def find(coords_list, min_or_max_criteria, height_or_width):           # поиск крайних вершин
    index = 0
    if min_or_max_criteria == 'max':
        for i in range(len(coords_list)):
            if coords_list[index][height_or_width] < coords_list[i][height_or_width]:
                index = i
    else:
        for i in range(len(coords_list)):
            if coords_list[index][height_or_width] > coords_list[i][height_or_width]:
                index = i

    return coords_list[index]


def get_corners(coords_list):            # определение крайних вершин
    a = find(coords_list, 'max', 0)
    b = find(coords_list, 'max', 1)
    c = find(coords_list, 'min', 0)
    d = find(coords_list, 'min', 1)
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

# в этой функции извлекаются контуры из входной картинки
def crop(transformated_img, canny_first, canny_second, coef_blur, count_morph, size):
    contours, hierarchy = fun(transformated_img, canny_first, canny_second, coef_blur, count_morph) # основная функция для вырезания флага из фотографии
    hier = hierarchy[0]                          

    rectangle_coords_list = []
    boxes_or_list_with_rectangles_coords = []

    for i in range(len(hier)):
        if len(contours[i]) > size and hier[i][3] != -1:
            cnt = contours[i]
            rectangle_coords = get_rectangle_coords(cnt) # поиск ближайшего к контуру прямоугольника
            boxes_or_list_with_rectangles_coords.append(rectangle_coords)
            for p in rectangle_coords:
                rectangle_coords_list.append(p)

    corners_coords = get_corners(rectangle_coords_list)
    transformated_img = perspective_transformation(transformated_img, corners_coords)
    return transformated_img, boxes_or_list_with_rectangles_coords


def adjust_gamma(image, gamma=1.0):         # гамма-коррекция
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def search_best_match(input_img):           # поиск наилучшего соответствия среди всех etalon-файлов
    rows_in_input_img, cols_in_input_img = input_img.shape[:2]
    fit = []
    ETALON_FLAGS_DIR = os.getcwd()+'/flags/etalon_img/'
    etalon_filenames_list = []
    for f in os.listdir(ETALON_FLAGS_DIR):
        full_etalon_filepath = ETALON_FLAGS_DIR + f
        if os.path.isfile(full_etalon_filepath):
            etalon_filenames_list.append(full_etalon_filepath)

    for etalon_filename in etalon_filenames_list:
            #print('baza ' + str(i) + ': ')
            etalon_img = cv2.imread(etalon_filename)
            etalon_img = cv2.cvtColor(etalon_img, cv2.COLOR_RGB2BGR)
            rows_etalon_img, cols_etalon_img = etalon_img.shape[:2]

            if rows_in_input_img < rows_etalon_img:
                x = rows_in_input_img
            else:
                x = rows_etalon_img
            if cols_in_input_img < cols_etalon_img:
                y = cols_in_input_img
            else:
                y = cols_etalon_img

            input_img = cv2.resize(input_img, (y, x)).astype("float64")
            etalon_img = cv2.resize(etalon_img, (y, x)).astype("float64")

            fit.append(sum(abs((input_img-etalon_img))))
            print(f"Fit with { etalon_filename.split('Flags/flags/')[1] } = {fit[-1]}")

    best_fit = min(fit)
    found_idx = fit.index(best_fit)
    found_idx = found_idx + 1

    print(f'\nBest match: {best_fit}')
    print(f"Found etalon: {etalon_filenames_list[found_idx].split('Flags/flags/')[1]}")
    print(f"Input filename: {args.input_img}\n\n")
    return etalon_filenames_list[found_idx]


# MAIN

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_img',default=INPUT_FILENAME, required=False)

args = parser.parse_args()

# Создание выходного pdf-файла
output_pdf_file = figure(figsize=(20, 10))


input_img = cv2.imread(os.getcwd() + args.input_img)
# Конвертируем входную картинку в Blue Green Red (потому что opencv так работает)
input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

# раньше здесь была ошибка
cutted_and_transformated_img, list_of_boxes_coordinates = crop(input_img, 100, 200, 3, 1, 100)
cutted_and_transformated_img = adjust_gamma(cutted_and_transformated_img, 0.7)

# добавление строки столбцов в выходной pdf-файл
# кол-во строк = 1
# кол-во столбцов = 2
# индекс = 1
subplot(1, 2, 1)
# Добавить входную картинку в выходной pdf-файл
imshow(input_img)

# главная функция сравнения
found_etalon_filename = search_best_match(cutted_and_transformated_img)



found_img = cv2.imread(found_etalon_filename)
found_img = cv2.cvtColor(found_img, cv2.COLOR_RGB2BGR)

# добавление строки столбцов в выходной pdf-файл
# кол-во строк = 1
# кол-во столбцов = 2
# индекс = 2
subplot(1, 2, 2)
# Добавить эталонную картинку в выходной pdf-файл
imshow(found_img)

output_pdf_file.savefig('flags.pdf')
