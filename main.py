# Code to extract table from the image
# Bank ke templates alag alag chahiye honge

import cv2
import numpy as np
import pandas as pd
import pytesseract
import glob
from ocr_detection.pdf_to_image import image_conversion
from ocr_detection.convert_to_df import df_conversion
from ocr_detection.generate_csv import csv_conversion

# Method not useful for dubai account images
def find_tables_1(image):

    edges = cv2.Canny(image,10,100,apertureSize=3, L2gradient =True)
    cv2.imwrite("edges.png", edges)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold = 50, minLineLength=300, maxLineGap = 1)
    # print(lines)

    if lines is not None:
        x_diff = []
        for i in range(len(lines)):
            l = lines[i][0]
            x1, y1, x2, y2 = l
            x_diff.append(abs(x2-x1))
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),5)

    ind = x_diff.index(max(x_diff))
    line1 = lines[ind][0]

    ## search for line 2
    y_diff = []
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        y_diff.append(abs(line1[1]-y1))

    ind = y_diff.index(max(y_diff))
    line2 = lines[ind][0]


    cv2.rectangle(image, (line1[0], line1[1]), (0,image.shape[1]), (0,255,0), 2)
    table_image = image[line2[3]:line1[1], line1[0]:line1[2]]

    return table_image

# Method useful and generalized version
def find_tables_2(image):

    BLUR_KERNEL_SIZE = (17, 17)
    STD_DEV_X_DIRECTION = 0
    STD_DEV_Y_DIRECTION = 0
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -2

    img_bin = cv2.adaptiveThreshold(
        ~blurred,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )

    vertical = horizontal = img_bin.copy()
    SCALE = 5
    image_width, image_height = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_height / SCALE)))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)

    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))

    mask = horizontally_dilated + vertically_dilated
    # cv2.imwrite("mask.png",mask)
    # cv2.imwrite("vertical.png",vertically_dilated)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    MIN_TABLE_AREA = 1e5
    contours = [c for c in contours if cv2.contourArea(c) > MIN_TABLE_AREA]
    # cv2.drawContours(image, contours, -1, (0,0,0),10)
    # cv2.imwrite("contours.png", image)
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.1 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]

    """
    Refer below link to understand this method
    Link: https://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
    """

    table_images = [image[y:y+h, x:x+w] for x, y, w, h in bounding_rects]
    return table_images, vertically_dilated


# To retrive the data from the input images
def OCR(table_image):

    ## For operations on the column images from the segmented table

    # ret,thresh = cv2.threshold(table_image,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imwrite("thresh.png", thresh)

    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    data = pytesseract.image_to_data(table_image, lang='eng', config='--psm 6')
    return data


# This function will break the image into columns and return the corresponding processed data
def data_preprocess(image, vertical):

    contours, hierarchy = cv2.findContours(vertical,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # image = cv2.drawContours(vertical, contours, -1, (255,0,0), 5)

    x_val = []
    images = []
    for i in range(len(contours)):
        cnt = contours[i]
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        x_val.append(cx)

    x_val = sorted(x_val)
    for i in range(len(x_val)-1):
        images.append(image[0:,x_val[i]:x_val[i+1]])   ## 20 is to avoid arabic

    # To retrive row wise info from the column images because there are no row separating lines

    data_list = []
    for img in images:
        data = OCR(img)
        # print(data)

        str_dict = {}
        concate = ""
        count = 0
        prev_key = 0

        for i in data:

            if i =="\t":
                count+=1
                continue

            if count == 7:
                concate = concate + i
                key = concate

            if count == 8:
                try:
                    if int(prev_key) -6 <= int(key) <= int(prev_key) +6:
                        key = prev_key
                    continue
                except:
                    continue
                finally:
                    concate = ""
                continue

            if i == "\n":
                if len(concate)>1:
                    if key not in str_dict.keys():
                        str_dict[key] = concate
                    else:
                        str_dict[key] = str_dict[key] + "-" + concate

                concate = ""
                count=0
                prev_key = key

                continue

            if count == 11:
                concate = concate + i

        del str_dict["top"]
        data_list.append(str_dict)

    # print(data_list)
    # [{},{},{},{},{}...]                                      Second element (description column is the refernce column)

    final_dict = data_list.pop(1)

    for key, value in final_dict.items():
        final_dict[key] = [value]

        for dic in data_list:
            for key1, value1 in dic.items():
                if int(key1)-10 <= int(key) <= int(key1)+10:
                    key1 = key
                    final_dict[key].append(value1)
                    break
            else:
                final_dict[key].append("-")

    return final_dict

def main(pdf):

    no_images = image_conversion(pdf)
    list_of_df = []

    # Converting images to csv file
    for img in glob.glob(f'ocr_results/images/*[0-{no_images}].*'):
        image = cv2.imread(img,0)
        # image = cv2.resize(image, (800,600))

        table_images, _ = find_tables_2(image)
        _, vertical = find_tables_2(table_images[0])

        final_dict = data_preprocess(table_images[0], vertical)
        df = df_conversion(final_dict)
        list_of_df.append(df)

    csv_conversion(list_of_df)

if __name__=="__main__":
    main(pdf)
