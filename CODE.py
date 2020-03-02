from PIL import *
from PIL import Image
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFilter
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance


# myLabels=[]

def main():
    img = Image.open('C:\\Users\\User\\Desktop\\numbers.jpg')
    img_gray = img.convert('L')  # converts the image to grayscale image
    img.show()
    ONE = 150
    zero = 0
    a = np.asarray(img_gray)  # from PIL to np array
    a_bin = threshold(a, 100, ONE, zero)
    labels, label = blob_coloring_8_connected(a_bin, ONE)
    for a in range(len(labels)):
        for b in range(len(labels[0])):
            print(labels[a][b], end=" ")
        print()

    mini, minj, maxi, maxj, label_set = find_rectangle(labels)
    print(mini)
    new_img2 = np2PIL_color(label)
    for x in range(len(mini)):
        if (mini[x] < 1000):
            shape = [(minj[x], mini[x]), (maxj[x], maxi[x])]
            rect = ImageDraw.Draw(new_img2)
            rect.rectangle(shape, outline="red")
    new_img2.show()
## Ekledigim yer :
    for x in range(len(mini)):
         if (mini[x] < 1000):
             cropped = new_img2.crop((minj[x], mini[x],maxj[x], maxi[x]))
             cropped = cropped.resize((21, 21))
             cropped.show()


def binary_image(nrow, ncol, Value):
    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow, ncol))

    x0, y0, r0 = 30, 30, 10
    x1, y1, r1 = 70, 30, 10

    for i in range(50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i - 20][90 - i + 1] = 1
        mask_lines[i - 20][90 - i + 2] = 1
        mask_lines[i - 20][90 - i + 3] = 1

    # mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute(x - x1), np.absolute(y - y1)) <= r1
    # mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    # mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    # mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    # imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value
    # imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge


def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(im, 'RGB')
    return img


def np2PIL_color(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape=im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    print("nrow, ncol", nrow, ncol)
    im = np.zeros(shape=(nrow, ncol), dtype=int)
    a = np.zeros(shape=max_label, dtype=int)
    a = np.arange(0, max_label, dtype=int)
    color_map = np.zeros(shape=(max_label, 3), dtype=np.uint8)
    color_im = np.zeros(shape=(nrow, ncol, 3), dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, 1, dtype=np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label - 1
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            c = bim[i][j]
            l = bim[i][j - 1]
            u = bim[i - 1][j]
            label_u = im[i - 1][j]
            label_l = im[i][j - 1]
            label_ul = im[i - 1][j - 1]
            label_ur = im[i - 1][j + 1]
            im[i][j] = max_label - 1
            if c == ONE:
                min_label = min(label_u, label_l, label_ul, label_ur)
                if min_label == max_label - 1:
                    k += 1
                    im[i][j] = k
                else:
                    im[i][j] = min_label
                    if min_label != label_u and label_u != max_label - 1:
                        update_array(a, min_label, label_u)

                    if min_label != label_l and label_l != max_label - 1:
                        update_array(a, min_label, label_l)

                    if min_label != label_ur and label_ur != max_label - 1:
                        update_array(a, min_label, label_ur)

                    if min_label != label_ul and label_ul != max_label - 1:
                        update_array(a, min_label, label_ul)

            else:
                im[i][j] = max_label - 1
    # final reduction in label array
    for i in range(k + 1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    # second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label - 1:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]
    return im, color_im


def update_array(a, label1, label2):
    index = lab_small = lab_large = 0
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else:  # a[index] == lab_small
            break

    return


def drawing_image():
    w, h = 220, 190
    shape = [(40, 40), (w - 10, h - 10)]

    # creating new Image object
    img = Image.new("RGB", (w, h))

    # create rectangle image
    img1 = ImageDraw.Draw(img)
    img1.rectangle(shape, fill="#ffff33", outline="red")

    img.show()


def find_rectangle(im):
    nrow = im.shape[0]
    ncol = im.shape[1]

    k_max = 0
    for i in range(nrow):
        for j in range(ncol):
            if im[i][j] == 10000 - 1:
                continue
            myLabel = im[i][j]
            if myLabel > k_max:
                k_max = myLabel

    mini = np.full(k_max + 1, np.inf)
    minj = np.full(k_max + 1, np.inf)
    maxi = np.zeros(k_max + 1)
    maxj = np.zeros(k_max + 1)
    label_set = set()
    for i in range(nrow):
        for j in range(ncol):
            if im[i][j] == 10000 - 1:
                continue
            myLabel = im[i][j]
            label_set.add(myLabel)
            if i < mini[myLabel]:
                mini[myLabel] = i
            if i > maxi[myLabel]:
                maxi[myLabel] = i
            if j < minj[myLabel]:
                minj[myLabel] = j
            if j > maxj[myLabel]:
                maxj[myLabel] = j
 

    return mini, minj, maxi, maxj, label_set,  # output


if __name__ == '__main__':
    main()