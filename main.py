import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class Laboratoria_1:
    def __init__(self, image):
        self.image = cv.resize(image, (300, 300))

    def channel_red(self):
        red = self.image.copy()
        red[:, :, 0] = 0
        red[:, :, 1] = 0
        return cv.imshow("ra", red)

    def channel_green(self):
        green = self.image.copy()
        green[:, :, 0] = 0
        green[:, :, 2] = 0
        return cv.imshow("ra", green)

    def channel_blue(self):
        blue = self.image.copy()
        blue[:, :, 2] = 0
        blue[:, :, 1] = 0
        return cv.imshow("ra", blue)


class Laboratoria_2(Laboratoria_1):

    def yuv(self):
        width, height, channel = self.image.shape
        yuv_image = np.zeros((height, width, channel), np.uint8)
        for i in range(width):
            for j in range(height):
                r = self.image.item(i, j, 0)
                g = self.image.item(i, j, 1)
                b = self.image.item(i, j, 2)

                y = b * .144 + .587 * g + .299 * r
                u = .493 * (b - y)
                v = .877 * (r - y)
                yuv_image.itemset((i, j, 0), y)
                yuv_image.itemset((i, j, 1), u)
                yuv_image.itemset((i, j, 2), v)
        return yuv_image

    def gray_brg(self):
        w, h, ch = self.image.shape
        gray_img = np.zeros((w, h, ch), np.uint8)
        for i in range(w):
            for j in range(h):
                r = self.image.item(i, j, 0)
                g = self.image.item(i, j, 1)
                b = self.image.item(i, j, 2)
                gray = (r + g + b) / 3
                gray_img.itemset((i, j, 0), gray)
                gray_img.itemset((i, j, 1), gray)
                gray_img.itemset((i, j, 2), gray)
        return gray_img

    def gray_yuv(self):
        w, h, ch = self.image.shape
        img1 = np.zeros((h, w, ch), dtype='uint8')
        for i in range(w):
            for j in range(h):
                img1[i, j] = self.image[i, j, 0] * .114 + .587 * self.image[i, j, 1] + .299 * self.image[i, j, 2]
        return img1

    def histogram_for_gray(self,image):
        w, h, ch = image.shape
        hist_img = cv.calcHist(image, [0], None, [256], [0, 256])
        cv.normalize(hist_img, hist_img, alpha=0, beta=h, norm_type=cv.NORM_MINMAX)
        plt.plot(hist_img)
        return plt.show()

    def histogram_for_color_img(self,image):
        b, g, r = cv.split(image)
        plt.hist(b.ravel(), 256, [0, 256], density=True)
        plt.hist(g.ravel(), 256, [0, 256], density=True)
        plt.hist(r.ravel(), 256, [0, 256], density=True)
        return plt.show()


class Laboratoria_3(Laboratoria_2):
    def __init__(self, image, first_thresh, second_thresh):
        super().__init__(image)
        self.image = cv.resize(cv.cvtColor(image, cv.COLOR_BGR2GRAY),(300,300))
        self.first_thresh = first_thresh
        self.second_thresh = second_thresh

    def threshold(self):
        w, h = self.image.shape
        pixels = np.zeros((w, h), np.uint8)

        for i in range(w):
            for j in range(h):
                if self.first_thresh < self.image[i][j] < self.second_thresh:
                    pixels[i][j] = 255
                else:
                    pixels[i][j] = 0
        return cv.imshow("binary image",pixels)


class Laboratoria_4():
    def __init__(self, img_1, img_2):
        self.img_1 = img_1
        self.img_2 = img_2

    def new(self):
        new_img = np.zeros_like(self.img_1)
        h, w = self.img_1.shape
        result = 0
        for i in range(w):
            for j in range(h):
                result =  self.img_1[i, j] - self.img_2[i, j]
                if result < 0:
                    result = 0
                elif result > 255:
                    result = 255
                new_img[i, j] = result
        return cv.imshow("result sub image",new_img)


class Laboratoria_5(Laboratoria_2):

    def blurring(self, size):
        width, height, channel = self.image.shape
        new_img = np.zeros_like(self.image)
        neighbor = size // 2
        for i in range(width):
            for j in range(height):
                for k in range(channel):
                    count = 0
                    for l in range(max(0, i - neighbor), min(width - 1, i + neighbor) + 1):
                        for m in range(max(0, j - neighbor), min(height - 1, j + neighbor) + 1):
                            count += self.image[l][m][k]
                    new_img[i][j][k] = count / (size ** 2)
        return new_img

    def kernel_loop(self, size, kernel):
        w, h, ch = self.image.shape
        new_image = np.zeros_like(self.image)
        """ ile ma faktycznie ma byc sasiadow """
        neighbor = size // 2

        for i in range(w):
            for j in range(h):
                for k in range(ch):
                    result = 0
                    for l in range(max(0, i - neighbor), min(w - 1, i + neighbor) + 1):
                        for m in range(max(0, j - neighbor), min(h - 1, j + neighbor) + 1):
                            x = l + neighbor - i
                            y = m + neighbor - j
                            values = kernel[x][y]
                            result += self.image[l][m][k] * values
                    if result < 0:
                        result = 0
                    elif result > 255:
                        result = 255

                    new_image[i][j][k] = result
        return new_image

    def kernel_menu(self, num_kernel):
        match num_kernel:

            case 0:
                # "filtr górno przepustowy"
                filter = Laboratoria_5.kernel_loop(self,3, np.array([[-1, -1, -1],
                                                                            [-1, 9, -1],
                                                                            [-1, -1, -1]]))

                return filter

            case 1:
                # "filtr dolno przepustowy"

                filter = Laboratoria_5.kernel_loop(self,3, np.array([[1, 1, 1],
                                                                            [1, 1, 1],
                                                                            [1, 1, 1]]))
                return filter

            case 2:
                # "filtr ukosny"

                filter = Laboratoria_5.kernel_loop(self,3, np.array([[-1, 0, 0],
                                                                            [0, 1, 0],
                                                                            [0, 0, 0]]))
                return filter

            case 3:
                # "filtr poziomy"
                filter = Laboratoria_5.kernel_loop(self,3, np.array([[0, 0, 0],
                                                                            [-1, 1, 0],
                                                                            [0, 0, 0]]))
                return filter

            case 4:
                # "filtr pionowy"
                filter = Laboratoria_5.kernel_loop(self,3, np.array([[0, -1, 0],
                                                                            [0, 1, 0],
                                                                            [0, 0, 0]]))
                return filter

            case 5:
                "filtr sobla"
                filter = Laboratoria_5.kernel_loop(self,3, np.array([[1, 2, 1],
                                                                            [0, 0, 0],
                                                                            [-1, -2, -1]]))
                return filter
            case 6:
                "ten tez sobel"
                filter = Laboratoria_5.kernel_loop(self,3, np.array([[1, 0, -1],
                                                                            [2, 0, -2],
                                                                            [1, 0, -1]]))
                return filter

            case _:
                print("I don't have anymore kernels :D")

    def filter_edges(self, image1, image2, image3):
        w, h, ch = image1.shape
        new_img = np.zeros_like(image1)
        for i in range(w):
            for j in range(h):
                for k in range(ch):
                    new_img[i][j][k] = np.sqrt((image1[i][j][k] ** 2) + (image2[i][j][k] ** 2) + (image3[i][j][k] ** 2))
        return new_img


class Laboratoria_6(Laboratoria_5):

    def translation(self):
        w, h, ch = self.image.shape
        new_img = np.zeros_like(self.image)
        tx, ty = w / 2, h / 2
        matrix = np.array([[1, 0, tx],
                           [0, 1, ty]], dtype="uint8")

        for i in range(w):
            for j in range(h):
                xy = np.array([i, j, 1])
                # np.dot to mnozenie cauchy'ego gdzie moge zapisac to jako tuple odrazu w tym przpadku odrazu ją rozpakuje
                x, y = np.dot(matrix, xy)
                if 0 < x < w and 0 < y < h:
                    new_img[x][y] = self.image[i][j]
        return cv.imshow("translation",new_img)

    def rotation(self):
        width, height, channel = self.image.shape
        new_img = np.zeros_like(self.image)
        teta = np.radians(45)
        center_x, center_y = width / 2, height / 2
        matrix = np.array([[np.cos(teta), -np.sin(teta), center_x * (1 - np.cos(teta) + center_y * np.sin(teta))],
                           [np.sin(teta), np.cos(teta), center_y * (1 - np.cos(teta) - center_x * np.sin(teta))]])
        for i in range(width):
            for j in range(height):
                xy = np.array([i, j, 1])
                rot_matrix = np.dot(matrix, xy)
                rot_matrix = rot_matrix.astype(np.int8)
                x, y = rot_matrix
                if 0 < x < width and 0 < y < height:
                    new_img[x][y] = self.image[i][j]

        return cv.imshow("rotation",new_img)


if __name__ == "__main__":
    image = cv.imread("ra.jpg")
    img_org = cv.imread("img_org.png",0)
    img_sub = cv.imread("img_sub.png",0)
    zad_1 = Laboratoria_1(image)
    zad_1.channel_red()
    cv.waitKey(0)
    zad_1.channel_green()
    cv.waitKey(0)
    zad_1.channel_blue()
    cv.waitKey(0)
    zad_2 = Laboratoria_2(image)
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    hsv = cv.resize(hsv,(300,300))
    yuv = zad_2.yuv()
    yuv_hsv = np.hstack([hsv,yuv])
    cv.imshow("yuv and hsv",yuv_hsv)
    cv.waitKey(0)
    gray_yuv = zad_2.gray_yuv()
    gray_brg = zad_2.gray_brg()
    grays = np.hstack([gray_yuv,gray_brg])
    cv.imshow("gray yuv and brg",grays)
    cv.waitKey(0)
    zad_2.histogram_for_gray(gray_yuv)
    zad_2.histogram_for_gray(gray_brg)
    zad_2.histogram_for_color_img(yuv)
    zad_2.histogram_for_color_img(hsv)
    zad_2.histogram_for_color_img(image)
    zad_3 = Laboratoria_3(image,127,255)
    zad_3.threshold()
    cv.waitKey(0)
    zad_4 = Laboratoria_4(img_org,img_sub)
    zad_4.new()
    cv.waitKey(0)
    zad_5 = Laboratoria_5(image)
    upper_bandwidth = zad_5.kernel_menu(0)
    bottom_bandwidth = zad_5.kernel_menu(1)
    filter_zigzag = zad_5.kernel_menu(2)
    new_img = zad_5.blurring(3)
    filter_horizontally = zad_5.kernel_menu(4)
    filter_vertically = zad_5.kernel_menu(3)
    sobel_1 = zad_5.kernel_menu(5)
    sobel_2 = zad_5.kernel_menu(6)
    sobel_stack = np.hstack([sobel_1, sobel_2])
    filter_stack = np.hstack([filter_horizontally, filter_zigzag, filter_vertically])
    kernel_stack = np.hstack([bottom_bandwidth, upper_bandwidth])
    image = cv.resize(image,(300,300))
    blurring_stack = np.hstack([image, new_img])
    filter_edges = zad_5.filter_edges(filter_zigzag, filter_vertically, filter_horizontally)
    stack1 = np.hstack([sobel_stack, filter_stack])
    stack2 = np.hstack([kernel_stack, blurring_stack])
    cv.imshow("sobel i filtry ukosny,poziomy,pionowy", stack1)
    cv.waitKey(0)
    cv.imshow("dolno przepustowy,gorno przepustowy,zwykly,rozmycie", stack2)
    cv.waitKey(0)
    zad_6 = Laboratoria_6(image)
    zad_6.translation()
    cv.waitKey(0)
    zad_6.rotation()
    cv.waitKey(0)
    cv.destroyAllWindows()
