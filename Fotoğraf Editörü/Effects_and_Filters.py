import cv2
import imageio
import numpy as np
import math
from copy import copy
from scipy import ndimage
from scipy import misc
import scipy.interpolate as inter
import numpy as np
import pilgram
from pilgram import css
from pilgram import util
from PIL import Image
import random


class Effect_and_Filters:

    def __init__(self, image):
        self.to_be_changed_image = image
        self.filters_effects_dict = {}
        self.run()

    def run(self):
        self.ying_yang()
        self.corrupted_grey()
        self.broken_television()
        self.anime_isnt_cartooon()
        self.back_in_black()
        self.breaking_reality()
        self.roses_are_red()
        self.violets_are_blue()
        self.sugar_is_sweet()
        self.gouache_natürmort()
        self.fifty_shades_of_grey()
        self.martini_with_an_olive()
        self.white_russian()
        #self.pina_colada()
        #self.long_island_iced_tea()
        self.faded_leaf()
        self.lightbringer()
        self.ethernal_sunshine()
        self.winter_is_coming()
        #self.black_hole()
        self.emotion_yellowrock()
        self.is_so_cute()
        self.color_passion()
        self.yellow_hot_chick()

        self.limbo_travellers()
        self.mordor()
        self.gondor()
        self.matrix()
        self.eric_clapton()
        return self.filters_effects_dict

    def ying_yang(self):
        grayImage = cv2.cvtColor(self.to_be_changed_image, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        image1 = cv2.GaussianBlur(blackAndWhiteImage, (5, 5), 0)
        self.filters_effects_dict["Ying Yang"] = image1

    def corrupted_grey(self):
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(self.to_be_changed_image, kernel, iterations=1)
        gray_dil = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2GRAY)
        self.filters_effects_dict["Corrupted Grey"] = gray_dil

    def broken_television(self):
        # split the image into its BGR components
        (B, G, R) = cv2.split(self.to_be_changed_image)
        # find the maximum pixel intensity values for each
        # (x, y)-coordinate,, then set all pixel values less
        # than M to zero
        M = np.maximum(np.maximum(R, G), B)
        R[R < M] = 0
        G[G < M] = 0
        B[B < M] = 0
        # merge the channels back together and return the image
        self.filters_effects_dict["Broken Television"] = cv2.merge([B, G, R])

    def anime_isnt_cartooon(self):
        gray = cv2.cvtColor(self.to_be_changed_image, cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
        self.filters_effects_dict["Anime isn't Cartoon"] = edges

    def back_in_black(self):
        rows, cols = self.to_be_changed_image.shape[:2]
        # Create a Gaussian filter
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        filter = 255 * kernel / np.linalg.norm(kernel)
        vintage_im = np.copy(self.to_be_changed_image)
        # for each channel in the input image, we will apply the above filter
        for i in range(3):
            vintage_im[:, :, i] = vintage_im[:, :, i] * filter
        self.filters_effects_dict["Back in Black"] = vintage_im

    def VignetteFilter(self):
        input_image = cv2.resize(self.to_be_changed_image, (480, 480))

        # Extracting the height and width of an image
        rows, cols = input_image.shape[:2]

        # generating vignette mask using Gaussian
        # resultant_kernels
        X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)

        # generating resultant_kernel matrix
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

        # creating mask and normalising by using np.linalg
        # function
        mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
        output = np.copy(input_image)

        # applying the mask to each channel in the input image
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask
        self.filters_effects_dict["Vignette"] = output

    def breaking_reality(self):

        edges1 = cv2.bitwise_not(
            cv2.Canny(self.to_be_changed_image, 100, 200))  # for thin edges and inverting the mask obatined
        gray = cv2.cvtColor(self.to_be_changed_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)  # applying median blur with kernel size of 5
        dst = cv2.edgePreservingFilter(self.to_be_changed_image, flags=2, sigma_s=64,
                                       sigma_r=0.25)  # you can also use bilateral filter but that is slow
        # flag = 1 for RECURS_FILTER (Recursive Filtering) and 2 for  NORMCONV_FILTER (Normalized Convolution). NORMCONV_FILTER produces sharpening of the edges but is slower.
        # sigma_s controls the size of the neighborhood. Range 1 - 200
        # sigma_r controls the how dissimilar colors within the neighborhood will be averaged. A larger sigma_r results in large regions of constant color. Range 0 - 1
        cartoon1 = cv2.bitwise_and(dst, dst, mask=edges1)  # adding thin edges to smoothened image
        self.filters_effects_dict["Breaking Reality"] = cartoon1

    def exponential_function(self, channel, exp):
        table = np.array([min((i ** exp), 255) for i in np.arange(0, 256)]).astype(
            "uint8")  # creating table for exponent
        channel1 = cv2.LUT(channel, table)
        return channel1

    def tone(self, number):
        original = self.to_be_changed_image.copy()
        for i in range(3):
            if i == number:
                original[:, :, i] = self.exponential_function(original[:, :, i],
                                                              1.05)  # applying exponential function on slice
            else:
                original[:, :, i] = 0  # setting values of all other slices to 0
        return original

    def roses_are_red(self):
        red = self.tone(2)  # change second parameter to 0 for blue, 1 for green and 2 for red screen
        self.filters_effects_dict["Roses are Red"] = red

    def violets_are_blue(self):
        blue = self.tone(0)
        self.filters_effects_dict["Violets are Blue"] = blue

    def sugar_is_sweet(self):
        green = self.tone(1)  # change second parameter to 0 for blue, 1 for green and 2 for red screen
        self.filters_effects_dict["Sugar is Sweet"] = green

    def gouache_natürmort(self):
        dst_gray, dst_color = cv2.pencilSketch(self.to_be_changed_image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        image1 = cv2.bilateralFilter(dst_color, 9, 75, 75)
        self.filters_effects_dict["Gouache Natürmort"] = image1

    def hsv(self, l, u):
        hsv = cv2.cvtColor(self.to_be_changed_image, cv2.COLOR_BGR2HSV)
        lower = np.array([l, 128, 128])  # setting lower HSV value
        upper = np.array([u, 255, 255])  # setting upper HSV value
        mask = cv2.inRange(hsv, lower, upper)  # generating mask
        return mask

    def fifty_shades_of_grey(self):
        res = np.zeros(self.to_be_changed_image.shape, np.uint8)  # creating blank mask for result
        l = 15  # the lower range of Hue we want
        u = 30  # the upper range of Hue we want
        mask = self.hsv(l, u)
        inv_mask = cv2.bitwise_not(mask)  # inverting mask
        gray = cv2.cvtColor(self.to_be_changed_image, cv2.COLOR_BGR2GRAY)
        res1 = cv2.bitwise_and(self.to_be_changed_image, self.to_be_changed_image,
                               mask=mask)  # region which has to be in color
        res2 = cv2.bitwise_and(gray, gray, mask=inv_mask)  # region which has to be in grayscale
        for i in range(3):
            res[:, :, i] = res2  # storing grayscale mask to all three slices
        img1 = cv2.bitwise_or(res1, res)  # joining grayscale and color region
        self.filters_effects_dict["Fifty Shades of Grey"] = img1

    def spread_lookup_table(self, x, y):
        spline = inter.InterpolatedUnivariateSpline(x, y)
        return spline(range(256))

    def martini_with_an_olive(self):
        increaseLookupTable = self.spread_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = self.spread_lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
        red_channel, green_channel, blue_channel = cv2.split(self.to_be_changed_image)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        image1 = cv2.merge((red_channel, green_channel, blue_channel))
        self.filters_effects_dict["Martini with an Olive"] = image1

    def white_russian(self):
        increaseLookupTable = self.spread_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = self.spread_lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
        red_channel, green_channel, blue_channel = cv2.split(self.to_be_changed_image)
        red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        image1 = cv2.merge((red_channel, green_channel, blue_channel))
        self.filters_effects_dict["White Russian"] = image1

    def pina_colada(self):
        x_sobel = cv2.Sobel(self.to_be_changed_image, cv2.CV_64F, 1, 0, ksize=5)
        lap = cv2.Laplacian(x_sobel, cv2.CV_64F, ksize=5)
        self.filters_effects_dict["Pina Colada"] = x_sobel

    def long_island_iced_tea(self):
        y_sobel = cv2.Sobel(self.to_be_changed_image, cv2.CV_64F, 0, 1, ksize=5)
        lap = cv2.Laplacian(y_sobel, cv2.CV_64F, ksize=5)
        self.filters_effects_dict["Long Island Iced Tea"] = y_sobel

    def convert_to_pil(self, img):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            return im_pil
        except Exception as e:
            print(e)

    def convert_to_opencv(self, pil_image):
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image

    def faded_leaf(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)
        cb = util.or_convert(pil_image, 'RGB')
        cs = util.fill(cb.size, [250, 100, 50, .4])
        cr = css.blending.difference(cb, cs)

        cr = css.brightness(cr, 0.9)
        cr = css.contrast(cr, 0.85)
        cr = css.saturate(cr, 1.3)
        final_image = self.convert_to_opencv(cr)
        self.filters_effects_dict["Faded Leaf"] = final_image

    def lightbringer(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)

        cb = util.or_convert(pil_image, 'RGB')

        cs = util.fill(cb.size, [66, 181, 233, .35])
        cr = css.blending.hard_light(cb, cs)

        cr = css.grayscale(cr, 0.3)

        final_image = self.convert_to_opencv(cr)
        self.filters_effects_dict["Light Bringer"] = final_image

    def ethernal_sunshine(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)

        cb = util.or_convert(pil_image, 'RGB')

        cs = util.radial_gradient(cb.size,
                                  [(208, 166, 142), (54, 1, 1), (29, 2, 16)],
                                  [.25, .75, 1])

        cr = css.blending.screen(cb, cs)

        cr = css.brightness(cr, 0.85)
        cr = css.contrast(cr, 1.2)
        cr = css.saturate(cr, 1.1)

        final_image = self.convert_to_opencv(cr)
        self.filters_effects_dict["Eternal Sunshine"] = final_image

    def winter_is_coming(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)

        cb = util.or_convert(pil_image, 'RGB')

        cs = util.fill(cb.size, [57, 183, 217])
        cs = css.blending.soft_light(cb, cs)
        cr = Image.blend(cb, cs, .54)

        cr = css.brightness(cr, .87)
        cr = css.contrast(cr, .85)

        final_image = self.convert_to_opencv(cr)
        self.filters_effects_dict["Winter is Coming"] = final_image


    def yellow_hot_chick(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)
        cb = util.or_convert(pil_image, 'RGB')
        cs = util.linear_gradient(cb.size, [0, 21, 154], [230, 190, 51], False)
        cs = css.blending.lighten(cb, cs)
        final_image = self.convert_to_opencv(cs)
        self.filters_effects_dict["House of the Rising Sun "] = final_image

    def emotion_yellowrock(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)
        cb = util.or_convert(pil_image, 'RGB')
        cs = util.linear_gradient(cb.size, [0, 21, 154], [230, 190, 51], False)
        cs = css.blending.soft_light(cb, cs)
        final_image = self.convert_to_opencv(cs)
        self.filters_effects_dict["Emotion Yellowrock"] = final_image

    def is_so_cute(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)

        cb = util.or_convert(pil_image, 'RGB')

        cs1 = util.fill(cb.size, [236, 212, 189, .55])
        cm1 = css.blending.multiply(cb, cs1)

        cs2 = util.fill(cb.size, [49, 28, 14, .47])
        cm2 = css.blending.multiply(cb, cs2)  # multiply

        gradient_mask1 = util.radial_gradient_mask(cb.size, length=.45)
        cm = Image.composite(cm1, cm2, gradient_mask1)

        cs3 = util.fill(cb.size, [201, 177, 150, .75])
        cm3 = css.blending.overlay(cm, cs3)

        gradient_mask2 = util.radial_gradient_mask(cb.size, scale=.84)
        cm_ = Image.composite(cm3, cm, gradient_mask2)
        cr = Image.blend(cm, cm_, .7)

        cr = css.brightness(cr, 1.1)
        cr = css.sepia(cr, .2)
        cr = css.contrast(cr, .9)
        final_image = self.convert_to_opencv(cr)
        self.filters_effects_dict["is so Cute"] = final_image

    def color_passion(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)
        cb = util.or_convert(pil_image, 'RGB')
        cs = util.fill(cb.size, [75, 75, 75])
        cs = css.blending.overlay(cb, cs)
        cs = css.brightness(cs, 1.25)
        cs = css.hue_rotate(cs, 1.3)

        final_image = self.convert_to_opencv(cs)
        self.filters_effects_dict["Color Passion"] = final_image

    def limbo_travellers(self):
        pil_image = self.convert_to_pil(self.to_be_changed_image)
        cb = util.or_convert(pil_image, 'RGB')
        cs = util.fill(cb.size, [75, 75, 75])
        cs = css.blending.darken(cb, cs)
        cs = css.brightness(cs, 1.25)
        cs = css.hue_rotate(cs, 1.3)
        final_image = self.convert_to_opencv(cs)
        self.filters_effects_dict["Limbo Travellers"] = final_image

    def mordor(self):
        first_map = cv2.applyColorMap(self.to_be_changed_image, cv2.COLORMAP_BONE)
        final_map = cv2.applyColorMap(first_map, cv2.COLORMAP_HOT)
        self.filters_effects_dict["Mordor"] = final_map

    def gondor(self):
        first_map = cv2.applyColorMap(self.to_be_changed_image, cv2.COLORMAP_OCEAN)
        final_map = cv2.applyColorMap(first_map, cv2.COLORMAP_CIVIDIS)
        self.filters_effects_dict["Gondor"] = final_map

    def matrix(self):
        first_map = cv2.applyColorMap(self.to_be_changed_image, cv2.COLORMAP_BONE)
        final_map = cv2.applyColorMap(first_map, cv2.COLORMAP_DEEPGREEN)
        self.filters_effects_dict["Matrix"] = final_map
    def eric_clapton(self):
        first_map = cv2.applyColorMap(self.to_be_changed_image, cv2.COLORMAP_BONE)
        final_map = cv2.applyColorMap(first_map, cv2.COLORMAP_TWILIGHT_SHIFTED)
        self.filters_effects_dict["Eric Clapton"] = final_map

