import cv2
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QTransform, QPixmap


class Basic_Operations:

    def __init__(self, image):
        self.to_be_changed_image = image

    def change_image_brightness_contrast(self, brightness, contrast):
        changed_image = cv2.addWeighted(self.to_be_changed_image, contrast, self.to_be_changed_image, 0,
                                        brightness)
        self.to_be_changed_image=changed_image
        return changed_image

    def arrange_rotation(self, angle):
        qpixmap = self.opencv_image_to_qpixmap(self.to_be_changed_image)
        transform = QTransform().rotate(angle)
        qpixmap = qpixmap.transformed(transform, QtCore.Qt.SmoothTransformation)
        self.to_be_changed_image = self.qpixmap_to_array(qpixmap)
        return qpixmap

    def qpixmap_to_array(self, pixmap):

        channels_count = 4
        image = pixmap.toImage()
        s = image.bits().asstring(pixmap.width() * pixmap.height() * channels_count)
        arr = np.fromstring(s, dtype=np.uint8).reshape((pixmap.height(), pixmap.width(), channels_count))

        return arr

    def opencv_image_to_qpixmap(self, opencvimg):
        opencvimg_rgb = cv2.cvtColor(opencvimg, cv2.COLOR_BGR2RGB)
        height, width, channel = opencvimg_rgb.shape
        step = channel * width
        q_img = QtGui.QImage(opencvimg_rgb.data, width, height, step, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap

    def image_inverse(self, inverseBoolean):
        if (inverseBoolean):
            new_image = cv2.bitwise_not(self.to_be_changed_image)
            self.to_be_changed_image = new_image
            return new_image

    def flip_operation(self, position):
        for i in position:
            flipped_image = cv2.flip(self.to_be_changed_image, i)
            self.to_be_changed_image = flipped_image
        return flipped_image

    def clahe(self):
        img_yuv = cv2.cvtColor(self.to_be_changed_image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        cl1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.to_be_changed_image = cl1
        return cl1

    def histogram_normalization(self):

        img_yuv = cv2.cvtColor(self.to_be_changed_image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        normalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.to_be_changed_image = normalized_image
        return normalized_image

    """
    def change_image_brightness(self, brightness_value):
        # TODO filtreler arası geçiş nasıl olacak değerini koruyacak mı yoksa reset mi
        converted_to_pil = self.convert_to_pil(self.to_be_changed_image)
        changed_brightness = self.image_brightness_process(converted_to_pil, brightness_value)
        converted_to_opencv = self.convert_to_opencv(changed_brightness)
        # self.image_on_label=converted_to_opencv
        return converted_to_opencv

    def change_image_contrast(self, contrast_value):
        # TODO filtreler arası geçiş nasıl olacak değerini koruyacak mı yoksa reset mi
        converted_to_pil_c = self.convert_to_pil(self.to_be_changed_image)
        changed_contrast_c = self.image_contrast_process(converted_to_pil_c, contrast_value)
        converted_to_opencv = self.convert_to_opencv(changed_contrast_c)
        return converted_to_opencv

    def image_brightness_process(self, converted_img, value):
        try:
            enhancer = ImageEnhance.Brightness(converted_img)
            trans_img = enhancer.enhance(value)
            return trans_img
        except Exception as e:
            print(e)

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

    def image_contrast_process(self, img_c, factor):
        try:
            enhancer = ImageEnhance.Contrast(img_c)
            contrast_img = enhancer.enhance(factor)
            return contrast_img
        except Exception as e:
            print(e)
    """

    def rotate_image(self):
        if self.has_image:
            self.ui.slider_rotate.setVisible(True)
            self.ui.label_rotate.setVisible(True)
            self.ui.slider_rotate.valueChanged.connect(self.make_rotation)

        else:
            self.show_error_message("Image Error", "There is no image in the app. Please upload one.")
