from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from Effects_and_Filters import Effect_and_Filters
from Image_Enhencement import Image_Enhancement
from main_page import Ui_MainWindow
import sys, os
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel
from PyQt5 import QtCore
import shutil
import numpy as np
import cv2
from Basic_Operations import Basic_Operations
from qcrop.ui import QCrop


class Window(QtWidgets.QMainWindow):  # QtWidget'tan miras alma
    def __init__(self):
        super().__init__()  # normal qwidget, pencere çalıştı
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.has_image = False
        self.ui.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.listWidget.setSpacing(5)
        self.ui.label_4.setText(str(0))
        self.ui.label_5.setText(str(1.0))
        self.set_disabled()
        self.temp_cont = 1
        self.temp_bright = 0
        self.click_count = 0
        self.enhancement_dict = {}
        self.flip_list = []

        self.flip_value = None
        self.contrast_boolean = False

        self.histogram_boolean = False

        self.inverse_boolean = False
        self.temp_angle = 0
        self.effect_filter_class_dict = {}
        QtWidgets.QLabel()
        self.open_image_action = QtWidgets.QAction(QtGui.QIcon('folder.png'), "Open",
                                                   self)  # open_image için self'i sil
        self.open_image_action.setStatusTip("Open image")
        self.ui.toolBar.addAction(self.open_image_action)
        self.open_image_action.triggered.connect(self.open_image)
        self.save_file_action = QtWidgets.QAction(QtGui.QIcon('floppy-disk.png'), "Save", self)
        self.save_file_action.setStatusTip("Save current image")
        self.ui.toolBar.addAction(self.save_file_action)
        self.save_file_action.triggered.connect(self.save_image)
        self.save_file_as_action = QtWidgets.QAction(QtGui.QIcon('diskette.png'), "Save As", self)
        self.save_file_as_action.setStatusTip("Save as current image")
        self.ui.toolBar.addAction(self.save_file_as_action)
        self.save_file_as_action.triggered.connect(self.save_as_image)
        self.ui.pushButton_6.setEnabled(False)
        self.ui.pushButton_4.setEnabled(False)
        self.ui.pushButton_5.setEnabled(False)
        # Connect
        self.ui.pushButton_4.clicked.connect(self.click)
        self.ui.pushButton_5.clicked.connect(self.click)
        self.ui.pushButton_6.clicked.connect(self.click)
        self.ui.horizontalSlider.valueChanged.connect(self.change_brightness_and_contrast_value)
        self.ui.horizontalSlider_2.valueChanged.connect(self.change_brightness_and_contrast_value)
        self.ui.horizontalSlider_3.valueChanged.connect(self.rotation)
        self.ui.pushButton.clicked.connect(self.click)
        self.ui.pushButton_9.clicked.connect(self.click)
        self.ui.pushButton_10.clicked.connect(self.click)
        self.ui.pushButton_11.clicked.connect(self.click)
        self.ui.pushButton_12.clicked.connect(self.click)
        self.ui.pushButton_2.clicked.connect(self.click)
        self.ui.pushButton_3.clicked.connect(self.click)

    def open_image(self):
        image_path, _ = QFileDialog.getOpenFileName()
        self.image_path = image_path
        if image_path == "":
            return
        if image_path.split(".")[-1] not in ["jpg", "png"]:
            self.show_error_message("File Type Error",
                                    "Unsupported file type, please try again. It should be .jpg, or .png")
        else:
            pixmap = QPixmap(image_path)
            w, h = self.fit_image(pixmap.width(), pixmap.height())
            pixmap = pixmap.scaled(w, h)
            self.pixmap = pixmap  # Aşağıda kullanılacağı için.
            self.ui.label_2.setPixmap(pixmap)
            self.has_image = True
            self.set_enabled()
            self.image_on_label = cv2.imread(self.image_path)
            self.image_on_label_temp = cv2.imread(self.image_path)

            # self.ui.label_2.pixmap().toImage().save("temp.png", 'PNG') bakılacak buraya temp gerek var mı
            """
            self.ui.label_manuel_enhancement.setVisible(True)
            self.ui.label_brightness.setVisible(True)
            self.ui.label_contrast.setVisible(True)
            self.ui.slider_brightness.setVisible(True)
            self.ui.slider_contrast.setVisible(True)
            """
            # self.basic_operations = Basic_Operations("temp.png")
            if (self.has_image == True):
                self.ui.listWidget.clear()
                self.reset_properties()
            self.list_widget_initialize()

    def fit_image(self, width, height):  # Hep fit çalışacak
        self.ui.label_2.setStyleSheet("""
            QLabel{
                border: 0px
            }
        """)
        k = self.ui.label_2.frameGeometry().height() / height
        if width * k <= self.ui.label_2.frameGeometry().width():
            w = width * k
            h = self.ui.label_2.frameGeometry().height()
        else:
            k = self.ui.label_2.frameGeometry().width() / width
            w = self.ui.label_2.frameGeometry().width()
            h = height * k

        return w, h

    def save_image(self):
        if self.has_image:
            sure_message = QMessageBox.question(self, 'Are you sure?',
                                                "This will overwrite to original image. Are you sure?",
                                                QMessageBox.Yes | QMessageBox.No)
            if sure_message == QMessageBox.Yes:
                image_path = self.image_path
                self.ui.label_2.pixmap().toImage().save(image_path, 'PNG')
        else:
            self.show_error_message("Image Error", "There is no image in the app. Please upload one.")

    def save_as_image(self):
        if self.has_image:
            image_path, _ = QFileDialog.getSaveFileName()
            if not image_path.endswith("png") or not image_path.endswith("jpg"):
                image_path = image_path + ".png"
            self.ui.label_2.pixmap().toImage().save(image_path, 'PNG')
        else:
            self.show_error_message("Image Error", "There is no image in the app. Please upload one.")

    def show_error_message(self, title, message):
        QMessageBox.warning(self, title, message)

    def set_disabled(self):
        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton_2.setEnabled(False)
        self.ui.pushButton_3.setEnabled(False)
        self.ui.pushButton_4.setEnabled(False)
        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_6.setEnabled(False)
        self.ui.pushButton_9.setEnabled(False)
        self.ui.pushButton_10.setEnabled(False)
        self.ui.pushButton_11.setEnabled(False)
        self.ui.pushButton_12.setEnabled(False)
        self.ui.horizontalSlider.setEnabled(False)
        self.ui.horizontalSlider_2.setEnabled(False)
        self.ui.horizontalSlider_3.setEnabled(False)

    def set_enabled(self):

        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_2.setEnabled(True)
        self.ui.pushButton_3.setEnabled(True)
        self.ui.pushButton_4.setEnabled(True)
        self.ui.pushButton_5.setEnabled(True)
        self.ui.pushButton_6.setEnabled(True)
        self.ui.pushButton_9.setEnabled(True)
        self.ui.pushButton_10.setEnabled(True)
        self.ui.pushButton_11.setEnabled(True)
        self.ui.pushButton_12.setEnabled(True)
        self.ui.horizontalSlider.setEnabled(True)
        self.ui.horizontalSlider_2.setEnabled(True)
        self.ui.horizontalSlider_3.setEnabled(True)

    def crop_image(self):
        # piximage = self.opencv_image_to_qpixmap(self.image_on_label_temp)
        basic_operations = Basic_Operations(self.image_on_label_temp)
        piximage_2 = self.opencv_image_to_qpixmap(self.image_on_label_temp)
        temp_pixmap = basic_operations.arrange_rotation(self.temp_angle)
        if (len(self.flip_list) != 0):
            temp_pixmap_1 = basic_operations.flip_operation(self.flip_list)
        temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
        if (len(self.enhancement_dict) != 0):
            if (list(self.enhancement_dict.keys())[0] == "contrast_en"):

                temp_pixmap3 = basic_operations.clahe()
                if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                    temp_pixmap4 = basic_operations.histogram_normalization()
            elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                temp_pixmap5 = basic_operations.histogram_normalization()
                if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                    temp_pixmap6 = basic_operations.clahe()
        new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
        piximage = self.opencv_image_to_qpixmap(new_image)
        crop_tool = QCrop(piximage)

        # crop_tool_2 = QCrop(piximage_2)

        status = crop_tool.exec()
        # status2=crop_tool_2.exec()
        if status == 1:
            # cropped_image = crop_tool.image
            croped_final_image = piximage_2.copy(crop_tool.crop_values())
            # w, h = self.fit_image(croped_final_image.width(), croped_final_image.height())
            self.image_on_label = self.qpixmap_to_array(croped_final_image)
            self.image_on_label_temp = self.qpixmap_to_array(croped_final_image)
            basic_operations = Basic_Operations(self.image_on_label_temp)
            piximage_2 = self.opencv_image_to_qpixmap(self.image_on_label_temp)
            if (len(self.flip_list) != 0):
                temp_pixmap_1 = basic_operations.flip_operation(self.flip_list)
            temp_pixmap = basic_operations.arrange_rotation(self.temp_angle)

            temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
            if (len(self.enhancement_dict) != 0):
                if (list(self.enhancement_dict.keys())[0] == "contrast_en"):
                    temp_pixmap3 = basic_operations.clahe()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                        temp_pixmap4 = basic_operations.histogram_normalization()
                elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                    temp_pixmap3 = basic_operations.histogram_normalization()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                        temp_pixmap4 = basic_operations.clahe()
            new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
            piximage_final = self.opencv_image_to_qpixmap(new_image)
            w, h = self.fit_image(piximage_final.width(), piximage_final.height())
            self.list_widget_initialize()
            cropped_image_final = piximage_final.scaled(w, h)
            self.ui.label_2.setPixmap(cropped_image_final)
        else:
            return

    def list_widget_initialize(self):
        if (self.has_image == True):
            self.ui.listWidget.clear()
        self.ui.pushButton_6.setEnabled(True)
        qpixmap_list = []
        # to_be_filtered_image = cv2.imread(self.image_path)
        opencvimg_rgb = cv2.cvtColor(self.image_on_label, cv2.COLOR_BGRA2BGR)
        to_be_filtered_image = opencvimg_rgb
        effect_filters = Effect_and_Filters(to_be_filtered_image)
        self.effect_filter_class_values = effect_filters.filters_effects_dict.copy().values()
        for key in effect_filters.filters_effects_dict:
            effect_filters.filters_effects_dict[key] = self.opencv_image_to_qpixmap(
                effect_filters.filters_effects_dict[key])
        # self.imageColl.append(self.pixMap)
        # self.imageArr.append(self.pixMap)
        self.effect_filter_class_dict = effect_filters.filters_effects_dict
        for key in effect_filters.filters_effects_dict:
            self.icons = QtWidgets.QListWidgetItem(QtGui.QIcon(effect_filters.filters_effects_dict[key]), key)
            self.iconSize = QtCore.QSize(150, 150)
            self.ui.listWidget.setIconSize(self.iconSize)
            self.ui.listWidget.addItem(self.icons)

    def opencv_image_to_qpixmap(self, opencvimg):
        opencvimg_rgb = cv2.cvtColor(opencvimg, cv2.COLOR_BGR2RGB)
        height, width, channel = opencvimg_rgb.shape
        step = channel * width
        q_img = QtGui.QImage(opencvimg_rgb.data, width, height, step, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap

    # TODO open_image_f ismini değiştir.
    def open_image_f(self, new_pixmap):
        pixmap = new_pixmap
        w, h = self.fit_image(pixmap.width(), pixmap.height())
        pixmap = pixmap.scaled(w, h)
        self.pixmap = pixmap  # Aşağıda kullanılacağı için.
        self.ui.label_2.setPixmap(pixmap)
        self.has_image = True
        # self.ui.label_2.pixmap().toImage().save("temp.png", 'PNG') bakılacak buraya temp gerek var mı
        """
        self.ui.label_manuel_enhancement.setVisible(True)
        self.ui.label_brightness.setVisible(True)
        self.ui.label_contrast.setVisible(True)
        self.ui.slider_brightness.setVisible(True)
        self.ui.slider_contrast.setVisible(True)
        """
        # self.basic_operations = Basic_Operations("temp.png")
        # self.image_on_label = self.pixmap;

    def revert_to_original_image(self):
        self.image_on_label = cv2.imread(self.image_path)
        self.image_on_label_temp = self.image_on_label
        self.list_widget_initialize()
        pixmap = QPixmap(self.image_path)
        w, h = self.fit_image(pixmap.width(), pixmap.height())
        pixmap = pixmap.scaled(w, h)
        self.pixmap = pixmap  # Aşağıda kullanılacağı için.
        self.ui.label_2.setPixmap(pixmap)

    def clear_image(self):
        self.ui.label_2.clear()
        self.ui.listWidget.clear()
        self.set_disabled()
        self.image_on_label = None
        self.image_on_label_temp = None

    def change_brightness_and_contrast_value(self):
        # before_change=self.image_on_label_temp.copy()
        # self.image_on_label_2=before_change

        brightness = self.ui.horizontalSlider.value()
        contrast = self.ui.horizontalSlider_2.value() / 10
        self.temp_cont = contrast
        self.temp_bright = brightness
        # self.image_on_label_temp=cv2.addWeighted(self.image_on_label_temp, contrast, self.image_on_label_temp, 0, brightness)
        self.ui.label_4.setText(str(brightness))
        self.ui.label_5.setText(str(contrast))
        # new_image = cv2.add(self.image_on_label, np.array([float(self.temp_brightness)]))
        basic_operations = Basic_Operations(self.image_on_label_temp)
        if (len(self.flip_list) != 0):
            temp_pixmap_1 = basic_operations.flip_operation(self.flip_list)
        if (len(self.enhancement_dict) != 0):

            if (list(self.enhancement_dict.keys())[0] == "contrast_en"):
                temp_pixmap3 = basic_operations.clahe()
                if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                    temp_pixmap4 = basic_operations.histogram_normalization()
            elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                temp_pixmap3 = basic_operations.histogram_normalization()
                if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                    temp_pixmap4 = basic_operations.clahe()
        temp_pixmap = basic_operations.arrange_rotation(self.temp_angle)
        temp_pixmap_2 = basic_operations.image_inverse(self.inverse_boolean)
        # temp_pixmap3 = basic_operations.flip_operation(self.flip_value)
        # w, h = self.fit_image(temp_pixmap.width(), temp_pixmap.height())
        # temp_pixmap = temp_pixmap.scaled(w, h)
        # self.ui.label_2.setPixmap(temp_pixmap)
        # self.basic_operations2=Basic_Operations(self.qpixmap_to_array(temp_pixmap))
        new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
        self.image_on_label = new_image
        final_pixmap = self.opencv_image_to_qpixmap(self.image_on_label)
        self.open_image_f(final_pixmap)
        # self.image_on_label = self.qpixmap_to_array(final_pixmap)
        # self.ui.label_4.setText(str(brightness / 10))
        # brightness_arranged = brightness / 10
        # self.basic_operations = Basic_Operations(self.image_on_label)
        # self.image_on_label=self.basic_operations.change_image_brightness(brightness_arranged)
        # final_pixmap = self.opencv_image_to_qpixmap(self.basic_operations.change_image_brightness(brightness_arranged))
        # self.open_image_f(final_pixmap)
        # self.image_on_label = self.qpixmap_to_array(final_pixmap)

    """
    def change_contrast_value(self):
        contrast = self.ui.horizontalSlider_2.value()
        self.temp_contrast += contrast
        new_image = cv2.addWeighted(self.image_on_label, self.temp_contrast, self.image_on_label, 0,
                                    self.temp_brightness)
        # self.image_on_label=new_image
        final_pixmap = self.opencv_image_to_qpixmap(new_image)
        self.open_image_f(final_pixmap)
        # self.basic_operations = Basic_Operations(self.image_on_label)
        # final_pixmap = self.opencv_image_to_qpixmap(self.basic_operations.change_image_contrast(contrast_arranged))
        # self.open_image_f(final_pixmap)
        """

    def rotation(self):

        angle = self.ui.horizontalSlider_3.value()
        self.temp_angle = angle
        basic_operations = Basic_Operations(self.image_on_label_temp)
        if (len(self.enhancement_dict) != 0):

            if (list(self.enhancement_dict.keys())[0] == "contrast_en"):
                temp_pixmap3 = basic_operations.clahe()
                if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                    temp_pixmap4 = basic_operations.histogram_normalization()
            elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                temp_pixmap3 = basic_operations.histogram_normalization()
                if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                    temp_pixmap4 = basic_operations.clahe()
        temp_image = basic_operations.image_inverse(self.inverse_boolean)
        if (len(self.flip_list) != 0):
            temp_pixmap1 = basic_operations.flip_operation(self.flip_list)
        new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
        self.image_on_label = new_image

        qpixmap = self.opencv_image_to_qpixmap(self.image_on_label)
        transform = QTransform().rotate(angle)
        qpixmap = qpixmap.transformed(transform, QtCore.Qt.SmoothTransformation)
        w, h = self.fit_image(qpixmap.width(), qpixmap.height())
        qpixmap = qpixmap.scaled(w, h)
        self.ui.label_2.setPixmap(qpixmap)
        self.pixmap = qpixmap

        # rotated_image = self.qpixmap_to_array(qpixmap)
        # self.image_on_label=rotated_image
        # self.image_on_label_temp=rotated_image
        # temp_pixmap = self.opencv_image_to_qpixmap(self.image_on_label_temp)
        # transform = QTransform().rotate(angle)
        # temp_pixmap = temp_pixmap.transformed(transform, QtCore.Qt.SmoothTransformation)
        # self.image_on_label_temp = self.qpixmap_to_array(temp_pixmap)

        # temp_pixmap=self.opencv_image_to_qpixmap(self.image_on_label_temp)
        # transform = QTransform().rotate(angle)
        # temp_pixmap = temp_pixmap.transformed(transform, QtCore.Qt.SmoothTransformation)
        # self.image_on_label_temp=self.qpixmap_to_array(temp_pixmap)
        # self.image_on_label = self.qpixmap_to_array(temp_pixmap)
        # self.image_on_label_temp=rotated_image
        # w,h=self.fit_image(temp_pixmap.width(),temp_pixmap.height())
        # temp_pixmap=temp_pixmap.scaled(w,h)

    def qpixmap_to_array(self, pixmap):

        channels_count = 4
        image = pixmap.toImage()
        s = image.bits().asstring(pixmap.width() * pixmap.height() * channels_count)
        arr = np.fromstring(s, dtype=np.uint8).reshape((pixmap.height(), pixmap.width(), channels_count))

        return arr

    def reset_properties(self):
        self.ui.label_4.setText(str(0))
        self.ui.label_5.setText(str(1.0))
        self.temp_cont = 1
        self.temp_bright = 0
        self.click_count = 0
        self.enhancement_dict = {}
        self.flip_list = []
        self.ui.horizontalSlider.setValue(0)
        self.ui.horizontalSlider_2.setValue(10)
        self.ui.horizontalSlider_3.setValue(0)

        self.flip_value = None
        self.contrast_boolean = False

        self.histogram_boolean = False

        self.inverse_boolean = False
        self.temp_angle = 0
        self.effect_filter_class_dict = {}

    def click(self):
        sender = self.sender()
        if sender.text() == "Apply":
            if len(self.ui.listWidget.selectedIndexes()) == 0:
                return
            open_cv_selected_image = list(self.effect_filter_class_values)[
                self.ui.listWidget.selectedIndexes()[0].row()]
            self.image_on_label = open_cv_selected_image
            self.image_on_label_temp = open_cv_selected_image
            basic_operations = Basic_Operations(self.image_on_label_temp)
            # temp_pixmap3 = basic_operations.flip_operation(self.flip_value)
            if (len(self.flip_list) != 0):
                temp_pixmap_1 = basic_operations.flip_operation(self.flip_list)
            temp_pixmap = basic_operations.arrange_rotation(self.temp_angle)

            temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
            if (len(self.enhancement_dict) != 0):

                if (list(self.enhancement_dict.keys())[0] == "contrast_en"):
                    temp_pixmap3 = basic_operations.clahe()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                        temp_pixmap4 = basic_operations.histogram_normalization()
                elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                    temp_pixmap3 = basic_operations.histogram_normalization()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                        temp_pixmap4 = basic_operations.clahe()
            # basic_operations = Basic_Operations(self.image_on_label_temp)
            new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
            self.image_on_label = new_image

            # final_pixmap = self.opencv_image_to_qpixmap(new_image)
            # self.open_image_f(final_pixmap)
            qpixmap_image = self.opencv_image_to_qpixmap(self.image_on_label)
            self.open_image_f(qpixmap_image)
            # self.list_widget_initialize()

        if sender.text() == "Original Image":
            self.reset_properties()
            self.revert_to_original_image()
        if sender.text() == "Clear":
            self.reset_properties()
            self.clear_image()
        if sender.text() == "Inverse":
            self.click_count += 1
            if (self.click_count % 2 == 1):
                self.inverse_boolean = True
            else:
                self.inverse_boolean = False

            basic_operations = Basic_Operations(self.image_on_label_temp)

            temp_pixmap = basic_operations.arrange_rotation(self.temp_angle)
            if (len(self.flip_list) != 0):
                temp_pixmap_1 = self.basic_operations.flip_operation(self.flip_list)
            temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
            if (len(self.enhancement_dict) != 0):
                if (list(self.enhancement_dict.keys())[0] == "contrast_en"):
                    temp_pixmap3 = basic_operations.histogram_normalization()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                        temp_pixmap4 = basic_operations.clahe()
                elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                    temp_pixmap3 = basic_operations.clahe()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                        temp_pixmap4 = basic_operations.histogram_normalization()

            new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)

            self.image_on_label = new_image
            # self.image_on_label_temp = inversed_image
            inversed_qpixmap = self.opencv_image_to_qpixmap(self.image_on_label)
            self.open_image_f(inversed_qpixmap)
        if sender.text() == "Horizontal":
            flip_value_temp = 1
            self.flip_list.append(flip_value_temp)
            basic_operations = Basic_Operations(self.image_on_label_temp)
            temp_pixmap = basic_operations.flip_operation(self.flip_list)
            if (len(self.enhancement_dict) != 0):

                if (list(self.enhancement_dict.keys())[0] == "contrast_en"):
                    temp_pixmap3 = basic_operations.clahe()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                        temp_pixmap4 = basic_operations.histogram_normalization()
                elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                    temp_pixmap3 = basic_operations.histogram_normalization()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                        temp_pixmap4 = basic_operations.clahe()
            # basic_operations = Basic_Operations(self.image_on_label_temp)
            temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
            temp_pixmap3 = basic_operations.arrange_rotation(self.temp_angle)
            new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
            self.image_on_label = new_image
            flipped_qpixmap = self.opencv_image_to_qpixmap(self.image_on_label)
            self.open_image_f(flipped_qpixmap)
        if sender.text() == "Vertical":
            flip_value_temp = 0
            self.flip_list.append(flip_value_temp)
            basic_operations = Basic_Operations(self.image_on_label_temp)
            temp_pixmap3 = basic_operations.flip_operation(self.flip_list)
            if (len(self.enhancement_dict) != 0):

                if (list(self.enhancement_dict.keys())[0] == "contrast_en"):
                    temp_pixmap3 = basic_operations.clahe()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                        temp_pixmap4 = basic_operations.histogram_normalization()
                elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                    temp_pixmap3 = basic_operations.histogram_normalization()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                        temp_pixmap4 = basic_operations.clahe()
            # basic_operations = Basic_Operations(self.image_on_label_temp)
            temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
            temp_pixmap = basic_operations.arrange_rotation(self.temp_angle)
            new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
            self.image_on_label = new_image
            flipped_qpixmap = self.opencv_image_to_qpixmap(self.image_on_label)
            self.open_image_f(flipped_qpixmap)
        if sender.text() == "Origin":
            flip_value_temp = -1
            self.flip_list.append(flip_value_temp)
            basic_operations = Basic_Operations(self.image_on_label_temp)
            temp_pixmap = basic_operations.flip_operation(self.flip_list)
            if (len(self.enhancement_dict) != 0):

                if (list(self.enhancement_dict.keys())[0] == "contrast_en"):
                    temp_pixmap3 = basic_operations.clahe()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "histogram_norm"):
                        temp_pixmap4 = basic_operations.histogram_normalization()
                elif (list(self.enhancement_dict.keys())[0] == "histogram_norm"):
                    temp_pixmap3 = basic_operations.histogram_normalization()
                    if (len(self.enhancement_dict) == 2 and list(self.enhancement_dict.keys())[1] == "contrast_en"):
                        temp_pixmap4 = basic_operations.clahe()
            # basic_operations = Basic_Operations(self.image_on_label_temp)
            temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
            temp_pixmap3 = basic_operations.arrange_rotation(self.temp_angle)
            new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
            self.image_on_label = new_image
            flipped_qpixmap = self.opencv_image_to_qpixmap(self.image_on_label)
            self.open_image_f(flipped_qpixmap)
        if sender.text() == "Contrast Enhancement":
            self.contrast_boolean = True
            if (not "contrast_en" in list(self.enhancement_dict.keys())):
                self.enhancement_dict["contrast_en"] = True
            basic_operations = Basic_Operations(self.image_on_label_temp)
            if (len(self.flip_list) != 0):
                temp_pixmap1 = basic_operations.flip_operation(self.flip_list)
            temp_pixmap = basic_operations.arrange_rotation(self.temp_angle)
            temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
            if (self.histogram_boolean):
                temp_pixmap4 = basic_operations.histogram_normalization()
                temp_pixmap5 = basic_operations.clahe()
            else:
                temp_pixmap4 = basic_operations.clahe()
            new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)

            self.image_on_label = new_image
            # self.image_on_label_temp = inversed_image
            inversed_qpixmap = self.opencv_image_to_qpixmap(self.image_on_label)
            self.open_image_f(inversed_qpixmap)
        if sender.text() == "Histogram Normalization":
            self.histogram_boolean = True
            if (not "histogram_norm" in list(self.enhancement_dict.keys())):
                self.enhancement_dict["histogram_norm"] = True
            basic_operations = Basic_Operations(self.image_on_label_temp)
            if (len(self.flip_list) != 0):
                temp_pixmap1 = basic_operations.flip_operation(self.flip_list)
            temp_pixmap = basic_operations.arrange_rotation(self.temp_angle)
            temp_pixmap2 = basic_operations.image_inverse(self.inverse_boolean)
            if (self.contrast_boolean):
                temp_pixmap4 = basic_operations.clahe()
                temp_pixmap5 = basic_operations.histogram_normalization()
            else:
                temp_pixmap6 = basic_operations.histogram_normalization()
            new_image = basic_operations.change_image_brightness_contrast(self.temp_bright, self.temp_cont)
            self.image_on_label = new_image
            # self.image_on_label_temp = inversed_image
            norm_image = self.opencv_image_to_qpixmap(self.image_on_label)

            self.open_image_f(norm_image)
        if sender.text() == "Crop":
            self.crop_image()


app = QtWidgets.QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec_())
