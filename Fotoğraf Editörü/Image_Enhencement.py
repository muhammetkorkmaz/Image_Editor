import cv2



class Image_Enhancement:

    def __init__(self, image):
        self.to_be_changed_image = image

    def inverse_image(self):
        new_image = cv2.bitwise_not(self.to_be_changed_image)
        return new_image