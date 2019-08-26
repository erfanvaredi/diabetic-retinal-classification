import cv2
import numpy as np
class EImage:

    def __init__(self):
        print('Inited')

    @classmethod
    def get_cropped_image_from_black_area(cls, img:np.ndarray) -> np.ndarray:
        """

        :param img: numpy.ndarray as input. This is your image read by cv2
        :return: image without black box
        """
        index_median_col = int((img.shape[1]-1)/2)
        index_min_col = 0
        index_max_col = img.shape[1]

        for index_col in range(img.shape[1]):
            # print(max(set(img[:, index_col]))<10)
            # if (len(set(img[:, index_col])) == 1) and (index_col <index_median_col):
            if (max(set(img[:, index_col]))<30) and (index_col <index_median_col):
                index_min_col = index_col if index_col > index_min_col else index_min_col
            if (max(set(img[:, index_col]))<30) and (index_col >index_median_col):
                index_max_col = index_col if index_col< index_max_col else index_max_col

        #
        # for index_row in range(img.shape[0]):
        #     for index_col  in range(img.shape[1]):
        #
        #         # If the pixel is BLACK (0)
        #         if (img[index_row, index_col] == 0) and (index_col < int(img.shape[1]-1/2)):
        #             index_min_col = index_col if index_col < index_min_col else index_min_col
        #         elif (img[index_row, index_col] == 0) and (index_col > int(img.shape[1]-1/2)):
        #             index_max_col = index_col if index_col > index_max_col else index_max_col

        print('image size was: ', img.shape)
        print('image size now is: ', (img.shape[0], index_max_col-index_min_col+1))
        return img[:,index_min_col: index_max_col+1]


    @classmethod
    def get_cropped_image_from_black_area_rgb(cls, img:np.ndarray) -> np.ndarray:
        """

        :param img: numpy.ndarray as input. This is your image read by cv2
        :return: image without black box
        """
        index_median_col = int((img.shape[1]-1)/2)
        index_min_col = 0
        index_max_col = img.shape[1]

        for index_col in range(img.shape[1]):
            # print(max(set(img[:, index_col]))<10)
            # if (len(set(img[:, index_col])) == 1) and (index_col <index_median_col):
            if all(x < 30 for x in max(img[:, index_col].tolist())) and (index_col <index_median_col):
                index_min_col = index_col if index_col > index_min_col else index_min_col
            if all(x < 30 for x in max(img[:, index_col].tolist())) and (index_col >index_median_col):
                index_max_col = index_col if index_col< index_max_col else index_max_col

        #
        # for index_row in range(img.shape[0]):
        #     for index_col  in range(img.shape[1]):
        #
        #         # If the pixel is BLACK (0)
        #         if (img[index_row, index_col] == 0) and (index_col < int(img.shape[1]-1/2)):
        #             index_min_col = index_col if index_col < index_min_col else index_min_col
        #         elif (img[index_row, index_col] == 0) and (index_col > int(img.shape[1]-1/2)):
        #             index_max_col = index_col if index_col > index_max_col else index_max_col

        print('image size was: ', img.shape)
        print('image size now is: ', (img.shape[0], index_max_col-index_min_col+1))
        return img[:,index_min_col: index_max_col+1]

    @classmethod
    def resize_image(cls, img: np.ndarray, dimension:tuple) -> np.ndarray:
        """

        :param img: numpy.ndarray as input. This is your image read by cv2
        :param dimension: size of resized image in tuple: (512, 512)
        :return: resized image
        """
        return cv2.resize(img,dimension, interpolation=cv2.INTER_AREA)    \

    @classmethod
    def read_image(cls, path_of_image: str(), if_read_as_grayscale=True) -> np.ndarray:
        if if_read_as_grayscale:
            data = cv2.imread(path_of_image, cv2.IMREAD_GRAYSCALE)
        else:
            data = cv2.imread(path_of_image)

        return data.copy()


    @classmethod
    def save_image(cls, file_name: str(), img=np.ndarray) -> None:
        cv2.imwrite(file_name, img)
        return None

