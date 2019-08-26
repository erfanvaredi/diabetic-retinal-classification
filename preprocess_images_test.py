from os import listdir

from os.path import isfile, join
from nn_final_image.ETools import EImage
import cv2
import pandas as pd

path_base_test = 'test'
path_base_test_destination = 'preprocess_clr_test'

train_files = [f for f in listdir(path_base_test) if isfile(join(path_base_test, f))]
print(train_files[0])

df_train_class_label  = pd.read_csv('b. IDRiD_Disease Grading_Testing Labels.csv')

size_total = len(train_files)
index_done = 0

print('total file size: {}'.format(size_total))

for train_file_name in train_files:
    if '.jpg' not in train_file_name:
        continue

    print('-------\n{} processing started...'.format(train_file_name))

    image_name = train_file_name.split('.')[0]
    class_label = df_train_class_label.loc[df_train_class_label['Image name'] == image_name].iloc[0, df_train_class_label.columns.get_loc('Risk of macular edema')]
    print('class label is: {}'.format(class_label))

    image_name_destination = '{cl}_{file_name}'.format(cl=class_label, file_name=train_file_name)


    # data = cv2.imread('{}/{}'.format(path_base_train, train_file_name), cv2.IMREAD_GRAYSCALE)
    # data = cv2.imread('data_train/'+file_name, 0)
    img = EImage.read_image('{}/{}'.format(path_base_test, train_file_name),if_read_as_grayscale=False)
    img = EImage.get_cropped_image_from_black_area_rgb(img)
    # img = cv2.medianBlur(img, 5)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGGRAY2BGR)
    # resized = cv2.resize(img,(512, 512), interpolation=cv2.INTER_AREA)
    resized = EImage.resize_image(dimension=(80,80), img=img)
    # cv2.imwrite('{}/{}'.format(path_base_train_destination,image_name_destination), resized)
    EImage.save_image('{}/{}'.format(path_base_test_destination, image_name_destination), resized)
    print('{} done. {}% completed'.format(image_name_destination, round(index_done*100/size_total),1))
    index_done+=1



