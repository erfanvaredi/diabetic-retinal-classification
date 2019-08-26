# diabetic-retinal-classification
Diabetic classification based on retinal images

# Introduction
In this assignment we are going to predict Diabet problem by using deep neural networks. As the result of prepared code has been gaint, this arriving solution has achieved a medium breakthrough on accuracy which is equal to 62%. This report will clarify the way this problem has been solved by deep neural networks

# Implementation Brief
We have to use multi layer neural networks as deep NN. Due to the fact that out data’s structure is Image, the best type of neural network satisfying our goal is Convolutional Neural Networks. As we have to do for most of data, normalization does an important role in our process. Before doing any tasks, preprocessing images (our dataset) is highly recommended. Consequently better accuracy will achieved by preprocessed data. After doing preprocessing and normalizing, prepared dataset could be used as input of our deep convolutional neural network. Then deep NN will be run and fit to our data and the result will be produced by that. This report will cover step by step how this deep convolutional network be implemented.


# 1 . Preprocessing Data
As it is mentioned before preprocessing play an important role in our goal. Hence, we use Image Processing technics to preprocess our dataset. For this,  mentioned techniques has been used to find and bold the intensity of the abnormal areas and pieces for decreasing the effect of outlayers. Some of images have abnormal structures. For instance optic disk and vessels are abnormal. Note that before trying to solve the problem with grayscaled data, multi-channel images have been tested and results was not very reliable at all. As a consequent, using gray-scaled images was decided to use. After gray-scaled images prepared, next requirement is Normalization.

# 2 . Normalization
By having preprocessed data, now data could be normalized easily by divide image intensities to 255 (image converted to gray-scale previously). Then normalized data have to be attached label for learning the network. Class labels are available in each image name in the first substring. Now preprocessed data have been normalized. Lets jump to the next step for creating deep model.

# 3 . Creating Deep Neural Network Model
This Deep neural network is contained 7 layers which will be described. The first layer is Convolutional1D network with kernel size 5 and activation function relu. After preprocessing data we resized data which produce 256*256 gray-scaled image. So our input shape is going to be (256,256, 1).. Then we use MaxPooling to combine important features then Flatten and then dropout. At the end Dense(1) be used because we have binary classes [is diabetic or not]. Due to the fact that we have binary class labels, binary_crossentropy is used as our loss function. We use adam as our optimizer to. batch_size is 10 to update the weighs in batch mode. This may prevent our model from overfitting.

# 4 . Running Model
After doing all steps now is time to run the network and make sure that it works well for our solution. So, at first model has to compiled then it has to be fit and run then by passing test data, result will generate then.
</br>
RESULT:
</br>

    | TRAIN ACCURACY | TEST ACCURACY |
    | -------------- | ------------- |
    |     0.92       |      0.63     |
   
   
# 5. Programming Language
One of the major decisions had to be made was choosing the suitable programming language satisfying our goal for extracting knowledge from our data. After some searching the suitable decision has been made by selecting Python3 as the project programming language. Duo to the fact that, a lot of tools and frameworks are available for Python to create powerful Artificial Neural Networks such as Tensorflow and Keras which have been used in this project.

![](https://github.com/erfanvaredi/diabetic-retinal-classification/blob/master/normal_image.png)
> Normal Image


![](https://github.com/erfanvaredi/diabetic-retinal-classification/blob/master/gray_scaled_image.jpg)
> Gray Scaled Image


![](https://github.com/erfanvaredi/diabetic-retinal-classification/blob/master/vessels_by_canny.png)
> Vessels detected by canny edge detection filter


![](https://github.com/erfanvaredi/diabetic-retinal-classification/blob/master/preprocessed_white_top_hat.png)
> Preprocessed image sent to the CNN. white_top_hat + gray_scaled 


# 6. Result Report On Train Data
Evaluation on train data is described by table in below:

|   class-label    |   precision     |  recall |  F1-score  | support | 
| :--------------- |:---------------:| -------:| ---------- | ------- |
| 0 (non-diabetic) |      0.93       |  0.91   |    0.92    |   177   |
| 1 (diabetic)     |      0.93       |  0.94   |    0.94    |   236   |
</br>

|       | NEGATIVE | POSITIVE |
| ----- | -------- |  ------- |
| TRUE  |   161    |    223   |
| FALSE |    13    |    13    |
</br>

| SENSITIVITY | SPECIFICITY |
| ----------- | ----------- |
|    0.9449   |   0.9096    |


# 7. Result Report On TEST Data
Evaluation on test data is described by table in below:

|   class-label    |   precision     |  recall |  F1-score  | support | 
| :--------------- |:---------------:| -------:| ---------- | ------- |
| 0 (non-diabetic) |      0.59       |  0.53   |    0.56    |   45    |
| 1 (diabetic)     |      0.66       |  0.71   |    0.68    |   58    |
</br>

|       | NEGATIVE | POSITIVE |
| ----- | -------- |  ------- |
| TRUE  |    24    |    41    |
| FALSE |    17    |    21    |
</br>

| SENSITIVITY | SPECIFICITY |
| ----------- | ----------- |
|    0.7068   |   0.5333    |

# 8. Conclusion
This was just a small trying to resolve diabetic classification based on retinal data. The dataset used in this project was IDRid. Clone teh project and try to improve its test accuracy ;). Maybe it causes some diabetic problems to be simplified.
</br>
Just try to make to world the better place.</br>
Thanks
