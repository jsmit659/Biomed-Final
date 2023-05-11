# Biomed Final Project

### Hardware Used:

CPU: 11th Gen Intel(R) Core(TM) i7-11600H @ 2.90GHz\
GPU: NVIDIA RTX3050Ti 8Gb RAM Boost Clock: 1.78GHz\
RAM: 40GB DDR4\ 
Google CoLab Pro+ 

#### Note: This Repository is not a comprehensive collection of the final projects files, only those relivant to the final product of the project needed to replicate the final results.

### Introduction to VGG19
VGG19 is a deep convolutional neural network (CNN) that was developed by the Visual Geometry Group at Oxford University. It is a modification of the VGG16 architecture, with additional convolutional layers and parameters, leading to better accuracy in image classification tasks.

The VGG19 architecture consists of 19 layers, with 16 convolutional layers, 3 fully connected layers, and one softmax layer. The first 13 layers of the network are convolutional layers with a 3x3 filter size and a stride of 1, followed by max-pooling layers with a 2x2 filter size and a stride of 2. The last 6 layers are fully connected layers with 4096 units each, except for the last layer which has 1000 units for classification of 1000 classes.

The convolutional layers in VGG19 learn to identify various features in an image, such as edges, textures, and patterns. The max-pooling layers help to reduce the spatial dimensions of the feature maps, while preserving the important features. The fully connected layers at the end of the network use these learned features to make predictions about the input image.

The VGG19 network is trained using a large dataset of images, such as ImageNet, which contains over 1 million images and 1000 categories. During training, the network learns to adjust its parameters to minimize the difference between the predicted outputs and the actual labels.

VGG19 has been shown to perform well on a variety of image recognition tasks, such as object recognition, scene recognition, and image classification. However, due to its large number of parameters, VGG19 requires significant computational resources and memory to train and deploy. As a result, it is often used as a baseline architecture for comparison with other CNN architectures, and is not always the best choice for practical applications.

### Introduction to DenseNET
DenseNET is a deep neural network architecture that aims to improve the flow of information in convolutional neural networks (CNNs) by addressing the problem of vanishing gradients. Developed by Gao Huang, Zhuang Liu, and Kilian Q. Weinberger in 2017, DenseNET has achieved state-of-the-art results on a variety of computer vision tasks, such as image classification, object detection, and semantic segmentation.

DenseNET architecture is based on a "Dense connectivity" concept that connects each layer to every other layer in a feedforward manner. Specifically, each layer in DenseNet takes as input the feature maps of all preceding layers and produces its own feature maps, which are then passed on to all subsequent layers in the network. This results in a dense block of layers, where each layer receives the concatenated feature maps of all preceding layers as its input.

The dense connectivity concept in DenseNet has several advantages. First, it promotes feature reuse and reduces the number of parameters in the network. Second, it allows for gradient flow throughout the network and mitigates the vanishing gradients problem. Third, it enhances the representation power of the network by allowing each layer to access a wide range of features learned by preceding layers.

DenseNet consists of several dense blocks, each followed by a transition layer that reduces the spatial dimensionality and controls the number of feature maps. The dense blocks can be stacked to any depth, and the transition layers can be used to gradually reduce the spatial size and feature map dimensionality of the network. A global average pooling layer and a fully connected layer are added at the end of the network to generate the final output.

DenseNet has several variants, including DenseNet-121, DenseNet-169, DenseNet-201, and DenseNet-264, which differ in the number of layers and the number of feature maps per layer. DenseNet has shown to achieve state-of-the-art performance on several benchmark datasets, such as CIFAR-10, CIFAR-100, ImageNet, and COCO.

In summary, DenseNet is a powerful neural network architecture that promotes feature reuse, mitigates the vanishing gradients problem, and enhances the representation power of the network. Its dense connectivity concept and stacked dense blocks make it an effective solution for various computer vision tasks.

### Introduction to Project
Multiple sclerosis (MS) is a chronic autoimmune disease that affects the central nervous system, including the
brain, spinal cord, and optic nerves. The disease is characterized by the body’s immune system attacking the myelin sheath,
a protective covering that surrounds nerve fibers, resulting in inflammation, scarring, and damage to the nerves. MS can cause a wide range of symptoms, including muscle weakness, numbness, tingling, fatigue, difficulty with coordination and balance, and vision problems. Below is a visual aide of what MS does to the neurons of the body. 

![image](https://github.com/jsmit659/Biomed-Final/assets/113131600/c975dc74-91da-4aac-a900-93437a277668)


MS can have a significant impact on a person’s quality of life and ability to carry out daily activities, as the symptoms
can be debilitating and unpredictable. The course of the disease is unpredictable, and the severity of symptoms can vary widely among individuals, making it challenging to diagnose
and treat.

Artificial intelligence (AI) has the potential to aid
in the early detection and accurate diagnosis of MS. AI
algorithms can analyze magnetic resonance imaging (MRI)
scans to identify patterns that may indicate the presence of
MS. Machine learning models can learn from large datasets
of MRI scans and medical records to identify patterns and
generate accurate predictions. This can help doctors make
more informed diagnoses and develop personalized treatment
plans for patients with MS.

Overall, the use of AI in diagnosing MS has the
potential to improve the accuracy and efficiency of diagnoses,
allowing for earlier detection and treatment of the disease.
For this project, the motivation was simple: family.
Growing up, with a parent suffering from MS, going through
countless tests, was rough. Not only for her with all the stress,
pain and anxiety of the diagnosis process, but on those around
us. This process, which took place 1999-2000, tore apart the
family and led to a bitter divorce. From this experience, I
always thought I wanted to go into the medical field, but
ultimately engineering was a better fit. Therefore, when the
opportunity to marry those two aspects of my life together,
the project choice was obvious


### Augment Images
First, the file paths are appended with a label for later classification. The following code block shows how these images are labeled and aumented to 224x224x1 
```python
path = 'FILE_PATH'
for f in glob.iglob(path):
    img8=cv2.imread(f)
    img8 = Image.fromarray(img8, 'RGB')
    img8 = img8.resize((224, 224))
    dataset.append(np.array(img8))
    label.append(0) # append number label

```

### Examples of Augmented Images from the MS and Brain Tumor Datasets

![random_images](https://github.com/jsmit659/Biomed-Final/assets/113131600/4d852502-d564-4523-9b9e-4ece727efc68)


### VGG19 Model
The first of the two viable models is the VGG19 model. Here, we use the imagenet weights, skip connections, Dropout layers with a dropout rate of 0.5, and L2 Regularization. Additional settings of note include a learning rate of 0.00001, batch size of 8, and the model was ran for 200 epochs.
```python
# define the input tensor
input_tensor = keras.layers.Input(shape=(224,224,3))

# build the convolutional base
vgg19 = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
for layer in vgg19.layers: 
    layer.trainable = False

# add skip connections
x = vgg19.output
x = keras.layers.Flatten()(x)
x = keras.layers.BatchNormalization()(x)

# add L2 regularization to the fully connected layers
x1 = keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x1 = keras.layers.Dropout(0.5)(x1)
x2 = keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x1)
x2 = keras.layers.Dropout(0.5)(x2)
x3 = keras.layers.Dense(132, activation='relu', kernel_regularizer=l2(0.01))(x2)
x3 = keras.layers.Dropout(0.5)(x3)

x = keras.layers.Concatenate()([x, x1, x2, x3])
predictions = keras.layers.Dense(8, activation='softmax')(x)

# create the model
full_model = keras.models.Model(inputs=input_tensor, outputs=predictions)

# compile the model
full_model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adamax(learning_rate=0.00001),
                    metrics=['accuracy'])

# print the model summary
full_model.summary()

# train the model with callbacks
checkp = ModelCheckpoint('./unet_model.h5', monitor='accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True)
history = full_model.fit(X_train, y_train, batch_size=8, epochs=200, validation_data=(X_test, y_test), workers=10, callbacks=[checkp])
```

This model produced suprising results with 99% Accuracy and 93.4% Validation Accuracy. While the accuracy was promising, the Validaiton Loss never got below 0.2160 loss. The Accuracy and Loss curves are shown below along with the confusion matrix and accuracy metrics. 

![download](https://github.com/jsmit659/Biomed-Final/assets/113131600/5fafc214-40ed-4bba-a97f-2f2ec1f9270d)
![Screenshot from 2023-05-10 19-54-32](https://github.com/jsmit659/Biomed-Final/assets/113131600/cae5e21f-4356-4a52-b152-d1dedb2c6ea1)

### DenseNET Model
In addition of the VGG19 model, DenseNET was also explored. This model produced much better results with lower validation loss and slightly better validation loss, achieving a high of 95.08% validation accuracy and 0.1577 validation loss. 
```python
# define the input tensor
input_tensor = keras.layers.Input(shape=(224,224,3))

# build the convolutional base
densenet = keras.applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)
for layer in densenet.layers: 
    layer.trainable = False

# add skip connections
x = densenet.output
x = keras.layers.Flatten()(x)
x = keras.layers.BatchNormalization()(x)

# add L2 regularization to the fully connected layers
x1 = keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x1 = keras.layers.Dropout(0.5)(x1)
x2 = keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x1)
x2 = keras.layers.Dropout(0.5)(x2)
x3 = keras.layers.Dense(132, activation='relu', kernel_regularizer=l2(0.01))(x2)
x3 = keras.layers.Dropout(0.5)(x3)

x = keras.layers.Concatenate()([x, x1, x2, x3])
predictions = keras.layers.Dense(8, activation='softmax')(x)

# create the model
full_model = keras.models.Model(inputs=input_tensor, outputs=predictions)

# compile the model
full_model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adamax(learning_rate=0.00001),
                    metrics=['accuracy'])

# print the model summary
full_model.summary()

# train the model with callbacks
checkp = ModelCheckpoint('./unet_model.h5', monitor='accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True)
history = full_model.fit(X_train, y_train, batch_size=8, epochs=150, validation_data=(X_test, y_test), workers=10, callbacks=[checkp])

```
Below are the Accuracy and Loss curves, along with the confusion matrix and accuracy metrics. 

![Screenshot from 2023-05-10 19-34-12](https://github.com/jsmit659/Biomed-Final/assets/113131600/621b288c-3f61-46af-8bcd-8b43e5cf05b9)

![Screenshot from 2023-05-10 19-59-13](https://github.com/jsmit659/Biomed-Final/assets/113131600/65f58650-3475-4038-832d-acde97bcb42c)

### Checking Results 
From the code block below, we can import the saved .h5 file from our model and check the results. This block will choose an image file at random and display the image, file path, and prediction. This way, one can visually check the accuracy of the model. 

```python
import tensorflow as tf
import numpy as np
import random
import os
from PIL import Image

# Choose a random file path and a random image from it
chosen_path = random.choice(paths)
file_names = os.listdir(chosen_path)
chosen_file_name = random.choice(file_names)
img_path = os.path.join(chosen_path, chosen_file_name)

# Load the saved model
model = tf.keras.models.load_model('FILE PATH to H5 file')

# Load the image you want to classify
img = Image.open(img_path).convert('RGB')
img = img.resize((224, 224))  # Resize the image to match the input size of the model

# Preprocess the image
img_arr = np.array(img)
img_arr = img_arr.astype('float32')
img_arr /= 255.0
img_arr = np.expand_dims(img_arr, axis=0)  # Add a batch dimension

# Make a prediction using the model
pred = model.predict(img_arr)

class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor', 'Healthy Axial', 'MS Axial', 'Healthy Sagittal', 'MS Sagittal']
predicted_class_name = class_names[np.argmax(pred)]

# You can also get the probability of each class
class_probabilities = pred[0]

# Print the chosen file path, the predicted class, and the probability of each class
img.show()
print('Chosen file path:', chosen_path)
print('Predicted class:', predicted_class_name)

```
Below is an example of the manual checking. 

![Screenshot from 2023-05-10 19-44-44](https://github.com/jsmit659/Biomed-Final/assets/113131600/bd97cfa6-6e36-4177-ac44-22e4f1ea6bb6)
