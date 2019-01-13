# Dhruvik Patel
# dp811 - 163001797


import os; 
os.environ['KERAS_BACKEND'] = 'theano'
import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
 
vgg_model = vgg16.VGG16(weights='imagenet')

inception_model = inception_v3.InceptionV3(weights='imagenet')

resnet_model = resnet50.ResNet50(weights='imagenet')

mobilenet_model = mobilenet.MobileNet(weights='imagenet')

 
filename = 'cat.jpg'

original = load_img(filename, target_size=(224, 224))
print('PIL image size',original.size)
plt.imshow(original)
plt.show()
 
numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size',numpy_image.shape)


image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))


processed_image = resnet50.preprocess_input(image_batch.copy())

predictions = resnet_model.predict(processed_image)

label = decode_predictions(predictions)
print (label)
 
