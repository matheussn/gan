import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image as img
import torch

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

print(device)

# def load_image(img_path: str):
#     image = img.load_img(img_path, target_size=(224, 224))
#     plt.imshow(image)
#     x = img.img_to_array(image)
#     return x
#
#
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# image = load_image('datasets/DSC_0897.jpg')
