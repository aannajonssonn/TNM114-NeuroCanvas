import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.applications import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras import backend as K

# Load pre-trained VGG19 model (excluding fully connected layers)
def load_vgg19():
    # Load VGG19 model with pre-trained weights
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_nrows, img_ncols, 3))

    # Create a model that outputs the specific layers needed for style transfer
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [base_model.get_layer(layer).output for layer in style_layers]

    return models.Model(inputs=base_model.input, outputs=outputs)

# Define content loss
def content_loss(base, combination):
    # Reshape combination to match the shape of base
    combination_reshaped = K.resize_images(combination, 16, 16, data_format='channels_last')  # Adjust dimensions as needed
    return K.sum(K.square(combination_reshaped - base))


# Define style loss using Gram matrix
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# Define Gram matrix
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# Define total variation loss to reduce noise in generated image
def total_variation_loss(x):
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# Combine loss components
def total_loss(x):
    content = content_weight * content_loss(base_content, x)
    style = style_weight * sum(style_loss(base_style, s) for s in x)  # Iterate over style layers
    variation = total_variation_weight * total_variation_loss(x)
    return content + style + variation

# Define image preprocessing and deprocessing functions
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Set parameters
content_image_path = 'content\Green_Sea_Turtle_grazing_seagrass.jpg'
style_image_path = 'style\The_Great_Wave_off_Kanagawa.jpg'
img_nrows, img_ncols = 512, 512
content_weight = 0.025
style_weight = 1.0
total_variation_weight = 1.0

# Load content and style images
base_content = preprocess_image(content_image_path)
base_style = preprocess_image(style_image_path)

#print(base_content.shape) # (1, 512, 512, 3)
#print(base_style.shape) # (1, 512, 512, 3)

# Create target image variable and initialize it with content image
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# Build the VGG19 network with shared weights
model = load_vgg19()
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
features = [outputs_dict[layer_name] for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']]
loss = total_loss(features)

# Compute gradients of the generated image with respect to the loss
grads = K.gradients(loss, combination_image)[0]

# Create function to fetch loss and gradients
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

# Define the optimization routine
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_nrows, img_ncols, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# Style transfer optimization loop
from scipy.optimize import fmin_l_bfgs_b
import time

x = preprocess_image(content_image_path).flatten()

epochs = 10

for i in range(epochs):
    print(f'Iteration {i+1}/{epochs}')
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print(f'Loss: {min_val}')
    img = x.copy().reshape((img_nrows, img_ncols, 3))
    img = deprocess_image(img)
    end_time = time.time()
    print(f'Iteration {i+1} completed in {end_time - start_time}s')

# Save the final stylized image
final_image_path = 'generated/saved.jpg'
image.save_img(final_image_path, img)
