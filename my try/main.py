# Imports
import style_transfer
from tools import load_img

# Import libraries
from absl import app
import tensorflow as tf
from tensorflow import keras
from tensorflow import io
from absl import flags

# Keras clear session
keras.backend.clear_session()

# FLAGS

# Style and content image
content_path = 'content\Tuebingen_Neckarfront.jpg'
style_path = 'style\Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'

max_dim = 512

# Mark required flags
#flags.mark_flags_as_required(['content_path', 'style_path'])

# Parse flags
#FLAGS = flags.FLAGS


# Main function
def main():
    # Load images
    content_img = load_img(content_path)
    style_img = load_img(style_path)

    # Train model
    style_transfer.train_model(content_img, style_img)

# Call main function
main()