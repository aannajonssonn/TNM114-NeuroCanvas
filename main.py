# Parts of this code is under Copyright (c) 2019 Drew Szurko, and licensed with the MIT licence. 
# For more information about the license, see the LICENSE file in the root directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from absl import app
from absl import flags
from tensorflow import keras

import models
from utils import load_img

keras.backend.clear_session()

style_path = ['style\Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg', 'style\Van_gogh_cypruses.jpg'] # , 'style\Van_gogh_first_steps.jpg'

flags.DEFINE_string('content_path', 'content\Tuebingen_Neckarfront.jpg', 'Path to content image.')
#flags.DEFINE_string('style_path', 'style\Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg', 'Path to style image.')
flags.DEFINE_string('output_dir', 'generated', 'Output directory.')
flags.DEFINE_integer('epochs', 5, 'Epochs to train.')
flags.DEFINE_integer('steps_per_epoch', 100, 'Steps per epoch.')
flags.DEFINE_float('tv_weight', 1e8, 'Total variation weight.')
flags.DEFINE_float('content_weight', 10000, 'Content weight.')
flags.DEFINE_float('style_weight', 0.10, 'Style weight.')
flags.DEFINE_float('learning_rate', 0.02, 'Learning rate.')
flags.DEFINE_float('beta_1', 0.99, 'Beta 1.')
flags.DEFINE_float('beta_2', 0.999, 'Beta 2.')
flags.DEFINE_float('epsilon', 0.10, 'Epsilon.')
flags.DEFINE_float('max_dim', 512, 'Max dimension to crop I/O image.')
flags.mark_flags_as_required(['content_path'])
FLAGS = flags.FLAGS

flags.DEFINE_integer('loops', 1, 'Big Boy loop.')

def main(argv):
    del argv
    content_img = load_img(FLAGS.content_path)

    for i in style_path:
        style_img = load_img(i)
        mdl = models.StyleContent(content_img, style_img)
        mdl.train()
        content_img = load_img(f'generated/2img_loop{FLAGS.loops}.jpg')
        FLAGS.loops += 1

if __name__ == '__main__':
   app.run(main)