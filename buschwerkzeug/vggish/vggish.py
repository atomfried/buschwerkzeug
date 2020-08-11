import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.contrib._warning = None

from . import vggish_input, vggish_params, vggish_postprocess, vggish_slim
import numpy as np

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

class VGGish:
    def __init__(self, fname_model, fname_pca_params):
        tf.get_logger().setLevel('WARNING')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session() 
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.sess, fname_model)
            self.features_tensor = self.sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        self.pproc = vggish_postprocess.Postprocessor(fname_pca_params)

    def __del__(self):
        if self.sess:
            self.sess.close()

    def features(self, wav, fs):
        win_len = int((vggish_params.EXAMPLE_WINDOW_SECONDS+vggish_params.STFT_WINDOW_LENGTH_SECONDS)*fs)
        if len(wav) < win_len:
            #print('WARNING: sample too short, padding with zero.')
            wav = np.pad(wav, (0, win_len-len(wav)), mode='constant')
        examples_batch = vggish_input.waveform_to_examples(wav[:win_len], fs)
        with self.graph.as_default():
            [embedding_batch] = self.sess.run([self.embedding_tensor],feed_dict={self.features_tensor: examples_batch})
        [r] = self.pproc.postprocess(embedding_batch)
        return r/255
