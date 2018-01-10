import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    
    # Set tensor and model names:
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Load TF model:
    # API: https://www.tensorflow.org/api_docs/python/tf/saved_model/loader/load
    model= tf.saved_model.loader.load( sess, [vgg_tag], vgg_path )

    # Get graph from TF model:
    graph = tf.get_default_graph()

    # Get tensors from graph:
    vgg_input_tensor = graph.get_tensor_by_name( vgg_input_tensor_name )
    vgg_keep_prob_tensor = graph.get_tensor_by_name( vgg_keep_prob_tensor_name )
    vgg_layer3_out_tensor = graph.get_tensor_by_name( vgg_layer3_out_tensor_name )
    vgg_layer4_out_tensor = graph.get_tensor_by_name( vgg_layer4_out_tensor_name )
    vgg_layer7_out_tensor = graph.get_tensor_by_name( vgg_layer7_out_tensor_name )

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Kernel sizes, stride sizes and padding information is taken from:
    # Long, Shelhamer, Darrell: 
    # "Fully Convolutional Networks for Semantic Segmentation"
    # (page 5, fiure 3).

    # Layer 7: 
    # 1x1 convolution of layer 7 in order to get logits and model depth of layer 7
    # conv2d params: inputs, num_outputs, kernel_size [,optional params]
    h7_1x1 = tf.layers.conv2d( vgg_layer7_out, num_classes, kernel_size=1, 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01) ) 

    # Layer 8: 
    # Transpose convolution of layer 7
    # API: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d_transpose
    h8 = tf.layers.conv2d_transpose( h7_1x1, num_classes, kernel_size=4, strides=(2,2), padding='same', 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01) )

    # Layer 9: 
    # Get hidden layer 4, add it to layer 8, and get a transpose convolution
    h4_1x1 = tf.layers.conv2d( vgg_layer4_out, num_classes, kernel_size=1, 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01) ) 
    h8_h4 = tf.add( h8, h4_1x1 )
    h9 = tf.layers.conv2d_transpose( h8_h4, num_classes, kernel_size=4, strides=(2,2), padding='same', 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01) )

    # Layer 10: 
    # Get hidden layer 3, add it to layer 9, and get a transpose convolution
    h3_1x1 = tf.layers.conv2d( vgg_layer3_out, num_classes, kernel_size=1, 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01) ) 
    h9_h3 = tf.add( h9, h3_1x1 )
    h10 = tf.layers.conv2d_transpose( h9_h3, num_classes, kernel_size=16, strides=(8,8), padding='same', 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01) )

    # Return hidden layer 10:
    return h10

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    # Reshape prediction and ground truth tensor
    y_Prediction_logits = tf.reshape( nn_last_layer, ( -1, num_classes ) ) 
    y_GroundTruth = tf.reshape( correct_label, ( -1, num_classes ) )

    # Cross entropy and loss:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits=y_Prediction_logits, labels=y_GroundTruth )
    cross_entropy_loss = tf.reduce_mean( cross_entropy )

    # Adam optimization to minimize loss:
    train_op = tf.train.AdamOptimizer( learning_rate ).minimize( cross_entropy_loss )

    return y_Prediction_logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())
    
    for epoch in range( epochs ):
        
        total_loss = 0 # []
        images = 0

        for image, label in  get_batches_fn( batch_size ):

            images = images + len( image )

            feed_dict = {
                input_image: image, 
                correct_label: label, 
                keep_prob: 0.5 # keep_prob
            }
            
            loss, _ = sess.run( [cross_entropy_loss, train_op], feed_dict=feed_dict )

            total_loss = total_loss + loss # total_loss.append( loss ) 

        avg_loss = total_loss / images

        print( "Loss: " + str( avg_loss ) + "\n" )

tests.test_train_nn(train_nn)

#"""
def save_model( sess, saver_path, graph_path, graph_name ):
    # API: https://www.tensorflow.org/api_docs/python/tf/train/Saver
    saver = tf.train.Saver()
    saver.save( sess, saver_path )
    # API: https://www.tensorflow.org/api_docs/python/tf/train/write_graph
    tf.train.write_graph( sess.graph_def, graph_path, graph_name, False )
#"""

def run():
    # Fixed variables:
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Model hyperparameters:
    learning_rate = tf.constant( 0.001 ) # 0.01 # To optimize the model further, lr could be larger or epochs should be increases
    epochs = 10 # 1 # 20 took ver long --> canceled after approx. 26 hours
    batch_size = 1

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        shape = [None,image_shape[0],image_shape[1],3]
        correct_label = tf.placeholder( tf.float32, [None,image_shape[0],image_shape[1],num_classes] )
        vgg_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg( sess, vgg_path )
        nn_last_layer = layers( vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes )
        logits, train_op, cross_entropy_loss = optimize( nn_last_layer, correct_label, learning_rate, num_classes )

        # TODO: Train NN using the train_nn function
        #keep_prob = tf.constant( 0.5 ) # Above I am reading in a tensor instead of the dropout- or keep-rate
        train_nn( sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input, 
            correct_label, keep_prob, learning_rate )
            # correct_label, keep_prob, learning_rate, runs_dir, data_dir, image_shape, logits )

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, vgg_input)
        # save_model( './fcn_model/saver/saver_fcn.pb', './fcn_model/graph/', 'graph_fcn.pb' )

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
