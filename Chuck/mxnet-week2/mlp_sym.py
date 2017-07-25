import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer,input_kernel,n_filter,activation,input_stride = (),pool_kernel = None,pool_stride = (),pool_type = None,pooling=False):
    """
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param input_kernel: convolution kernel description (#,#,#) with 2-d plus channel
    :param input_stride: convolution stride size description (#,#,#) with 2-d plus channel
    :param n_filter: number of filters in convolution
    :param activation: activation function type for convolution layer
    :param pool_kernel: pooling kernel
    :param pool_stride: pooling stride
    :param pool_type: pooling type
    :param pooling: the switch for adding pooling layer
    :return: the symbol as output
    """
    l = mx.sym.Convolution(data=input_layer,kernel = input_kernel,stride = input_stride,num_filter=n_filter)
    if pooling is False:
        l = mx.sym.Activation(data=l, act_type=activation)
    else:
        try:
            l = mx.sym.Activation(data=l, act_type=activation)
            l = mx.sym.Pooling(data=l, pool_type=pool_type, kernel=pool_kernel, stride=pool_stride)
        except EnvironmentError:
            print "Please add input for pooling kernel and pooling activation"
    return l


# Optional
def inception_layer():
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    pass




def get_conv_sym():

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?
    # first conv layer

    # l = conv_layer(input_layer=data, input_kernel=(5, 5), n_filter=20,
    #                   activation="tanh", pool_kernel=(2,2),pool_stride=(2,2),pool_type="max",pooling=True)
    l = conv_layer(input_layer=data, input_kernel=(5, 5), n_filter=50,activation="relu")
    # l = conv_layer(input_layer=l, input_kernel=(5, 5), n_filter=50,
    #                   activation="tanh", pool_kernel=(2,2),pool_stride=(2,2),pool_type="max",pooling=True)
    data_f = mx.sym.flatten(data=l)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    #l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    conv = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return conv



