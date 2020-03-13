import tensorflow as tf 


#Type == 'first' and 'no_preact' or 'both_preact'是否是第一层应该有影响
def Residual_Block(inputT, inputC, outputC, stride, name, First_block=False, wether_deconv =False, batch_size=4):
    #inputT: NxHxWxC
    #inputC: C
    #stride: int
    #both preact
    if First_block==False:
        with tf.variable_scope(name+'Pre_activation'):
            bn1 = mybatch_norm(name+'preact', inputT, inputC)
            ori_input = tf.nn.relu(bn1)
    else:
        ori_input = inputT

    if stride == 2:
        if wether_deconv == False:  #two convlution layers
            conv1 = myConv2d(name+'Resi_conv1', ori_input, [4,4, inputC, outputC], stride, skip=2)
            bn2 = mybatch_norm(name+'Resi_conv1',conv1, outputC)
            relu2 = tf.nn.relu(bn2)
            conv2 = myConv2d(name+'Resi_conv2', relu2, [3,3, outputC, outputC], 1)
        else:
            conv1 = myDeConv2d(name+'Resi_deconv1', ori_input, outputC, 4, stride)
            bn2 = mybatch_norm(name+'Resi_deconv1', conv1, outputC)
            relu2 = tf.nn.relu(bn2)
            conv2 = myDeConv2d(name+'Resi_deconv2', relu2, outputC, 3, 1)
    else: #unconv
        conv1 = myDeConv2d(name+'Resi_deconv1', ori_input, outputC, 4, 1, pad = 'SAME')
        bn2 = mybatch_norm(name+'Resi_deconv1', conv1, outputC)
        relu2 = tf.nn.relu(bn2)
        conv2 = myDeConv2d(name+'Resi_deconv2', relu2, outputC, 3, 1)

    #add shortcut and conv-layer's output: f(x) = h(x) + x.
    if inputC != outputC:
        if wether_deconv == False:
            cut = myConv2d(name+'Resi_cut1', ori_input, [4,4, inputC, outputC], stride, skip=2)
        else:
            cut = myDeConv2d(name+'Resi_cut2', ori_input, outputC, 4, max(1,stride))
    else:
        cut = ori_input
    fx = cut + conv2  
    return fx


def get_encoder(input_tensor, training = True):
    #fetch the params
    #input_tensor=(batch,h,w,c) 4x224x224x1
    input_tensor = tf.expand_dims(input_tensor, 3)
    #convlution
    #print(input_tensor.shape)

    with tf.variable_scope('Encod_conv1'):
        conv1_weights = tf.get_variable("weights",[4,4,1,74*4],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[74*4],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
        #E_DLconv1 = tf.nn.atrous_conv2d(input_tensor, conv1_weights, 3, padding='SAME')
        E_DLconv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides= [1,2,2,1],dilations=[1,1,2,2], padding='SAME',data_format='NHWC')
        conv1_BN = tf.layers.batch_normalization(E_DLconv1,training=training)      #!!!bn layer which is correct?
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1_BN,conv1_biases))      #wether to add bias really makes diference?
    #relu1 size: 112x112
    #compute: out_size = (input_size+1-3-(input_size-2)!=odd) / stride 向下取整. 因为使用了一次dilated conv
    #卷积核大小为4，则用'SAME'进行padding的时候 上下左右各补一行
    #residual Block -basic block is defined in the function: BasicBlock()

    print('relu1:', relu1.shape)
    en_resblock1 = Residual_Block(relu1, 74*4, 74*6, 2, 'en_resblock1', First_block=True)
    print('en_resblock1:', en_resblock1.shape)
    #output size: 56x56
    en_resblock2 = Residual_Block(en_resblock1, 74*6, 74*8, 2, 'en_resblock2')
    print('en_resblock2:', en_resblock2.shape)
    #output size: 28x28
    en_resblock3 = Residual_Block(en_resblock2, 74*8, 74*6, 2, 'en_resblock3')
    print('en_resblock3:', en_resblock3.shape)
    #output size: 14x14
    en_resblock4 = Residual_Block(en_resblock3, 74*6, 74*3, 2, 'en_resblock4')
    print('en_resblock4:', en_resblock4.shape)
    #output size: 7x7

    en_resblock5 = Residual_Block(en_resblock4, 74*3, 74, 2, 'en_resblock5')
    print('en_resblock5', en_resblock5.shape)

    bn2 = mybatch_norm('bn2',en_resblock5, 74)
    #output: [4, 4, 4, 74]
    print('bn2:', bn2.shape)

    #change to 1-D tensor
    fc_input = tf.reshape(bn2,[-1, 4*4*74])

    #Fully connected layer 1
    en_fc1_1 = myFC('en_fc1_1', fc_input, 74*4*4, 74*4*2)
    en_fc1_1_relu = tf.nn.relu(en_fc1_1)
    en_fc1_2 = myFC('en_fc1_2', en_fc1_1_relu, 74*4*2, 100) #means
    #en_softmax1 = tf.nn.sigmoid(en_fc1_2)
    means = en_fc1_2

    #Fully connected layer 2
    en_fc2_1 = myFC('en_fc2_1', fc_input, 74*4*4, 74*4*2)
    en_fc2_1_relu = tf.nn.relu(en_fc2_1)
    en_fc2_2 = myFC('en_fc2_2', en_fc2_1_relu, 74*4*2, 100) #log of variance
    #en_softmax2 = tf.nn.sigmoid(en_fc2_2)
    log_var = en_fc2_2

    '''
    output of encoder: two 2-D tensors [batch, 100]
    If it's conditional network, the output should be three independent tensors.
    The new one is [batch, catagories]
    '''

    return means, log_var


def get_decoder(input_tensor, training = True):
    '''
    input: z = [batch, ]
    反卷积过程：ow = ksize + s x (w - 1) - paddings
    resi deconv1:  4 + 2w - 2 - 2x1 = 2w
    resi deconv2:  3 + ww - 1 - 2 = 2w
    special: 第一次反卷积图像大小4x4，卷积核大小4x4
             4 + 3x1 - 2x0 = 7
    '''
    de_fc1 = myFC('de_fc1', input_tensor, 100, 74*7*7*2)
    de_fc1_relu = tf.nn.relu(de_fc1)
    #de_fc2 = myFC('de_fc2', de_fc1_relu, 74*2*2, 74*7*7*2)
    #reshape
    de_reshaped = tf.reshape(de_fc1, [-1,7,7,74*2])
    de_bn1 = tf.nn.relu(mybatch_norm('de_bn1' ,de_reshaped, 74*2))
    #output size: [batch,7,7,148]
    print('de_bn1:', de_bn1.shape)
    de_resblock1 = Residual_Block(de_bn1, 74*2, 74*6, -1, 'de_resblock1', First_block=True, wether_deconv = True)
    #output size: 7x7
    print('de_resblock1:', de_resblock1.shape)
    de_resblock2 = Residual_Block(de_resblock1, 74*6, 74*8, 2, 'de_resblock2', wether_deconv = True)
    #output size: 14x14
    print('de_resblock2:', de_resblock2.shape)
    de_resblock3 = Residual_Block(de_resblock2, 74*8, 74*7, 2, 'de_resblock3', wether_deconv = True)
    print('de_resblock3:', de_resblock3.shape)
    de_bn2 = mybatch_norm('de_bn2', de_resblock3, 74*7)
    #output size: 28x28

    de_deconv1 = myDeConv2d('de_deconv1', de_bn2, 74*4, 4, 2)  #74*6
    de_relu1 = tf.nn.relu(mybatch_norm('de_relu1', de_deconv1, 74*4))
    #output size: 56x56
    print('de_relu1:', de_relu1.shape)
    de_deconv2 = myDeConv2d('de_deconv2', de_relu1, 74, 4, 2)  #74*4
    de_relu2 = tf.nn.relu(mybatch_norm('de_relu2', de_deconv2, 74))  
    #output size: 112x112
    print('de_relu2', de_relu2.shape)



    #output layer 1 generates the depth maps
    with tf.variable_scope('out1'):
        output1 = tf.contrib.layers.conv2d_transpose(de_relu2, 20, 4, 2)
        depth_maps = tf.nn.sigmoid(output1)
        depth_maps = tf.reshape(depth_maps, [-1, 20, 224, 224])
    #output size: 224x224

    #output layer 2 generates the sillhouettes
    with tf.variable_scope('out2'):
        output2 = tf.contrib.layers.conv2d_transpose(de_relu2, 20, 4, 2)
        sillhouettes = tf.nn.sigmoid(output2)
        sillhouettes = tf.reshape(sillhouettes, [-1, 20, 224, 224])
    #output size: 224x224

    return depth_maps, sillhouettes


def myConv2d(name, inputs, filter_, stride, skip=1):
    '''
    name: the layer's name
    inputs: the input tensor: NxHxWxC 
    filter_: 4D tensor:[f_h, f_w, inChannels, outChannels]
    stride: int, h's stride is same with w's stride 
    '''
    with tf.variable_scope(name):
        filter_weight = tf.get_variable("filter_weight", filter_, initializer=tf.truncated_normal_initializer(stddev=0.1))
        myConv = tf.nn.conv2d(inputs, filter_weight, [1,stride,stride,1],dilations=[1,1,skip,skip],padding = 'SAME')
        #bias
    return myConv


def myDeConv2d(name, inputs, outputC, ksize, stride, pad='SAME'):  
    '''
    name: the layer's name
    inputs: the input tensor: NxHxWxC 
    filter_: 4D tensor:[f_h, f_w, outChannels, inChannels]
    stride: int, h's stride is same with w's stride 
    
    with tf.variable_scope(name):
        filter_weight = tf.get_variable("filter_weight", filter_, initializer=tf.truncated_normal_initializer(stddev=0.1))
        myDeConv = tf.nn.conv2d_transpose(inputs, filter_weight, outShape, [1,stride,stride,1], padding = pad)
        #bias
    return myDeConv
    '''
    with tf.variable_scope(name):
        deconv = tf.contrib.layers.conv2d_transpose(inputs, outputC, [ksize,ksize], stride, padding=pad)
    return deconv


def myFC(name, inputT, inputL, outputL):
    '''
    输入输出的长度，[batch x inL] --> [batch x outL]
    '''
    with tf.variable_scope(name):
        fc_weight = tf.get_variable('weight', [inputL, outputL], initializer = tf.truncated_normal_initializer(stddev=0.1))
        fc_bias = tf.get_variable('bias', [outputL], initializer = tf.constant_initializer(0.1))

        fc = tf.nn.bias_add(tf.matmul(inputT, fc_weight), fc_bias)
        
        return fc


def mybatch_norm(name, input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor == channel
    :return: the 4D tensor after being normalized
    '''

    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

    return bn_layer