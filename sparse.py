import tensorflow as tf

def sparse_conv(tensor,binary_mask = None,filters=32,kernel_size=3,strides=2):

    if binary_mask == None: #first layer has no binary mask
        b,h,w,c = tensor.get_shape()
        channels=tf.split(tensor,c,axis=3)
        #assume that if one channel has no information, ALL CHANNELS HAVE NO INFORMATION
        binary_mask = tf.where(tf.equal(channels[0], 0), tf.zeros_like(channels[0]), tf.ones_like(channels[0])) #mask should only have the size of (B,H,W,1)

    features = tf.multiply(tensor,binary_mask)
    features = tf.layers.conv2d(features, filters=filters, kernel_size=kernel_size, strides=(strides, strides), trainable=True, use_bias=False, padding="same")

    norm = tf.layers.conv2d(binary_mask, filters=filters,kernel_size=kernel_size,strides=(strides, strides),kernel_initializer=tf.ones_initializer(),trainable=False,use_bias=False,padding="same")
    norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.reciprocal(norm))
    _,_,_,bias_size = norm.get_shape()

    b = tf.Variable(tf.constant(0.01, shape=[bias_size]))
    feature = tf.multiply(features,norm)+b
    mask = tf.layers.max_pooling2d(binary_mask,strides = strides,pool_size=3,padding="same")

    return feature,mask



image = tf.placeholder(tf.float32, shape=[None,64,64,2], name="input_image")
b_mask = tf.placeholder(tf.float32, shape=[None,64,64,1], name="binary_mask")
features,b_mask = sparse_conv(image)
features,b_mask = sparse_conv(features,binary_mask=b_mask)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

