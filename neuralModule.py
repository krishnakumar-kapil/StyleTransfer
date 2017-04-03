import tensorflow as tf
import numpy as np 
import scipy.io  
import argparse 
import struct
import errno
import time                       
import cv2
import os


vgg19_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


# model weights is the name of the file to pass in 
def build_vgg19(input_img, model_weights_path):
    print "building vgg19 for input img"

    net = {}
    _, h, w, d = input_img.shape
    
    vgg_rawnet     = scipy.io.loadmat(model_weights_path)
    vgg_layers     = vgg_rawnet['layers'][0]
    net['input']   = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

    net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
    net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))

    net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
    net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))

    net['pool1']   = pool_layer('pool1', net['relu1_2'])

    net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))

    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
    net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))

    net['pool2']   = pool_layer('pool2', net['relu2_2'])

    net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
    net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))

    net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
    net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))

    net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
    net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))

    net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
    net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))

    net['pool3']   = pool_layer('pool3', net['relu3_4'])

    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))

    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))

    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))

    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))

    net['pool4']   = pool_layer('pool4', net['relu4_4'])

    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))

    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))

    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))

    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))

    net['pool5']   = pool_layer('pool5', net['relu5_4'])

    return net



def conv_layer(layer_name, layer_input, W):
    return tf.nn.conv2d(layer_input, W, strides=[1,1,1,1], padding='SAME')

def relu_layer(layer_name, layer_input, b):
    return tf.nn.relu(layer_input+b)

def pool_layer(layer_name, layer_input):
    return tf.nn.max_pool(layer_input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def get_weights(vgg_layers, i):
    return tf.constant(vgg_layers[i][0][0][2][0][0])

def get_bias(vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    return b

def content_layer_loss(p, x):
    # Needs to be \sum (F^l - P^l)^2
    # should we put the things through the NN at this point?
    # what if they just pass in the results at that layer and call it p, x
    _, h,w,d = p.get_shape()
    # TODO: there are some other other constants 
    K = .5
    loss = K * tf.reduce_sum(tf.pow((p - x), 2))
    return loss

# TODO: definitely wrong
# Needs to be \sum_{l} w_l \sum (G^l - A^l)^2
def style_layer_loss(a,x):
    # where do we get the weights from
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    G = gram_matrix(x, M, N)
    A = gram_matrix(a, M, N)

    K = 1.0/(4 * N ** 2 * M ** 2)
    loss = K * tf.reduce_sum(tf.pow((G - A),2))
    return loss
   
def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G

def sum_style_loss(sess, net, imgs):
    total_style_loss = 0.0
    for img in imgs:
        sess.run(net['input'].assign(img))
        style_loss = 0
        for layer, weight in zip(args['style_layers'], args['style_layer_weights']):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a,x) * weight
        style_loss /= len(args['style_layers'])
        total_style_loss += style_loss
    return total_style_loss

def sum_content_loss(sess, net, content_img):
    content_loss = 0.0
    sess.run(net['input'].assign(content_img))
    for layer, weight in zip(args['content_layers'], args['content_layer_weights']):
         p = sess.run(net[layer])
         x = net[layer]
         p = tf.convert_to_tensor(p)
         content_loss += content_layer_loss(p,x)
    content_loss /= len(args['content_layers'])
    return content_loss

def get_optimizer(loss):
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      loss, method='L-BFGS-B',
      options={'maxiter': args['max_iterations'],
		  })
    #elif args.optimizer == 'adam':
#	optimizer = tf.train.AdamOptimizer(args.learning_rate)
    return optimizer

def minimize_with_lbfgs(sess, net, optimizer, init_img):
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)


def write_image(path, img):
    img = postprocess(img, vgg19_mean)
    cv2.imwrite(path, img)



def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):  
        os.makedirs(dir_path)

def write_image_output(output_img, content_img, style_imgs, init_img):
    out_dir = os.path.join(args['img_output_dir'], args['img_name'])
    maybe_make_directory(out_dir)
    img_path = os.path.join(out_dir, args['img_name']+'.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)
    index = 0
    for style_img in style_imgs:
	path = os.path.join(out_dir, 'style_'+str(index)+'.png')
	write_image(path, style_img)
	index += 1


def stylize(content_img, style_imgs, init_img):
    # with tf.device(args.device), tf.session() as sess
	
    with tf.Session() as sess:
        net = build_vgg19(content_img, args['model_weights_path'])
        L_style = sum_style_loss(sess, net, style_imgs)
        L_content = sum_content_loss(sess, net, content_img)

        L_total = args['alpha'] * L_content + args['beta'] * L_style

	optimizer = get_optimizer(L_total)

	minimize_with_lbfgs(sess, net, optimizer, init_img)
	
	output_img = sess.run(net['input'])

        # output_img = postprocess(output_img, vgg19_mean)

        write_image_output(output_img, content_img, style_imgs, init_img)

def postprocess(img, mean):
    img += mean
    img = img[0]
    img = np.clip(img,0,255).astype('uint8')
    img = img[...,::-1]
    return img

def preprocess(img, mean):
    img = img[...,::-1]
    img = img[np.newaxis,:,:,:]
    img -= mean
    return img



def render_single_image():
    content_img = get_content_image(args['content_img'])
    style_imgs = get_style_images(content_img)
    with tf.Graph().as_default():
	print('\n---- RENDERING SINGLE IMAGE ----\n')
	init_img = get_noise_image(args['noise_ratio'], content_img)
	tick = time.time()
	stylize(content_img, style_imgs, init_img)
	tock = time.time()
	print('Single image elapsed time: {}'.format(tock - tick))

def get_content_image(content_img):
    path = os.path.join(args['content_img_dir'], content_img)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    h, w, d = img.shape
    mx = args['max_size']
    # resize if > max size
    if h > w and h > mx:
	w = (float(mx) / float(h)) * w
	img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
	h = (float(mx) / float(w)) * h
	img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img, vgg19_mean)
    return img
	


def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)


def get_style_images(content_img):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in args['style_imgs']:
	path = os.path.join(args['style_imgs_dir'], style_fn)
	# bgr image
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	check_image(img, path)
	img = img.astype(np.float32)
	img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
	img = preprocess(img, vgg19_mean)
	style_imgs.append(img)
    return style_imgs


def get_noise_image(noise_ratio, content_img):
    np.random.seed(args['seed'])
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
    return img



def main():
    global args
    args = {}
    args['style_layers'] = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    args['style_layer_weights'] = [0.2, 0.2, 0.2, 0.2, 0.2]
    args['content_layers'] = ['conv4_2']
    args['content_layer_weights'] = [1.0]
    args['style_imgs_dir'] = "../Pictures/"
    args['max_size'] = 512
    args['img_output_dir'] = "."
    args['img_name'] = "stylizedImage"
    args['noise_ratio'] = 1.0
    args['seed'] = 1234
    args['alpha'] = 5e0
    args['beta'] = 1e4


    args['style_imgs'] = ["starryNight.jpg"]
    args['content_img'] = "boat.jpg"
    args['content_img_dir'] = "../Pictures/"
    args['model_weights_path'] = "imagenet-vgg-verydeep-19"
    args['max_iterations'] = 1000
    render_single_image()

if __name__ == '__main__':
    main()
