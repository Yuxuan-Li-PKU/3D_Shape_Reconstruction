from skimage import io,transform
import tensorflow as tf
import numpy as np
from PIL import Image

def read_one_image(path):
    a = []
    img = io.imread(path)
    a.append(img)
    a = np.asarray(a).astype(np.float32)
    return a

with tf.Session() as sess:
    path = 'test.png'
    data = read_one_image(path)
    print(data.shape)

    saver = tf.train.import_meta_graph('Model/RCmodel.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('Model/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    G = graph.get_tensor_by_name("G_val:0")
    S = graph.get_tensor_by_name("S_val:0")

    O_G, O_S = sess.run([G,S],feed_dict)
    print(O_G.shape)
    print(O_S.shape)

    save_path = 'test/'
    for l in range(20):
        d_name = 'depth_'+str(l).zfill(2)+'.png'
        s_name = 'sil_'+str(l).zfill(2)+'.png'
        d_im = np.array(O_G[0][l]*255).astype(np.uint8)
        s_im = np.array(O_S[0][l]*255).astype(np.uint8)
        save_d = Image.fromarray(d_im)
        save_s = Image.fromarray(s_im)
        save_d.save(save_path+d_name)
        save_s.save(save_path+s_name)
    
    

