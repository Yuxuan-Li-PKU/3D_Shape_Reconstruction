
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os, sys
import glob


path = 'Data'

c = os.listdir(path+'/test_data')
'''
def Depth_G(path):
    for x in range(40):
        data_p = path+c[x]+'/'+c[x]+'_depth_rgb'
        depth_p = path+c[x]+'/D/'

        if not os.path.exists(depth_p):
            os.makedirs(depth_p)
        print('Saving depth maps of ' + c[x] + '...')
        nu = 0
        for im in glob.glob(data_p + '/'+ '*.png'):
            img = np.array(Image.open(im))
            img = img[:,:, 0]
            name = im.split('/')[-1]
            new_name = name.split('.')[-2] + '.png'
            new_im = Image.fromarray(img)
            new_im.save(depth_p+new_name)
            nu = nu + 1
        print('The number of %s is %d.' % (c[x], nu))
'''
z = range(18)
def Sil_G(path):
    for x in z:
        data_p = path+'/test_data/'+c[x]    #test/airplane
        sil_p = path+'/test_S/'+c[x]

        if not os.path.exists(sil_p):
            os.makedirs(sil_p)
        print('Saving silhouttes of ' + c[x] + '...')
        nu = 0
        for im in glob.glob(data_p +'/*.jpg'):
            img = np.array(Image.open(im))
            sil = img
            for i in range(256):
                for j in range(256):
                    if img[i][j] > 230:
                        sil[i][j] = 255
                    else:
                        sil[i][j] = 0
            name = im.split('\\')[-1]
            new_name = name
            new_im = Image.fromarray(sil)
            new_im.save(sil_p+'/'+new_name)
            nu = nu + 1
        print('The number of %s is %d.' % (c[x], nu))

Sil_G(path)



'''
if __name__ == '__main__':
    #Sil_G(path)
    im = 'aa.png'
    img = np.array(Image.open(im))
    print(img.shape)
    img = img[:,:, 0]
    #print(img[120:130, 120:130])
    #print(np.max(img))
    temp = np.zeros([224,224], dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            if img[i][j] > 10:
                temp[i][j] = 188
            else:
                temp[i][j] = 5

    new_name = 'b.png'

    new_im = Image.fromarray(temp)
    new_im.save(new_name)
'''