import glob
import os, sys
import numpy as np
from skimage import io,transform


from PIL import Image

path='Data/ModelNet20/train/'

def read_img(path):
    c = os.listdir(path)
    print(c)
    cate_depth = [path+c[x]+'/D/' for x in [2,5]]   #airplane_/D/
    #cate_sil = [path+c[y]+'/S/' for y in [16]]      #airplane_/S/

    data = np.zeros([2])
    dep = np.zeros([2])
    
    #GET the depth maps
    for idx,folder in enumerate(cate_depth):
        print('%s:' % folder)
        nu = 0
        all_depth = glob.glob(folder+'/*.jpg')   #所有的深度图路径
        if len(all_depth) % 18 != 0:
            print('!!!!Error directory:', folder)

        for i in range(int(len(all_depth)/18)):
            tmp_data =[]    # [1,18,256,256]
            tmp_dep = []
            tmpdep = []
            tmpdata = []
            
            nu = nu + 1
            for j in range(18): 
                img1 = io.imread(all_depth[i*18 + j])
                tmpdep.append(img1)                    #[18, 256*]
                if j < 6:
                    tmpdata.append(img1)               #[6, 256*]
            tmp_dep.append(tmpdep)   #[1,18,256,256]
            tmp_data.append(tmpdata) #[1,18 256*]
            to_append_data = np.asarray(tmp_data)
            to_append_dep = np.asarray(tmp_dep)
            if idx == 0 and i == 0:
                data = to_append_data
                dep = to_append_dep
            else:
                data = np.vstack((data,to_append_data))
                dep = np.vstack((dep, to_append_dep))
            to_append_dep = []
            to_append_data = []
        print('The number is %d.' % nu)
    
    #Get the sils
        for idx1,folder1 in enumerate(cate_sil):
        print('%s:' % folder1)
        nu1 = 0
        sil = np.zeros([2])
        all_sil = glob.glob(folder1+'/*.jpg')   #所有的剪影图路径
        for i in range(int(len(all_sil)/18)):
            tmp_sil = []
            tmpsil = []
            nu1 += 1
            for j in range(18): 
                img2 = io.imread(all_sil[i*18 + j])
                tmpsil.append(img2)       #18*256*256
            tmp_sil.append(tmpsil) #1,18,256...
            to_append_sil = np.asarray(tmp_sil)
            if i == 0 and idx1 == 0:
                sil = to_append_sil
            else:
                sil = np.vstack((sil, to_append_sil))
        print(nu1)
    
    return data, dep, sil

print('=========== Reading the data... ===========')
data, depth = read_img(path)

print(data.shape)
print(depth.shape)
#print(silhouette.shape)

def read_img1(path):
    c = os.listdir(path)
    cate_depth = [path+c[x]+'/D/' for x in range(10)]   #airplane_/D/
    cate_sil = [path+c[y]+'/S/' for y in range(10)]      #airplane_/S/

    data = []
    dep = []
    sil = []
    #GET the depth maps
    for idx,folder in enumerate(cate_depth):
        print('%s:' % folder)
        nu = 0
        all_depth = glob.glob(folder+'/*.jpg')   #所有的深度图路径
        if len(all_depth) % 18 != 0:
            print('!!!!Error directory:', folder)
        for i in range(int(len(all_depth)/18)):
            tmp = []
            tmpdata = []
            sid = rd.randint(0,17)
            nu = nu + 1
            for j in range(18): 
                img1 = io.imread(all_depth[i*18 + j])
                tmp.append(img1)
                if j < 6:
                    tmpdata.append(img1)
            dep.append(tmp)
            data.append(tmpdata)
        print('The number is %d.' % nu)

    #Get the sils
    for idx1,folder1 in enumerate(cate_sil):
        print('%s:' % folder1)
        nu1 = 0
        all_sil = glob.glob(folder1+'/*.jpg')   #所有的剪影图路径
        for i in range(int(len(all_sil)/18)):
            tmp = []
            nu1 += 1
            for j in range(18): 
                img2 = io.imread(all_sil[i*18 + j])
                tmp.append(img2)
            sil.append(tmp)
        print(nu1)
    return np.asarray(data), np.asarray(dep), np.asarray(sil)