import os, sys
import glob
from PIL import Image
ori_path = 'Data/ModelNet40/'
new_path = 'Data/ModelNet20/'

name = ['airplane','bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',  'door',  
'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'mantel','vase','sink']
#bed 515-508
#desk 0 dresser 0 bathtub 0
#mantel, sofa, vase
for x in name:
	path = ori_path + x + '/test'
	a = glob.glob(path+'/*.jpg')
	la = len(a)
	if(la % 66 != 0):
		print('ERROR:', path)
	nu = 0
	for i in range(la):
		if int(a[i].split('_')[-1].split('.')[0]) < 18:
			nu += 1
			img = Image.open(a[i])
			oir_dir_name = a[i].split('\\')
			ndn = oir_dir_name[0].split('/')    #D M air train
			new_name = ndn[0]+'/ModelNet20/'+ndn[3]+'/'+ndn[2]
			if not os.path.exists(new_name):
				os.makedirs(new_name)
			iname = oir_dir_name[1].split('_')
			new_iname = iname[0]+'_'+iname[1]+'_'+iname[2].split('.')[0].zfill(2) + '.jpg'
			img.save(new_name+'/'+new_iname)
	print(x, nu)		