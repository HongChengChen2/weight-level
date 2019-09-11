
dir_path= "data3/val/rabbits/" #193 -50 =143


import os
import random
import shutil

'''

files = os.listdir(dir_path)
sample = random.sample(files, 102)

for each in sample:
    file_path = os.path.join(dir_path, each)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)

    except PermissionError as e:
        pass
'''
#ramdom index



Imgnum = 100
files = os.listdir(dir_path) 
i = 0
L=random.sample(range(0,Imgnum),Imgnum)
filetype = ".jpg"
for filename in files:    
    portion = os.path.splitext(filename)
    if portion[1] == filetype:
        newname =dir_path+ 'rabbits' + str(L[i]) + filetype
        file = os.path.join(dir_path,filename)
        os.rename(file, newname)
        i = i+1
'''
from urllib import request
import socket
import traceback
import time

urls_file = 'txt_other/aft_50k_700/tiger.txt'   #beer 312/42
urls_file_2 = 'txt_other/aft/tiger.txt'   #beer 312/42

file = open(urls_file, 'r')
file_2 = open(urls_file_2, 'w')

i = 0

for line in file :
	if i%7 <4 :
		#print(i)
		i=i+1
		file_2.write(line)
	else:
		#print("else:",i)
		i=i+1

print(i)
file.close()
file_2.close()

#remove broken pics

from PIL import Image
import imghdr
import os 

for root,dir,file in os.walk(dir_path):
    for name in file:
        target = (os.path.join(root,name))
        result_type = imghdr.what(target)
        if result_type == None:
            print(target)
            os.remove(target)
'''



