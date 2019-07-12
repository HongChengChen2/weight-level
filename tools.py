
dir_path = ('../rabbits')
#delete some pic
import os
import random
import shutil
files = os.listdir(dir_path)
sample = random.sample(files, 349)
for each in sample:
    file_path = os.path.join(dir_path, each)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)

    except PermissionError as e:
        pass

#ramdom index
'''
import random
import os
Imgnum = 649
files = os.listdir(dir_path) 
i = 0
L=random.sample(range(0,Imgnum),Imgnum)
filetype = ".jpg"
for filename in files:    
    portion = os.path.splitext(filename)
    if portion[1] == filetype:
        newname = '0' + str(L[i]) + filetype
        os.rename(filename, newname)
        i = i+1
'''