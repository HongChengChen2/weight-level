
dir_path = ('data/cats/')

import os
import random
import shutil
files = os.listdir(dir_path)
sample = random.sample(files, 755)
for each in sample:
    file_path = os.path.join(dir_path, each)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)

    except PermissionError as e:
        pass
