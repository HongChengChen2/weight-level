from PIL import Image
import os

path = ('data/rabbits/')
badFilesList = []
for root, dirs, files in os.walk(path):
    for each in files:
        try:
            im = Image.open(os.path.join(root, each))
            # im.show()
        except Exception as e:
            print('Bad file:', os.path.join(root, each))
            badFilesList.append(os.path.join(root, each))

if len(badFilesList) != 0:
    for each in badFilesList:
        try:
            os.remove(each)
        except Exception as e:
            print('Del file: %s failed, %s' % (each, e))
