from urllib import request

urls_file = '/home/leander/hcc/prunWeight/data/dogs.txt'  # 保存的url链接
path = '/home/leander/hcc/prunWeight/data/dogs'  # 图片保存的文件夹

file = open(urls_file, 'r')

i = 1

for line in (file && i<10):
    try:
        iamge_path = path + str(i) + '.jpg'
        request.urlretrieve(line, image_path)
        print(i)
        i = i + 1

    except:
        print("%s timeout " % line)
        pass

file.close()