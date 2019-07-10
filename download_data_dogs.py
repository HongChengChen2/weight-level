from urllib import request
import socket
import traceback
import time

urls_file = 'data/dogs_2.txt'  
path = 'data/dogs/'  
socket.setdefaulttimeout(20)

file = open(urls_file, 'r')

i = 338
k = 0

for line in file :
    try:
        image_path = path + str(i) + '.jpg'
        request.urlretrieve(line, image_path)
        print( "已下载图片：" , i)
        i = i + 1

    except Exception as e:
        print('***', type(e), e, '***'  )
        pass

    k = k + 1
    print( "共尝试地址：" , k)

file.close()