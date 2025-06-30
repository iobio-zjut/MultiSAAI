import chardet

with open('/home/data/user/lvzexin/zexinl/abag_data/preprocess/cov.csv', 'rb') as f:
    print(f.readline())
    print(f.readline())
    result = chardet.detect(f.read())  # 读取一定量的数据进行编码检测

print(result['encoding'])  # 打印检测到的编码
