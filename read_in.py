import numpy as np
import matplotlib.pyplot as plt


def get_image_pixel(path, start, stop, step=1, drop=1):
    """
    path:文件路径,字符串类型
    start:起始图片位置，0是第一张
    stop:最后一张,不包含在内
    step:步幅，可能需要用，还是提供了
    drop:文件的前多少行不是数据
    """
    with open(path) as f:
        for _ in range(drop + start):
            f.readline()
        res = []
        for _ in range(start, stop, step):
            dat_str = f.readline().split(',')
            dat_int = [int(i) for i in dat_str]
            res.append(np.array(dat_int))
    label_list = []
    image_list = []
    for l in res:
        label_list.append(l[0])
        image_list.append(l[1:])
    return np.array(label_list), np.array(image_list)



