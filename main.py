from matplotlib import pyplot as plt
from tqdm import tqdm

from read_in import get_image_pixel
from tools import cs_auto_romp, generate_t_line, generate_random_signal, draw_single_signal, draw_double_signal, \
    load_out_gray_image, simple_load_out_gray_image, count_mse
import numpy as np
import romp


def generate_new_dictionary(original_signal, gate=0.9, steps=30):
    length = original_signal.shape[1]
    dictionary = np.column_stack((np.zeros(length), np.zeros(length)))

    for i in tqdm(range(steps)):
        all_residual = np.zeros(length)
        mse = 0
        for signal in original_signal:
            support_all, residual = romp.romp(signal, dictionary)
            all_residual += residual
            mse += count_mse(signal, signal-residual)
        print('Average MSE =', mse/original_signal.shape[0])
        max_residual = np.max(abs(all_residual))
        gate_residual = gate * max_residual
        new_dictionary_element = np.where(abs(all_residual) > gate_residual, 1, 0)
        dictionary = np.column_stack((dictionary, new_dictionary_element))
        simple_load_out_gray_image('Output/' + str(i) + '.jpg', dictionary.T[i+2], (28, 28))
    return dictionary


path = 'train.csv'
labels, images = get_image_pixel(path, 0, 10)
"""
1.start务必小于stop,因为这个读取是文件一行一行的读取的，真做逆序还是先输出再做
2.文件前一行是非数据，如果不是1，需要传入drop参数矫正
3.文件的数据如果是浮点的话，会出问题，现在先不改这个
"""
plt.imshow(images[0].reshape(28, 28), cmap='gray')
plt.show()

generate_new_dictionary(images, gate=0.8, steps=30)
