# coding=utf-8

"""
使用numpy实现卷积神经网络
使用了python3
"""

import numpy
import sys


def conv_(img, conv_filter):
    """计算单个滤波器与图像卷积
    """
    filter_size = conv_filter.shape[1]
    result = numpy.zeros((img.shape))
    # 循环计算每个区域的卷积结果
    for r in numpy.uint16(numpy.arange(filter_size / 2.0,
                                       img.shape[0] - filter_size / 2.0 + 1)):
        for c in numpy.uint16(numpy.arange(filter_size / 2.0,
                                           img.shape[1] - filter_size / 2.0 + 1)):
            curr_region = img[r - numpy.uint16(numpy.floor(filter_size / 2.0)):r + numpy.uint16(
                numpy.ceil(filter_size / 2.0)),
                          c - numpy.uint16(numpy.floor(filter_size / 2.0)):c + numpy.uint16(
                              numpy.ceil(filter_size / 2.0))]
            # 在区域和滤波器之间应用逐元素乘法
            curr_result = curr_region * conv_filter
            # 将相乘的结果做和运算
            conv_sum = numpy.sum(curr_result)
            # 在对应位置保存结果
            result[r, c] = conv_sum
    # 从result矩阵中截出最后的结果
    final_result = result[numpy.uint16(filter_size / 2.0):result.shape[0] - numpy.uint16(filter_size / 2.0),
                   numpy.uint16(filter_size / 2.0):result.shape[1] - numpy.uint16(filter_size / 2.0)]
    return final_result


def conv(img, conv_filter):
    # 检查滤波器的深度和图像的通道数是否相等
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:  # 最后一个维度存放滤波器的深度
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    # 检查滤波器的维度数否相等
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    # 检查滤波器的维度是否为奇数
    if conv_filter.shape[1] % 2 == 0:
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # 创建一个空的特征图，用于存放卷积的结果
    feature_maps = numpy.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                                img.shape[1] - conv_filter.shape[1] + 1,
                                conv_filter.shape[0]))

    # 使用滤波器对图像进行卷积
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        # 获取当前滤波器
        curr_filter = conv_filter[filter_num, :]
        # 检测每组滤波器中是否有多个通道
        if len(curr_filter.shape) > 2:
            # 计算每组通道产生的特征图
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
            # 对剩余的做累加
            for ch_num in range(1, curr_filter.shape[-1]):
                conv_map = conv_map + conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])
        else:
            # 否则只是对图像做单通道卷积
            conv_map = conv_(img, curr_filter)
        # 保存当前滤波器组生成的特征图
        feature_maps[:, :, filter_num] = conv_map
    # 返回所有的特征图
    return feature_maps


def pooling(feature_map, size=2, stride=2):
    """池化操作
    """
    # 存放池化后的输出结果
    pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0] - size + 1) / stride + 1),
                            numpy.uint16((feature_map.shape[1] - size + 1) / stride + 1),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in numpy.arange(0, feature_map.shape[0] - size + 1, stride):
            c2 = 0
            for c in numpy.arange(0, feature_map.shape[1] - size + 1, stride):
                pool_out[r2, c2, map_num] = numpy.max([feature_map[r:r + size, c:c + size, map_num]])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out


def relu(feature_map):
    """激活函数
    """
    # 存放激活后的输出结果
    relu_out = numpy.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in numpy.arange(0, feature_map.shape[0]):
            for c in numpy.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
    return relu_out
