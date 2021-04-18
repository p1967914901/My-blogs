from sklearn.feature_selection import VarianceThreshold

if __name__ == '__main__':
    data = [
        [1, 2, 3, 4, 5],
        [1, 7, 8, 9, 10],
        [1, 12, 13, 14, 15]
    ]
    # 示例化一个转化器类
    transfer = VarianceThreshold()  # `threshold` 用默认值 0
    # 调用 transfer.fit_transform
    data_final = transfer.fit_transform(data)
    print('返回结果为：', data_final)

'''
返回结果为： [[ 2  3  4  5]
 [ 7  8  9 10]
 [12 13 14 15]]
'''