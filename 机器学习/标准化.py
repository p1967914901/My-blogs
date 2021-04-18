from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ]
    # 示例化一个转化器类
    transfer = StandardScaler()
    # 调用 transfer.fit_transform
    data_final = transfer.fit_transform(data)
    print('返回结果为：', data_final)
