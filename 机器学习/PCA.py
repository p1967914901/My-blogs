from sklearn.decomposition import PCA

def pca(data):
    # 示例化一个转化器类
    transfer = PCA(n_components=2)
    # 调用 transfer.fit_transform
    data_final = transfer.fit_transform(data)
    print('返回结果为：', data_final)
    # 输出特征名字
    # print('特征名字为：', transfer.get_feature_names())


if __name__ == '__main__':
    data = [
        [2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1]
    ]
    pca(data)

    '''
    返回结果为： [[ 1.28620952e-15  3.82970843e+00]
     [ 5.74456265e+00 -1.91485422e+00]
     [-5.74456265e+00 -1.91485422e+00]]
    '''