from sklearn.feature_extraction.text import CountVectorizer


if __name__ == '__main__':
    data = [
        'Everyone is fighting their own battle.',
        'To be free from their past.',
        'To live in their present.',
        'And to create their future.'
    ]

    # 示例化一个转化器类
    transfer = CountVectorizer()
    # 调用 transfer.fit_transform
    data_new = transfer.fit_transform(data)
    print('返回结果为：', data_new.toarray())
    # 输出特征名字
    print('特征名字为：', transfer.get_feature_names())

# 输出结果为
'''
返回结果为： [[0 1 0 0 1 1 0 0 0 0 1 0 1 0 0 1 0]
 [0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 1 1]
 [0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 1 1]
 [1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1 1]]
特征名字为： ['and', 'battle', 'be', 'create', 'everyone', 'fighting', 'free', 'from', 'future', 'in', 'is', 'live', 'own', 'past', 'present', 'their', 'to']
'''