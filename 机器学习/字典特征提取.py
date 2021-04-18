from  sklearn.feature_extraction import DictVectorizer


if __name__ == '__main__':
    data = [
        {'city': '北京', 'value' : 150},
        {'city': '浙江', 'value': 10},
        {'city': '上海', 'value': 50}
    ]

    # 示例化一个转化器类
    transfer = DictVectorizer()
    # 调用 transfer.fit_transform
    data = transfer.fit_transform(data)
    print('返回结果为：', data)
    # 输出特征名字
    print('特征名字为：', transfer.get_feature_names())


# 输出结果为
'''
返回结果为：   (0, 1)	1.0
  (0, 3)	150.0
  (1, 2)	1.0
  (1, 3)	10.0
  (2, 0)	1.0
  (2, 3)	50.0
特征名字为： ['city=上海', 'city=北京', 'city=浙江', 'value']
'''