from  sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def linear1(x, y):
    """
    正规方程的优化方法对波士顿房价进行预测
    :return:
    """
    # 获取数据
    # boston = load_boston()

    # 划分数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 标准化
    # transfer = StandardScaler()
    # x_train = transfer.fit_transform(x_train)
    # x_test = transfer.fit_transform([[2070]])  # 控制变量, 用同样的参数进行标准化

    # 预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 得出模型
    # print('正规方程——权重系数为：\n', estimator.coef_)
    # print('正规方程——偏重为：\n', estimator.intercept_)

    # 模型评估
    y_predict = estimator.predict([[2070]])
    print('实际：\n', y_test)
    print('预测：\n', y_predict)
    # error = mean_squared_error(y_test, y_predict)
    # print('正规方程——均方误差为：\n', error)


if __name__ == '__main__':
    y = [21.26292038, 22.10585022, 22.88397026, 23.66081047,
            24.21640015,
            24.79043007,
            25.42106056,
            26.11203003,
            26.58672905,
            27.56347084,
            28.34149933,
            29.4718399,
            30.30003929,
            31.0941391,
            31.8190403,
            38.43338013,
            40.22895813,
            41.52280045,
            42.5217 ]
    
    print(len(y))
    print([i for i in range(2020-len(y), 2020)])
    # x = [[i] for i in range(1959, 2020)]
    # print(len(y), len(x))
    # linear1(x, y)