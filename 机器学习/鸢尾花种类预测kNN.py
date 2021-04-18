from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    #  获取数据
    iris = load_iris()

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)     # 控制变量, 用同样的参数进行标准化

    # KNN 算法预估器
    estimator = KNeighborsClassifier(n_neighbors=10)
    estimator.fit(x_train, y_train)

    # 模型评估
    # 方法一：直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict：\n', y_predict)
    print('直接对比真实值和预测值:\n', y_test == y_predict)

    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率:\n', score)