from sklearn.datasets import fetch_20newsgroups

if __name__ == '__main__':
    # 获取数据
    news = fetch_20newsgroups('./data', subset='all')
