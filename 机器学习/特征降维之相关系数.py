from scipy.stats import pearsonr
import numpy as np


if __name__ == '__main__':
    data = [
        [1, 2, 3, 4, 5],
        [1, 7, 8, 14, 10],
        [1, 12, 13, 24, 15]
    ]
    m = np.array(data)
    r = pearsonr(m[:, 1], m[:, 3])
    print(r)    # 第一个值即为它们之间的相关系数
'''
(0.9999999999999998, 1.3415758552508151e-08)
'''