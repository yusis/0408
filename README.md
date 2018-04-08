# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# データの作成
x = np.arange(0, 6, 0.1) # 0から6まで0.1刻みで生成
y1 = np.sin(x)
y2 = np.cos(x)

# グラフの描画
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos")
plt.xlabel("x") # x軸のラベル
plt.ylabel("y") # y軸のラベル
plt.title('sin & cos')
plt.legend()
plt.show()


### matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.show()



# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# データの作成
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# グラフの描画
plt.plot(x, y)
plt.show()


Python数据分析基础教程NumPy学习指南第2版.pdf
https://github.com/shihyu/python_ebook/blob/master/NumPy/Python%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8BNumPy%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97%E7%AC%AC2%E7%89%88.pdf

NumPy Beginner's Guide(3rd).pdf
https://github.com/shihyu/python_ebook/blob/master/NumPy/NumPy%20Beginner's%20Guide(3rd).pdf

『ゼロから作る Deep Learning』のリポジトリ
https://github.com/oreilly-japan/deep-learning-from-scratch

很多書
https://github.com/shihyu/python_ebook


