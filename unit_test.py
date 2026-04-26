import numpy as np


a = 'multimodal'
b = 'multimodal-1'

if a in b:
    print('xxx')

exit(0)
Mat1 = np.random.random((3, 4))
Mat2 = np.random.random((3, 4))

print(Mat1.shape, Mat2.shape)
correlation = np.corrcoef(Mat1, Mat2, rowvar=False)
print("矩阵1=\n", Mat1)
print("矩阵2=\n", Mat2)
print("相关系数矩阵=\n", correlation)
print(correlation.shape)


print(50 * '*')
import numpy as np
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
Array1 = [[1, 2, 3], [4, 5, 6]]
Array2 = [[11, 25, 346], [734, 48, 49]]
Mat1 = np.array(Array1)
Mat2 = np.array(Array2)
correlation = np.corrcoef(Mat1, Mat2, rowvar=False)
print("矩阵1=\n", Mat1)
print("矩阵2=\n", Mat2)
print("相关系数矩阵=\n", correlation)

print(correlation.shape)




fig = sns.heatmap(correlation)
scatter_fig = fig.get_figure()
scatter_fig.savefig('./test.png', dpi = 400)