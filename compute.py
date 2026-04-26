from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# fig = sns.heatmap(correlation)
# scatter_fig = fig.get_figure()
# scatter_fig.savefig('./test.png', dpi = 400)


def plot(x, y1, y2, i, types = 'cc'):
    plt.clf()
    plt.scatter(x, y1, color='red', label='condition')
    # 画第二种散点图
    plt.scatter(x, y2, color='blue', label='=non-condition')
    # 添加标题和标签
    plt.title('Scatter Plot Example')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 添加图例
    plt.legend()
    # 显示图形
    plt.savefig(f'./images/image_{types}_{i}.png')


# 0, 5, 10, 15, 20, 25

for i in [49]:
    text = np.load(f'./cc/texts_{i}.npy')
    visual = np.load(f'./cc/visuals_{i}.npy')


    text1 = np.load(f'./cc1/texts_{i}.npy')
    visual1 = np.load(f'./cc1/visuals_{i}.npy')
    
    # index = 10
    # x1 = text[10]
    # y1 = visual[10]

    # x2 = text1[10]
    # y2 = visual1[10]
    x1 = text
    y1 = visual

    x2 = text1
    y2 = visual1

    # cov1 = np.cov(x1, y1, rowvar=False)
    # # cov2 = np.cov(x2, y2, rowvar=False)
    # print(cov1.shape, cov2.shape)
    corr_cond = np.corrcoef(x1, y1, rowvar=False)
    corr_non_cond = np.corrcoef(x2, y2, rowvar=False)
    print(corr_cond.shape, corr_non_cond.shape)
    corr_cond = corr_cond[768:, 0:768]
    corr_non_cond = corr_non_cond[768:, 0:768]
    print(corr_cond.shape, corr_non_cond.shape)

    fig = sns.heatmap(corr_non_cond)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig('./test2.png', dpi = 400)
    exit(0)



    bsz, dim = text.shape

    px = []
    py1 = []
    py2 = []
    
    py_mi_1 = []
    py_mi_2 = []
    
    for j in range(bsz):
        x = text[j]
        y = visual[j]
        rho = np.corrcoef(x, y)
        mi_1 = metrics.mutual_info_score(x, y)

        x1 = text1[j]
        y1 = visual1[j]
        rho1 = np.corrcoef(x1, y1)
        mi_2 = metrics.mutual_info_score(x1, y1)

        px.append(j)
        py1.append(rho[0][1])
        py2.append(rho1[0][1])

        py_mi_1.append(mi_1)
        py_mi_2.append(mi_2)
        print(mi_1, mi_2, rho[0][1])
    plot(px, py1, py2, i)
    plot(px, py_mi_1, py_mi_2, i, types='mi')