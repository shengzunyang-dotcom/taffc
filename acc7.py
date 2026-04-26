import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse, Polygon

# x = np.arange(1, 5)
# y1 = np.arange(1, 5)
# y2 = np.ones(y1.shape) * 4

# fig = plt.figure()
# # axs = fig.subplot_mosaic([['bar1', 'patches'], ['bar2', 'patches']])
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# # axs['bar1'].bar(x, y1, edgecolor='black', hatch="/")
# # axs['bar1'].bar(x, y2, bottom=y1, edgecolor='black', hatch='//')

# # axs['bar2'].bar(x, y1, edgecolor='black', hatch=['--', '+', 'x', '\\'])
# # axs['bar2'].bar(x, y2, bottom=y1, edgecolor='black',
# #                 hatch=['*', 'o', 'O', '.'])
# axs[0].bar(x, y1, edgecolor='black', hatch=['--', '+', 'x', '\\'])
# axs[0].set_title('Left Bars (Original Bottom Layer)')

# # 右子图：绘制原顶层数据（y2）
# axs[1].bar(x, y2, edgecolor='black', hatch=['*', 'o', 'O', '.'])
# axs[1].set_title('Right Bars (Original Top Layer)')

# x = np.arange(0, 40, 0.2)
# axs['patches'].fill_between(x, np.sin(x) * 4 + 30, y2=0,
#                             hatch='///', zorder=2, fc='c')
# axs['patches'].add_patch(Ellipse((4, 50), 10, 10, fill=True,
#                                  hatch='*', facecolor='y'))
# axs['patches'].add_patch(Polygon([(10, 20), (30, 50), (50, 10)],
#                                  hatch='\\/...', facecolor='g'))
# axs['patches'].set_xlim([0, 40])
# axs['patches'].set_ylim([10, 60])
# axs['patches'].set_aspect(1)
# categories = ['Chat-gpt', 'GPT-4V', 'GPT-4o', 'LLaMA-3.2','Qwen2.5','DeepSeek-VL2','TC^2RAHP']  # 横坐标类别
# data1 = [89.60, 90.91, 89.39,85.32,89.14,86.09,91.31]           # 第一组数据
# data2 = [44.44, 61.19, 53.73, 38.16,42.11,36.84,48.71]           # 第二组数据

# data1 = [84.97, 87.10, 86.38,78.80,85.56,84.02,87.58]           # 第一组数据
# data2 = [	40.77, 49.44, 45.25, 34.39,45.96,40.75,55.4]           # 第二组数据

categories1 = ['Acc_2','F1_weight',  'F1_binary','F1_micro','F1_macro' ]  # 横坐标类别
categories2 = [ 'Acc_2','F1_weight',  'F1_binary','F1_micro','F1_macro' ]  # 横坐标类别
data1 = [78.81,78.89, 75.83,78.81,78.48]           # 第一组数据
data2 = [91.31,91.27,90.17,91.31,90.92]           # 第二组数据

data3 = [ 80.76,80.63,84.95,80.76,79.16 ]           # 第一组数据
data4 = [ 87.58,87.39,  87.01,87.58, 87.17]           # 第二组数据

fig, axs = plt.subplots(1, 2, figsize=(10, 5), 
                        gridspec_kw={'width_ratios': [1, 1]})
# 2. 计算位置与宽度
x = np.arange(len(categories1))      # 生成基础横坐标 [0,1,2,3]
bar_width = 0.35                    # 单组柱宽（小于0.5）

# 3. 绘制并列柱状图
# plt.bar(x - bar_width/2, data1, width=bar_width, 
#         color='#1f77b4', edgecolor='black', label='组1',hatch=['//', '.', 'x', '\\','*', 'o', 'O'])
# plt.bar(x + bar_width/2, data2, width=bar_width, 
#         color='#ff7f0e', edgecolor='black', label='组2',hatch=['////', '..', 'xx', '\\\\','**', 'oo', 'OO'])

bars1a = axs[0].bar(x - bar_width/2, data1, width=bar_width, 
        color='#1f77b4', edgecolor='black', label='组1')
bars1b = axs[0].bar(x + bar_width/2, data2, width=bar_width, 
        color='#ff7f0e', edgecolor='black', label='组2')

# 4. 设置坐标轴标签
axs[0].set_xticks(x)
axs[0].set_xticklabels(categories1)
axs[0].set_ylim(60, 95)  # 优化Y轴范围
axs[0].grid(axis='y',  visible=False)

bars2a = axs[1].bar(x - bar_width/2, data3, width=bar_width, 
        color='#1f77b4', edgecolor='black', label='组1')
bars2b = axs[1].bar(x + bar_width/2, data4, width=bar_width, 
        color='#ff7f0e', edgecolor='black', label='组2')


# 4. 设置坐标轴标签
axs[1].set_xticks(x)
axs[1].set_xticklabels(categories2)
axs[1].set_ylim(70, 90)  # 优化Y轴范围
axs[1].grid(axis='y',  visible=False)
# plt.xlabel('类别')
# plt.ylabel('数值')
# plt.title('并列柱状图示例')

# # 5. 添加数据标签（精确值）
# for i in range(len(categories)):
#     # 第一组数据标签
#     plt.text(x[i] - bar_width/2, data1[i] + 0.5, str(data1[i]), 
#              ha='center', va='bottom', fontsize=9)
#     # 第二组数据标签
#     plt.text(x[i] + bar_width/2, data2[i] + 0.5, str(data2[i]), 
#              ha='center', va='bottom', fontsize=9)

# 6. 添加图例与显示
plt.tight_layout()  # 自动调整间距
plt.show()
plt.savefig('./acc7/knn.png', dpi=300)
