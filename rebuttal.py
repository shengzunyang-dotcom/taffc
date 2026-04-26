import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['end to end ', 'bank', 'encoder']
acc = [91.31, 90.70, 91.62]  # 这些值是示例
f1 = [91.27, 90.69, 91.61]  # 这些值是示例

# 横坐标位置
x = np.arange(len(methods))

# 宽度设置
width = 0.35

# 绘制柱状图
fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, acc, width, edgecolor='black',label='Acc', color='#1f77b4')
rects2 = ax.bar(x + width/2, f1, width, edgecolor='black',label='F1', color='#ff7f0e')

# 添加一些文本标签
# ax.set_xlabel('Methods')
# ax.set_ylabel('Scores')
# ax.set_title('Comparison of Acc and F1 for Different Methods')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(88, 93)
# ax.legend()

# 显示数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()
plt.savefig('./acc7/pool.png', dpi=300)
