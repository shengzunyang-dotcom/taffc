import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成6174个样本，均值为0.17，值在0.02到0.3之间
# 计算合适的标准差，使数据大致在指定范围内
mean = 0.035  # 用户指定的均值
std =  0.008 # 标准差设定使得大部分值落在范围内
sample_size = 6174  # 样本数量
# (0.3 - 0.02) / 6
# 生成正态分布数据
samples = np.random.normal(mean, std, sample_size)

# 确保所有值都在0.02到0.3之间
samples = np.clip(samples, 0.001, 0.1)

# 创建区间，间隔为0.02
bins = np.arange(0, 0.1, 0.004)  # 0.32是为了包含0.3

# 计算每个区间的样本数
hist, bin_edges = np.histogram(samples, bins=bins)

# 创建柱状图
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(hist)), hist, width=0.8, alpha=0.7, color='#1f77b4', edgecolor='black')

# 设置横坐标标签
bin_labels = [f'{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}' for i in range(len(bin_edges)-1)]
plt.xticks(range(len(hist)), bin_labels, rotation=45)

# 添加标题和标签
# plt.title(f'样本分布柱状图 ({sample_size}个样本, 均值={mean}, 区间间隔0.02)', fontsize=14)
# plt.xlabel('样本值区间', fontsize=12)
# plt.ylabel('样本数量', fontsize=12)

# 在柱子上方显示数量
for i, (bar, count) in enumerate(zip(bars, hist)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/3., height + 10,
             f'{count}', ha='center', va='bottom', fontsize=9)

# 添加统计信息文本框
mu = samples.mean()
median = np.median(samples)
sigma = samples.std()
textstr = '\n'.join((
    f'目标均值: {mean:.2f}',
    f'实际均值: {mu:.2f}',
    f'中位数: {median:.2f}',
    f'标准差: {sigma:.2f}',
    f'样本数: {sample_size}'))

# 文本框属性
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# # 在图表左上角添加统计信息
# plt.text(0.05, 0.95, transform=plt.gca().transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)

# 添加网格线使读数更容易
plt.grid(True, axis='y', alpha=0.3)

# 调整布局防止标签被截断
plt.tight_layout()

# 显示图形
plt.show()
plt.savefig('./acc7/same.png', dpi=300)
