# x = [0.0002,0.0004,0.0006,0.0008,0.001,0.0012,0.0014,0.0016,0.0018]
# y1 = [87.35,89.79,90.70,91.01,91.31,89.93,90.09,88.41,86.74]

# y2 = [1.8072,1.3714,0.9356,0.4998,0.064,0.3718,0.8076,1.2434,1.6014]

import matplotlib.pyplot as plt
import numpy as np

# 示例数据
prompt_lengths = np.array([val*10000 for val in [0.0002,0.0004,0.0006,0.0008,0.001,0.0012,0.0014,0.0016,0.0018]])
improved_acc = np.array([89.79,90.40,90.70,91.01,91.31,90.24,91.16,90.09,89.93])
param_utilization = np.array([1.8072,1.1041,0.6853,0.3913,0.1751,0.3402,0.6047,1.0854,1.6014])

# 创建画布

fig, ax1 = plt.subplots(figsize=(10,6))
# plt.title("Improved Accuracy & Parameter Utilization vs Prompt Length", pad=20)
#width=1, alpha=0.8, color="#87CEFA", edgecolor='#1f77b4',linewidth=3,
# 柱状图设置（左轴）
bars = ax1.bar(prompt_lengths, improved_acc, 
               width=1.5, alpha=0.8, color="#299D8F", edgecolor='#000000',linewidth=3,
               label="Acc2")
ax1.set_xlabel("Hierarchical Prompt Analysis ", labelpad=10,fontsize=16)
ax1.set_ylabel("Acc2", 
              labelpad=10,fontsize=16)
ax1.set_ylim(85,93)  # 根据截图刻度设置
ax1.set_xticks(prompt_lengths)
# ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.grid(False,axis='y')
ax1.tick_params(
    axis='both', 
    which='major',
    labelsize=14,  # 主刻度字号[3,5](@ref)
    length=6,       # 刻度线长度
    width=2         # 刻度线宽度
)

# 折线图设置（右轴）
ax2 = ax1.twinx()
ax2.tick_params(
    axis='both',
    labelsize=14,  # 与主坐标轴保持字号一致[3](@ref)
    length=4,      # 副轴刻度稍短
    width=1.5
)
#color="#ff7f0e", marker='o', 
line, = ax2.plot(prompt_lengths, param_utilization, 
                color="#E53528", marker='o', 
                linewidth=6, markersize=16,
                label="Unit length ontribution rate")
ax2.set_ylabel("Unit length contribution rate", 
               labelpad=10,fontsize=16)
ax2.set_ylim(0, 1.6)  # 留出顶部空白

# 双坐标轴刻度对齐（关键技巧）
ax1.set_yticks(range(85, 93, 1))  # 0-12，步长2
ax2.set_yticks([x/10 for x in range(0, 32,4)])  # 0-1.0，步长0.2

# # 组合图例
# lines = [bars, line]
# labels = [l.get_label() for l in lines]
# # ax1.legend(lines, labels, loc="upper left", 
# #           bbox_to_anchor=(0.12, 0.88))
# ax1.legend(lines, labels, loc="upper center",fontsize=16)

lines = [bars, line]
labels = [l.get_label() for l in lines]
legend = ax1.legend(lines, labels, 
                   loc='lower center',   # 基准定位点
                   bbox_to_anchor=(0.5, 1.03),  # 坐标(水平居中, 超出顶部10%)
                   ncol=2,              # 双列布局
                   frameon=False,        # 显示边框
                   edgecolor='black',   # 边框颜色
                   facecolor='white',   # 背景色
                   fontsize=16)         # 字体大小


plt.savefig('alpha.png', dpi=300, bbox_inches='tight')  # 保存优化[1,8](@ref)
plt.show()