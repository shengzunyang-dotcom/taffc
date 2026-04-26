import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 数据准备
prompt_lengths = [64,80,96,112,128,144,160,176,192]
improved_acc = [88.87,89.63,90.24,89.79,91.31,90.24,89.33,89.94,89.79]
param_utilization = [a / b for a, b in zip(improved_acc, prompt_lengths)]

# 创建画布

fig, ax1 = plt.subplots(figsize=(10,6))
# plt.title("Improved Accuracy & Parameter Utilization vs Prompt Length", pad=20)
##87CEFA
#width=12, alpha=0.8, color="#87CEFA", edgecolor='#1f77b4',linewidth=3,
# 柱状图设置（左轴）
bars = ax1.bar(prompt_lengths, improved_acc, 
               width=12, alpha=0.8, color="#1f77b4", edgecolor='#000000',linewidth=3,
               label="Acc2")
ax1.set_xlabel("Hierarchical Prompt Analysis ", labelpad=10,fontsize=16)
ax1.set_ylabel("Acc2 ", 
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
                color="#ff7f0e", marker='o',
                linewidth=6, markersize=16,
                label="Unit length contribution rate")
ax2.set_ylabel("Unit length contribution rate", 
               labelpad=10,fontsize=16)
ax2.set_ylim(0, 1.6)  # 留出顶部空白

# 双坐标轴刻度对齐（关键技巧）
ax1.set_yticks(range(85, 93, 1))  # 0-12，步长2
ax2.set_yticks([x/10 for x in range(0, 16, 2)])  # 0-1.0，步长0.2

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

plt.tight_layout()
plt.show()
plt.savefig('prompt.png', dpi=300)