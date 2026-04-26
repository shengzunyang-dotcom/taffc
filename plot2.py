
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

plt.style.use('fivethirtyeight')

x = np.linspace(0, 50, 50)  # 关键修改点[8](@ref)

# 数据放大10倍
y1 = [val*10 for val in [0.301, 0.320, 0.311, 0.315, 0.324, 0.335, 0.330, 0.344, 0.340, 0.348,
    0.356, 0.350, 0.360, 0.371, 0.375, 0.370, 0.378, 0.382, 0.393, 0.401,
    0.411, 0.415, 0.410, 0.414, 0.427, 0.433, 0.425, 0.435, 0.446, 0.450,
    0.461, 0.465, 0.468, 0.478, 0.462, 0.480, 0.487, 0.495, 0.490, 0.487,
    0.496, 0.502, 0.506, 0.501, 0.512, 0.505, 0.510, 0.504, 0.509, 0.502]]

y2 = [val*10 for val in [0.069, 0.063, 0.075, 0.087, 0.088, 0.081, 0.080, 0.089, 0.092, 0.095,
    0.093, 0.098, 0.105, 0.107, 0.095, 0.108, 0.114, 0.117, 0.118, 0.125,
    0.143, 0.126, 0.133, 0.135, 0.139, 0.143, 0.150, 0.163, 0.160, 0.165,
    0.159, 0.160, 0.168, 0.170, 0.173, 0.170, 0.174, 0.168, 0.160, 0.164,
    0.168, 0.167, 0.169, 0.172, 0.163, 0.169, 0.170, 0.173, 0.174, 0.168]]

y3 = [val*10 for val in [0.051, 0.062, 0.057, 0.059, 0.064, 0.068, 0.075, 0.070, 0.076, 0.078,
    0.074, 0.076, 0.080, 0.085, 0.087, 0.094, 0.080, 0.088, 0.095, 0.101,
    0.106, 0.114, 0.110, 0.119, 0.112, 0.105, 0.117, 0.120, 0.130, 0.132,
    0.129, 0.135, 0.136, 0.138, 0.140, 0.147, 0.142, 0.145, 0.146, 0.139,
    0.135, 0.139, 0.141, 0.137, 0.135, 0.139, 0.134, 0.140, 0.142, 0.136]]

# 设置1:2.5高宽比（宽12.5英寸，高5英寸）
fig, ax = plt.subplots(figsize=(12.5, 5))  # 修改关键参数[1,6](@ref)

# 调整坐标轴范围适应放大后的数据
ax.set_ylim(min(y1+y2+y3)*0.9, max(y1+y2+y3)*1.1)  # 自动计算y轴范围

# 网格配置（保持可见性优化）
plt.minorticks_on()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))  # 精确控制次刻度[3](@ref)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))    # 增加y轴次刻度密度

ax.grid(True, which='major', 
        color='#888888', linestyle='--', linewidth=0.8, alpha=0.7)  # 加深主网格
ax.grid(True, which='minor', 
        color='#CCCCCC', linestyle=':', linewidth=0.5, alpha=0.5)   # 优化次网格

# 调整边距以适应新比例
plt.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.1)  # 优化布局参数[8](@ref)

# 绘制曲线
ax.plot(x, y1, lw=2)
ax.plot(x, y2, lw=2)
ax.plot(x, y3, lw=2)

# 添加图例和标题
# ax.legend(loc='upper right', frameon=False)
# ax.set_title("Positive Sample", pad=20)

# ax.set_yticks([])                    # 隐藏y轴刻度标签 [1,4](@ref)
# ax.spines['left'].set_visible(False) # 隐藏左侧轴线 [4](@ref)

# # === 显式设置网格 ===
# ax.grid(True, axis='x',              # 强制显示x轴主网格线
#         which='both',                # 同时显示主次网格线 [7](@ref)
#         linestyle='--',              # 虚线样式 [6,7](@ref)
#         alpha=0.5)                   # 透明度设置 [6](@ref)


plt.savefig('prompt1.png', dpi=300, bbox_inches='tight')  # 保存优化[1,8](@ref)
plt.show()