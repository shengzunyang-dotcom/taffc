
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

plt.style.use('fivethirtyeight')

x = np.linspace(0, 50, 50)  # 关键修改点[8](@ref)

# 数据放大10倍
y1 = [val*10 for val in [0.299, 0.289, 0.284, 0.290, 0.283, 0.292, 0.287, 0.284, 0.280, 0.281,
    0.276, 0.265, 0.263, 0.268, 0.270, 0.263, 0.260, 0.254, 0.256, 0.253,
    0.247, 0.245, 0.234, 0.231, 0.227, 0.224, 0.219, 0.215, 0.210, 0.216,
    0.210, 0.201, 0.194, 0.190, 0.184, 0.179, 0.174, 0.180, 0.176, 0.172,
    0.160, 0.154, 0.158, 0.150, 0.143, 0.140, 0.142, 0.137, 0.132, 0.130]]

y2 = [val*10 for val in [0.059, 0.062, 0.060, 0.058, 0.062, 0.054, 0.056, 0.053, 0.050, 0.048,
    0.049, 0.045, 0.050, 0.046, 0.042, 0.043, 0.041, 0.040, 0.037, 0.035,
    0.036, 0.034, 0.032, 0.037, 0.030, 0.025, 0.034, 0.032, 0.042, 0.031,
    0.028, 0.026, 0.025, 0.023, 0.020, 0.024, 0.021, 0.020, 0.024, 0.018,
    0.014, 0.020, 0.021, 0.019, 0.017, 0.014, 0.016, 0.013, 0.010, 0.017]]

y3 = [val*10 for val in [0.064, 0.060, 0.063, 0.057, 0.054, 0.056, 0.050, 0.046, 0.045, 0.043,
    0.048, 0.046, 0.042, 0.047, 0.040, 0.043, 0.039, 0.037, 0.036, 0.038,
    0.034, 0.036, 0.032, 0.030, 0.031, 0.027, 0.028, 0.024, 0.026, 0.024,
    0.025, 0.024, 0.020, 0.021, 0.023, 0.020, 0.014, 0.017, 0.020, 0.016,
    0.025, 0.020, 0.017, 0.014, 0.016, 0.010, 0.014, 0.017, 0.013, 0.015]]

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
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# 绘制曲线
ax.plot(x, y1, lw=2)
ax.plot(x, y2, lw=2)
ax.plot(x, y3, lw=2)


# 添加图例和标题
# ax.legend(loc='upper right', frameon=False)
# ax.set_title("Negative Samples", pad=20)

plt.savefig('prompt2.png',  dpi=300, bbox_inches='tight')  # 保存优化[1,8](@ref)
plt.show()