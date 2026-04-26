import numpy as np
from scipy.stats import ttest_rel
import pickle as pkl

# 假设你有两组模型预测值（numpy数组或列表）
# 例如：
# preds_modelA = np.loadtxt('modelA_preds.txt')
# preds_modelB = np.loadtxt('modelB_preds.txt')

with open("/data/yangshengzun/knowledge-injection/checkpoint/mosei_my_list.pkl", "rb") as handle:
        our_mosi = pkl.load(handle)
with open("/data/yangshengzun/knowledge-injection/checkpoint/gemma-3-12b-it_moseimy_list.pkl", "rb") as handle:
        qwen_mosi = pkl.load(handle)

preds_model_our = our_mosi['pred']
preds_model_qwen = qwen_mosi['pred']

# 转换为 numpy 数组
preds_modelA = np.array(preds_model_our)
preds_modelB = np.array(preds_model_qwen)
# preds_modelA = preds_modelA[:preds_modelB.shape[0]]
print(preds_modelA.shape)
print(preds_modelB.shape)

# 配对 t 检验
t_stat, p_value = ttest_rel(preds_modelA, preds_modelB)

print("Paired t-test results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4e}")

# 判断显著性水平
alpha = 0.05
if p_value < alpha:
    print("→ The difference between the two models is statistically significant (p < 0.05).")
else:
    print("→ No significant difference (p ≥ 0.05).")