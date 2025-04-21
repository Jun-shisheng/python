import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# 参数设置
n = 5000
p = 0.0015
k_max = 10

# 计算 P(X <= 10)
prob_cdf = binom.cdf(k_max, n, p)
print(f"P(X <= 10) = {prob_cdf:.4f}")

# 计算并输出每个 k 值的概率 P(X = k)
k_values = np.arange(0, k_max + 1)
prob_pmf = binom.pmf(k_values, n, p)

for k, prob in zip(k_values, prob_pmf):
    print(f"P(X = {k}) = {prob:.6f}")

# 计算 x 从 1 到 10 的和（即 P(X=1) + P(X=2) + ... + P(X=10)）
sum_prob = np.sum(prob_pmf[1:])  # 从 k=1 到 k=10 的概率和
print(f"Sum of P(X = k) for k from 1 to 10: {sum_prob:.6f}")

# 绘制二项分布的概率质量函数（PMF）图
plt.figure(figsize=(10, 6))
plt.bar(k_values, prob_pmf, color='blue', alpha=0.7, label=f'Binomial PMF (n={n}, p={p})')
plt.title('Binomial Distribution PMF')
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability P(X = k)')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()