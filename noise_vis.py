import os
import numpy as np
import matplotlib.pyplot as plt

# 模型列表和列名称
model_list = [
    'gsnet.clear', 'gsnet.0.002', 'gsnet.0.005', 'gsnet.0.01', 
    'ignet_v0.8.2.clear', 'ignet_v0.8.2.0.002', 'ignet_v0.8.2.0.005', 'ignet_v0.8.2.0.01'
]
camera_type = 'realsense'

# 噪声级别和数据存储
noise_levels = [0, 0.002, 0.005, 0.01]
gsnet_ap_mean = []
ignet_ap_mean = []

# 遍历模型，计算AP_mean
for model in model_list:
    root = os.path.join('experiment', model)
    split_ap = []
    for split in ['seen', 'similar', 'novel']:
        res = np.load(os.path.join(root, f'ap_test_{split}_{camera_type}.npy'))
        ap_top50 = np.mean(res[:, :, :50, :])
        split_ap.append(ap_top50)
    # 计算AP_mean
    ap_mean = np.mean(split_ap) * 100.0
    
    if 'gsnet' in model:
        gsnet_ap_mean.append(ap_mean)
    elif 'ignet_v0.8.2' in model:
        ignet_ap_mean.append(ap_mean)

# # 绘图
# plt.figure(figsize=(8, 6))
# plt.plot(noise_levels, gsnet_ap_mean, label='GSNet', marker='o')
# plt.plot(noise_levels, ignet_ap_mean, label='MMGNet', marker='s')

# plt.title('AP_mean vs Gaussian Noise Level', fontsize=14)
# plt.xlabel('Gaussian Noise Level', fontsize=12)
# plt.ylabel('AP_mean', fontsize=12)
# plt.xticks(noise_levels)
# plt.legend()
# plt.grid(True)
# plt.savefig('ap_mean_vs_noise_level.png')
# # plt.show()

differences = [g - i for g, i in zip(ignet_ap_mean, gsnet_ap_mean)]

# 绘图
plt.figure(figsize=(10, 7))
plt.plot(noise_levels, gsnet_ap_mean, label='GSNet', marker='o')
plt.plot(noise_levels, ignet_ap_mean, label='MMGNet', marker='s')
# plt.plot(noise_levels, differences, label='Difference (MMGNet - GSNet)', linestyle='--', marker='d')

# 为每个点标注差值
for x, diff, gs, ig in zip(noise_levels, differences, gsnet_ap_mean, ignet_ap_mean):
    plt.annotate(f'{diff:.1f}', (x, (gs + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

# 图例和标题
plt.title('$\mathbf{AP}_{mean}$ vs Gaussian Noise Level', fontsize=14)
plt.xlabel('Gaussian Noise Level', fontsize=12)
plt.ylabel('$\mathbf{AP}_{mean}$', fontsize=12)
plt.xticks(noise_levels)
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('ap_mean_vs_noise_level.png')