import numpy as np
import os
import matplotlib.pyplot as plt

data_root = 'depth_distance'

# 定义噪声级别和路径
# gaussian_noise_levels = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019, 0.02]
gaussian_noise_levels = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015]
smooth_noise_levels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
# smooth_levels = [0, 1, 3, 5]
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 用于存储不同噪声级别的深度距离数据
gaussian_depth_distances = []
smooth_depth_distances = []
dropout_depth_distances = []

# 加载 Gaussian Noise 数据
for gaussian_noise_level in gaussian_noise_levels:
    depth_dis_path = os.path.join(data_root, f'g{gaussian_noise_level}s0_depth_distance.npy')
    clear_depth_dis = np.load(depth_dis_path)  # (scene_list, 256, 1)
    clear_depth_dis = np.mean(clear_depth_dis, axis=1)  # (scene_list, 1)
    gaussian_depth_distances.append(np.mean(clear_depth_dis))
    
# 加载 Smooth Noise 数据
# for smooth_noise_level in smooth_noise_levels:
#     depth_dis_path = os.path.join(data_root, 'bs', f'g0.0bs{smooth_noise_level}_depth_distance.npy')
#     clear_depth_dis = np.load(depth_dis_path)  # (scene_list, 256, 1)
#     clear_depth_dis = np.mean(clear_depth_dis, axis=1)  # (scene_list, 1)
#     smooth_depth_distances.append(np.mean(clear_depth_dis))

for smooth_noise_level in smooth_noise_levels:
    depth_dis_path = os.path.join(data_root, f'g0.0s{smooth_noise_level}_depth_distance.npy')
    clear_depth_dis = np.load(depth_dis_path)  # (scene_list, 256, 1)
    clear_depth_dis = np.mean(clear_depth_dis, axis=1)  # (scene_list, 1)
    smooth_depth_distances.append(np.mean(clear_depth_dis))

# for smooth_noise_level in smooth_levels:
#     depth_dis_path = os.path.join(data_root, f'smooth_{smooth_noise_level}_depth_distance.npy')
#     clear_depth_dis = np.load(depth_dis_path)  # (scene_list, 256, 1)
#     clear_depth_dis = np.mean(clear_depth_dis, axis=1)  # (scene_list, 1)
#     smooth_depth_distances.append(np.mean(clear_depth_dis))
    
# 加载 Dropout Rate 数据
for dropout_rate in dropout_rates:
    depth_dis_path = os.path.join(data_root, f'dropout_rate_{dropout_rate}_depth_distance.npy')
    clear_depth_dis = np.load(depth_dis_path)  # (scene_list, 256, 1)
    clear_depth_dis = np.mean(clear_depth_dis, axis=1)  # (scene_list, 1)
    dropout_depth_distances.append(np.mean(clear_depth_dis))

# 绘图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Gaussian Noise 图
axs[0].plot(gaussian_noise_levels, gaussian_depth_distances, marker='o', color='b')
axs[0].set_title('Gaussian Noise')
axs[0].set_xlabel('Level')
axs[0].set_ylabel('Chamfer Distance')
axs[0].grid(True)

# Smooth Noise 图
axs[1].plot(smooth_noise_levels, smooth_depth_distances, marker='o', color='r')
axs[1].set_title('Smooth Noise')
axs[1].set_xlabel('Level')
axs[1].set_ylabel('Chamfer Distance')
axs[1].grid(True)

# Dropout Rate 图
axs[2].plot(dropout_rates, dropout_depth_distances, marker='o', color='g')
axs[2].set_title('Dropout Noise')
axs[2].set_xlabel('Level')
axs[2].set_ylabel('Chamfer Distance')
axs[2].grid(True)

# 显示图表
plt.tight_layout()
# plt.show()
plt.savefig('depth_distance.png')
