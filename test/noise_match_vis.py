import numpy as np
import os
import matplotlib.pyplot as plt

gaussian_noise_levels = [0.001, 0.003, 0.005, 0.007, 0.01]
blur_sigma_colors=[0.1, 0.3, 0.5, 0.7, 1.0]

# 构造搜索空间
search_space = [(noise, blur) for noise in gaussian_noise_levels for blur in blur_sigma_colors]
data_root = 'depth_distance'
depth_dis_list = []
for gaussian_noise, blur_sigma_color in search_space:
    # 保存搜索空间
    depth_dis = np.load(os.path.join(data_root, 'g{}bss5.0bsc{}_depth_distance.npy'.format(gaussian_noise, blur_sigma_color)))
    # 读取搜索空间
    depth_dis_list.append(np.mean(depth_dis))
    print("gaussian_noise: {}, blur_sigma_color: {}, depth distance: {}".format(gaussian_noise, blur_sigma_color, np.mean(depth_dis)))
    
min_depth_dis = np.argmin(depth_dis_list)
print(search_space[min_depth_dis])

# 加载 .npy 文件
clear_depth_path = os.path.join(data_root, "clear_depth_distance.npy")
clear_depth_dis = np.load(clear_depth_path)  # (scene_list, 256, 1)
clear_depth_dis = np.mean(clear_depth_dis, axis=1)  # (scene_list, 1)

match_depth_path = os.path.join(data_root, "match_depth_distance.npy")
match_depth_dis = np.load(match_depth_path)  # (scene_list, 256, 1)
match_depth_dis = np.mean(match_depth_dis, axis=1)  # (scene_list, 1)

gaussian_noise, blur_sigma_color = search_space[min_depth_dis]
our_depth_path = os.path.join(data_root, 'g{}bss5.0bsc{}_depth_distance.npy'.format(gaussian_noise, blur_sigma_color))
our_depth_dis = np.load(our_depth_path)  # (scene_list, 256, 1)
our_depth_dis = np.mean(our_depth_dis, axis=1)  # (scene_list, 1)

# 计算每个场景的两个 CD 的平均值
scene_list = our_depth_dis.shape[0]

# 绘制折线图
plt.figure(figsize=(12, 6))
x = np.arange(scene_list)

# 绘制两条曲线
plt.plot(x, clear_depth_dis, label="CD of clear PCs", marker='o')
# plt.plot(x, match_depth_dis, label="CD of Simulated PCs (SimSense)", marker='o')
plt.plot(x, our_depth_dis, label="CD of Simulated PCs (Ours))", marker='o')
# 设置图形属性
plt.xlabel("Scene Index")
plt.ylabel("Average Chamfer Distance")
plt.title("Average Chamfer Distances per Scene")
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.savefig("cd_per_scene.png")
# plt.show()

print("Average CD of clear PCs: {}, Simulated PCs (DRED): {}, Simulated PCs (Ours): {}".format(np.mean(clear_depth_dis), np.mean(match_depth_dis), np.mean(our_depth_dis)))