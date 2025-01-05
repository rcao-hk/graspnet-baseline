import numpy as np
import os
import matplotlib.pyplot as plt

# 加载 .npy 文件
clear_depth_path = os.path.join("clear_depth_distance.npy")
clear_depth_dis = np.load(clear_depth_path)  # (scene_list, 256, 1)
clear_depth_dis = np.mean(clear_depth_dis, axis=1).squeeze(-1)  # (scene_list,)
# clear_depth_dis = clear_depth_dis[:30]
# vis_split = 'seen'
clear_depth_dis = clear_depth_dis[30:60]
vis_split = 'similar'
# clear_depth_dis = clear_depth_dis[60:90]
# vis_split = 'novel'
# clear_depth_dis = clear_depth_dis
# vis_split = 'full'

# top_k = 10
# if top_k != 30:
#     vis_split = vis_split + '_' + str(top_k)
# scene_list = np.argsort(clear_depth_dis)[::-1][:top_k]  # 按降序排列场景
# scene_list = np.arange(len(clear_depth_dis))  # 重新编号

percentile = 80
threshold = np.percentile(clear_depth_dis, percentile)
print(f"The {percentile}th percentile for {vis_split} split is: {threshold}")

vis_split = vis_split + '_' + str(percentile)
# 找到小于该分位值的场景索引
scene_list = np.where(clear_depth_dis >= threshold)[0]
scene_order = np.argsort(clear_depth_dis[scene_list])[::-1]
scene_list = scene_list[scene_order]
print(f"Number of scenes: {len(scene_list)}")

# 模型列表
model_list = ['ignet_v0.8.2', 'ignet_v0.6.2', 'gsnet', 'gsnet_base', 'scale_grasp']
experiment_root = '/media/gpuadmin/rcao/result/ignet/experiment'
camera_type = 'realsense'

# 存储每个模型的AP结果
ap_result = {}

for model in model_list:
    if model == 'ignet_v0.8.2':
        root = os.path.join(experiment_root, 'ignet_v0.8.3.10')
    else:
        root = os.path.join(experiment_root, model)
    result_list = []
    # 加载 'seen' 数据
    for split in ['seen', 'similar', 'novel']:
        res = np.load(os.path.join(root, f'ap_test_{split}_{camera_type}.npy'))
        result_list.extend(res)
    res = np.array(result_list)  # (3, 256, 50, 6)
    # 按场景计算AP
    ap_scene = np.mean(res[:, :, :50, :], axis=(1, 2, 3)) * 10 # 平均AP
    
    # res_ap_max = np.min(res[:, :, :50, :], axis=1)  # (3, 50, 6)
    # ap_scene = np.mean(res_ap_max, axis=(1, 2)) * 10  # 最大AP
    ap_result[model] = ap_scene

from scipy.ndimage import gaussian_filter1d
def gaussian_smoothing(data, sigma):
    return gaussian_filter1d(data, sigma=sigma)

# 计算差值
diff_mmgnet_vs_baseline = ap_result['ignet_v0.8.2'] - ap_result['ignet_v0.6.2']
diff_mmgnet_vs_anygrasp = ap_result['ignet_v0.8.2'] - ap_result['gsnet']
diff_mmgnet_vs_gsnet = ap_result['ignet_v0.8.2'] - ap_result['gsnet_base']
diff_mmgnet_vs_scale_grasp = ap_result['ignet_v0.8.2'] - ap_result['scale_grasp']

# 可视化
fig, ax1 = plt.subplots(figsize=(14, 8))

# 主坐标轴：绘制 AP Difference
scene_indices = np.arange(len(scene_list))
sorted_diff_mmgnet_vs_baseline = diff_mmgnet_vs_baseline[scene_list]
sorted_diff_mmgnet_vs_anygrasp = diff_mmgnet_vs_anygrasp[scene_list]
sorted_diff_mmgnet_vs_gsnet = diff_mmgnet_vs_gsnet[scene_list]
sorted_diff_mmgnet_vs_scale_grasp = diff_mmgnet_vs_scale_grasp[scene_list]

# 对差值进行平滑处理
window_size = 5  # 滑动窗口大小
sigma = 1  # 高斯滤波的标准差

smooth_diff_mmgnet_vs_baseline = gaussian_smoothing(diff_mmgnet_vs_baseline[scene_list], sigma=sigma)
smooth_diff_mmgnet_vs_anygrasp = gaussian_smoothing(diff_mmgnet_vs_anygrasp[scene_list], sigma=sigma)
smooth_diff_mmgnet_vs_gsnet = gaussian_smoothing(diff_mmgnet_vs_gsnet[scene_list], sigma=sigma)
smooth_diff_mmgnet_vs_scale_grasp = gaussian_smoothing(diff_mmgnet_vs_scale_grasp[scene_list], sigma=sigma)

# 可视化
fig, ax1 = plt.subplots(figsize=(14, 8))
scene_indices = np.arange(len(scene_list))

# 绘制原始点和平滑后的曲线：MMGNet - MMGNet (Baseline)
ax1.scatter(scene_indices, diff_mmgnet_vs_baseline[scene_list], label='Original: MMGNet - MMGNet (Baseline)', color='blue', alpha=0.6)
ax1.plot(scene_indices, smooth_diff_mmgnet_vs_baseline, label='Smoothed: MMGNet - MMGNet (Baseline)', color='blue', linestyle='--')

# 绘制原始点和平滑后的曲线：MMGNet - Anygrasp
ax1.scatter(scene_indices, diff_mmgnet_vs_anygrasp[scene_list], label='Original: MMGNet - Anygrasp', color='orange', alpha=0.6)
ax1.plot(scene_indices, smooth_diff_mmgnet_vs_anygrasp, label='Smoothed: MMGNet - Anygrasp', color='orange', linestyle='--')

# 绘制原始点和平滑后的曲线：MMGNet - GSNet
ax1.scatter(scene_indices, diff_mmgnet_vs_gsnet[scene_list], label='Original: MMGNet - GSNet', color='purple', alpha=0.6)
ax1.plot(scene_indices, smooth_diff_mmgnet_vs_gsnet, label='Smoothed: MMGNet - GSNet', color='purple', linestyle='--')

# 绘制原始点和平滑后的曲线：MMGNet - Scale Grasp
ax1.scatter(scene_indices, diff_mmgnet_vs_scale_grasp[scene_list], label='Original: MMGNet - Scale Grasp', color='red', alpha=0.6)
ax1.plot(scene_indices, smooth_diff_mmgnet_vs_scale_grasp, label='Smoothed: MMGNet - Scale Grasp', color='red', linestyle='--')

# 添加水平线和标签
# ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, label='No Difference')
ax1.set_xlabel('Scene Index (Sorted by Clear Depth)', fontsize=12)
ax1.set_ylabel('AP Difference', fontsize=12)
ax1.tick_params(axis='y')
ax1.set_xticks(scene_indices)
ax1.set_xticklabels(scene_list, rotation=90)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True)

# 副坐标轴：绘制 Depth Distance
ax2 = ax1.twinx()
sorted_depth_dis = clear_depth_dis[scene_list]
ax2.plot(scene_indices, sorted_depth_dis, label='Depth Distance', linestyle='--', marker='x', color='green')
ax2.set_ylabel('Depth Distance', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')

# 图表标题
fig.suptitle('AP Difference and Depth Distance per Scene', fontsize=16)
# plt.show()

plt.savefig(os.path.join('vis', f'ap_{vis_split}_vs_noise_scene_vis_all.png'))
