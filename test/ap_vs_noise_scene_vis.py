import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 平滑函数：高斯滤波
def gaussian_smoothing(data, sigma):
    return gaussian_filter1d(data, sigma=sigma)

# 加载 .npy 文件
clear_depth_path = os.path.join('depth_distance',"clear_depth_distance.npy")
# clear_depth_path = os.path.join('clear_depth_distance.npy')
clear_depth_dis = np.load(clear_depth_path)  # (scene_list, 256, 1)
clear_depth_dis = np.mean(clear_depth_dis, axis=1).squeeze(-1)  # (scene_list,)

# 定义 splits
splits = {
    'seen': clear_depth_dis[:30],
    'similar': clear_depth_dis[30:60],
    'novel': clear_depth_dis[60:90]
}

# 模型列表及其对应名称
# model_list = ['ignet_v0.6.2', 'gsnet', 'gsnet_base', 'scale_grasp']
# model_name_list = ['MMGNet (Baseline)', 'Anygrasp', 'GSNet', 'ScaleGrasp']
model_list = ['gsnet_base', 'scale_grasp', 'gsnet', 'ignet_v0.6.2']
model_name_list = ['GSNet', 'ScaleGrasp', 'Anygrasp', 'MMGNet (Baseline)']
experiment_root = 'experiment'
camera_type = 'realsense'

mmgnet_vis_model_list = ['ignet_v0.8.2.26.2']

for mmgnet_vis_model in mmgnet_vis_model_list:
    # 存储每个模型的AP结果
    ap_result = {}

    # 加载 AP 数据
    for model in [mmgnet_vis_model] + model_list:
        root = os.path.join(experiment_root, model)
        result_list = []
        for split in ['seen', 'similar', 'novel']:
            res = np.load(os.path.join(root, f'ap_test_{split}_{camera_type}.npy'))
            result_list.extend(res)
        res = np.array(result_list)  # (3, 256, 50, 6)
        ap_scene = np.mean(res[:, :, :50, :], axis=(1, 2, 3)) * 100  # 平均AP
        ap_result[model] = ap_scene

    # 创建多子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=False)

    bar_width = 0.2  # 控制柱状图宽度
    x = np.array([0, 1, 2])  # x轴代表高、中、低三个分组

    for idx, (split, depth_dis) in enumerate(splits.items()):
        # 分为高、中、低三组
        high_threshold = np.percentile(depth_dis, 66)
        low_threshold = np.percentile(depth_dis, 33)

        high_group = np.where(depth_dis >= high_threshold)[0]
        medium_group = np.where((depth_dis < high_threshold) & (depth_dis >= low_threshold))[0]
        low_group = np.where(depth_dis < low_threshold)[0]

        # 计算 AP 差值
        ap_diff_results = {
            'high': {model: ap_result[mmgnet_vis_model][high_group] - ap_result[model][high_group] for model in model_list},
            'medium': {model: ap_result[mmgnet_vis_model][medium_group] - ap_result[model][medium_group] for model in model_list},
            'low': {model: ap_result[mmgnet_vis_model][low_group] - ap_result[model][low_group] for model in model_list},
        }

        # 平均AP差值
        avg_ap_diff_results = {
            group: {model: np.mean(ap_diff_results[group][model]) for model in model_list}
            for group in ['high', 'medium', 'low']
        }

        # 绘制子图
        ax = axes[idx]
        
        # 为每个方法分配不同的 x 位置
        offset = np.array([-bar_width, 0, bar_width, 2 * bar_width])  # 每个模型的偏移量

        for i, (model, model_name) in enumerate(zip(model_list, model_name_list)):
            avg_diff = [avg_ap_diff_results[group][model] for group in ['high', 'medium', 'low']]
            bars = ax.bar(x + offset[i], avg_diff, width=bar_width, label=f'{model_name}')
                    # 在每个柱上方显示数字
            for bar in bars:
                height = bar.get_height()  # 获取每个柱的高度
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # x 坐标：柱子的中心位置
                    height,  # y 坐标：柱子的顶部，稍微上移一点
                    f'{height:.2f}',  # 显示的数字（保留两位小数）
                    ha='center',  # 水平居中
                    va='bottom',  # 垂直对齐到顶部
                    fontsize=12,  # 字体大小
                    color='black'  # 字体颜色
                )
                
        ax.set_ylim(bottom=0)
        ax.set_xticks(x)
        # ax.set_xticklabels(['High Distance', 'Medium Distance', 'Low Distance'], fontsize=12)
        ax.set_xticklabels(['High Noise', 'Medium Noise', 'Low Noise'], fontsize=12)
        
        ax.set_title(f'{split.capitalize()} Split', fontsize=14)
        ax.set_ylabel('Average AP Difference', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        # ax.grid(True)

    axes[-1].set_xlabel('Point Cloud Distance Group', fontsize=12)
    fig.suptitle('AP Difference for Different Methods and Distance Groups', fontsize=16)

    # 调整布局并保存
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join('vis', f'{mmgnet_vis_model}_ap_diff_groups.png'))
    # plt.show()
