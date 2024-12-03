import os
import numpy as np
import matplotlib.pyplot as plt

experiment_root = '/media/user/data1/rcao/result/ignet/experiment'
# 模型列表和列名称
# model_list = [
#     'gsnet.clear', 'gsnet.0.002', 'gsnet.0.005', 'gsnet.0.01', 
#     'ignet_v0.6.2.clear', 'ignet_v0.6.2.0.002', 'ignet_v0.6.2.0.005', 'ignet_v0.6.2.0.01',
#     'ignet_v0.8.2.clear', 'ignet_v0.8.2.0.002', 'ignet_v0.8.2.0.005', 'ignet_v0.8.2.0.01'
# ]
# noise_levels = [0, 0.002, 0.005, 0.01]
# noise_type = 'gassuian' # 'smooth'  'dropout'

model_list = [
    'gsnet.clear', 'gsnet.s5', 'gsnet.s15', 'gsnet.s29', 
    'ignet_v0.6.2.clear', 'ignet_v0.6.2.s5', 'ignet_v0.6.2.s15', 'ignet_v0.6.2.s29',
    'ignet_v0.8.2.clear', 'ignet_v0.8.2.s5', 'ignet_v0.8.2.s15', 'ignet_v0.8.2.s29'
]
noise_levels = [0, 5, 15, 29]
noise_type = 'smooth'

# model_list = [
#     'gsnet.clear', 'gsnet.d1', 'gsnet.d2', 'gsnet.d3', 
#     'ignet_v0.6.2.clear', 'ignet_v0.6.2.d1', 'ignet_v0.6.2.d2', 'ignet_v0.6.2.d3',
#     'ignet_v0.8.2.clear', 'ignet_v0.8.2.d1', 'ignet_v0.8.2.d2', 'ignet_v0.8.2.d3'
# ]
# noise_levels = [0, 1, 2, 3]
# noise_type = 'dropout'

camera_type = 'realsense'


# 噪声级别和数据存储
for split in ['seen', 'similar', 'novel']:

    gsnet_ap_split = []
    ignet_baseline_ap_split = []
    ignet_ap_split = []

    # 遍历模型，计算AP_mean
    for model in model_list:
        root = os.path.join(experiment_root, model)
        split_ap = []

        res = np.load(os.path.join(root, f'ap_test_{split}_{camera_type}.npy'))
        ap_top50 = np.mean(res[:, :, :50, :])
        split_ap.append(ap_top50)
        # 计算AP_mean
        ap_mean = np.mean(split_ap) * 100.0
        
        if 'gsnet' in model:
            gsnet_ap_split.append(ap_mean)
        elif 'ignet_v0.8.2' in model:
            ignet_ap_split.append(ap_mean)
        elif 'ignet_v0.6.2' in model:
            ignet_baseline_ap_split.append(ap_mean)

    split_differences = [g - i for g, i in zip(ignet_ap_split, gsnet_ap_split)]

    # 绘图
    plt.figure(figsize=(10, 7))
    try:
        plt.plot(noise_levels, gsnet_ap_split, label='GSNet', marker='o')
    except:
        pass
    plt.plot(noise_levels, ignet_baseline_ap_split, label='MMGNet (Baseline)', marker='x')
    plt.plot(noise_levels, ignet_ap_split, label='MMGNet', marker='s')
    # plt.plot(noise_levels, differences, label='Difference (MMGNet - GSNet)', linestyle='--', marker='d')

    # 为每个点标注差值
    for x, diff, base, ig in zip(noise_levels, split_differences, gsnet_ap_split, ignet_ap_split):
        plt.annotate(f'{diff:.1f}', (x, (base + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    # 图例和标题
    plt.title(split + ' $\mathbf{AP}_{mean}$ vs ' + noise_type + ' Noise Level', fontsize=14)
    plt.xlabel('Noise Level', fontsize=12)
    plt.ylabel('$\mathbf{AP}_{mean}$', fontsize=12)
    plt.xticks(noise_levels)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('{}_ap_mean_vs_{}_noise.png'.format(split, noise_type))


gsnet_ap_mean  = []
ignet_baseline_ap_mean = []
ignet_ap_mean = []
# 遍历模型，计算AP_mean
for model in model_list:
    root = os.path.join(experiment_root, model)
    split_ap = []
    for split in ['seen', 'similar', 'novel']:
        res = np.load(os.path.join(root, f'ap_test_{split}_{camera_type}.npy'))
        ap_top50 = np.mean(res[:, :, :50, :])
        split_ap.append(ap_top50)
    # 计算AP_mean
    ap_mean = np.mean(split_ap) * 100.0
    
    if 'gsnet' in model:
        gsnet_ap_mean.append(ap_mean)
    elif 'ignet_v0.6.2' in model:
        ignet_baseline_ap_mean.append(ap_mean)
    elif 'ignet_v0.8.2' in model:
        ignet_ap_mean.append(ap_mean)


differences = [g - i for g, i in zip(ignet_ap_mean, gsnet_ap_mean)]
# 绘图
plt.figure(figsize=(10, 7))
try:
    plt.plot(noise_levels, gsnet_ap_mean, label='GSNet', marker='o')
except:
    pass
plt.plot(noise_levels, ignet_baseline_ap_mean, label='MMGNet (Baseline)', marker='x')
plt.plot(noise_levels, ignet_ap_mean, label='MMGNet', marker='s')

# 为每个点标注差值
for x, diff, base, ig in zip(noise_levels, differences, gsnet_ap_mean, ignet_ap_mean):
    plt.annotate(f'{diff:.1f}', (x, (base + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

# 图例和标题
plt.title('$\mathbf{AP}_{mean}$ vs ' + noise_type + ' Noise Level', fontsize=14)
plt.xlabel('Noise Level', fontsize=12)
plt.ylabel('$\mathbf{AP}_{mean}$', fontsize=12)
plt.xticks(noise_levels)
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('ap_mean_vs_{}_noise.png'.format(noise_type))