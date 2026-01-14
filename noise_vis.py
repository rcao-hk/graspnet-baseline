import os
import numpy as np
import matplotlib.pyplot as plt

experiment_root = '/media/user/data1/rcao/result/ignet/experiment'
# 模型列表和列名称
# model_list = [
#     'gsnet_base.clear', 'gsnet_base.0.002', 'gsnet_base.0.005', 'gsnet_base.0.01', 
#     'scale_grasp.clear', 'scale_grasp.0.002', 'scale_grasp.0.005', 'scale_grasp.0.01',
#     'gsnet.clear', 'gsnet.0.002', 'gsnet.0.005', 'gsnet.0.01', 
#     'ignet_v0.6.2.clear', 'ignet_v0.6.2.0.002', 'ignet_v0.6.2.0.005', 'ignet_v0.6.2.0.01',
#     'ignet_v0.8.2.clear', 'ignet_v0.8.2.0.002', 'ignet_v0.8.2.0.005', 'ignet_v0.8.2.0.01'
# ]
# noise_levels = [0, 0.002, 0.005, 0.01]
# noise_type = 'gassuian'

# model_list = [
#     'gsnet_base.clear', 'gsnet_base.s5', 'gsnet_base.s15', 'gsnet_base.s29', 
#     'scale_grasp.clear', 'scale_grasp.s5', 'scale_grasp.s15', 'scale_grasp.s29',
#     'gsnet.clear', 'gsnet.s5', 'gsnet.s15', 'gsnet.s29', 
#     'ignet_v0.6.2.clear', 'ignet_v0.6.2.s5', 'ignet_v0.6.2.s15', 'ignet_v0.6.2.s29',
#     'ignet_v0.8.2.clear', 'ignet_v0.8.2.s5', 'ignet_v0.8.2.s15', 'ignet_v0.8.2.s29'
# ]
# noise_levels = [0, 5, 15, 29]
# noise_type = 'smooth'

# scale_grasp need to redo
# model_list = [
#     'gsnet.clear', 'gsnet.d1', 'gsnet.d2', 'gsnet.d3', 
#     'scale_grasp.clear', 'scale_grasp.d1', 'scale_grasp.d2', 'scale_grasp.d3',
#     'ignet_v0.6.2.clear', 'ignet_v0.6.2.d1', 'ignet_v0.6.2.d2', 'ignet_v0.6.2.d3',
#     'ignet_v0.8.2.clear', 'ignet_v0.8.2.d1', 'ignet_v0.8.2.d2', 'ignet_v0.8.2.d3'
# ]
# noise_levels = [0, 1, 2, 3]
# noise_type = 'dropout'

# model_list = [
#     # 'gsnet.ds0.002', 'gsnet.ds0.005', 'gsnet.ds0.007', 'gsnet.ds0.01',
#     # 'scale_grasp.ds0.002', 'scale_grasp.ds0.005', 'scale_grasp.ds0.007', 'scale_grasp.ds0.01',
#     'gsnet_base.ds0.002', 'gsnet_base.ds0.005', 'gsnet_base.ds0.007', 'gsnet_base.ds0.01',
#     'ignet_v0.6.2.ds0.002', 'ignet_v0.6.2.ds0.005', 'ignet_v0.6.2.ds0.007', 'ignet_v0.6.2.ds0.01',
#     'ignet_v0.8.2.ds0.002', 'ignet_v0.8.2.ds0.005', 'ignet_v0.8.2.ds0.007', 'ignet_v0.8.2.ds0.01',
#     'scale_grasp.ds0.002', 'scale_grasp.ds0.01', 'scale_grasp.ds0.007', 'scale_grasp.ds0.005'
# ]
# noise_levels = [0.002, 0.005, 0.007, 0.01]

# noise_type = 'sparse'

# model_list = [
#     'gsnet.clear', 'gsnet.7500', 'gsnet.3750', 'gsnet.1875',
#     'scale_grasp.clear', 'scale_grasp.7500', 'scale_grasp.3750', 'scale_grasp.1875',
#     'ignet_v0.6.2.clear', 'ignet_v0.6.2.7500.1024', 'ignet_v0.6.2.3750.512', 'ignet_v0.6.2.1875.256',
#     'ignet_v0.8.2.clear', 'ignet_v0.8.2.7500.1024', 'ignet_v0.8.2.3750.512', 'ignet_v0.8.2.1875.256'
# ]
# noise_levels = [100, 50, 25, 12.5]
# noise_type = 'sparse_number'

model_list = [
    'gsnet_base.12000', 'gsnet_base.9000', 'gsnet_base.6000', 'gsnet_base.3000',
    'scale_grasp.12000', 'scale_grasp.9000', 'scale_grasp.6000', 'scale_grasp.3000',
    'gsnet.12000', 'gsnet.9000', 'gsnet.6000', 'gsnet.3000',
    'ignet_v0.6.2.12000', 'ignet_v0.6.2.9000', 'ignet_v0.6.2.6000', 'ignet_v0.6.2.3000',
    'ignet_v0.8.2.12000', 'ignet_v0.8.2.9000', 'ignet_v0.8.2.6000', 'ignet_v0.8.2.3000'
]
noise_levels = [12000, 9000, 6000, 3000]
noise_type = 'sparse_number'
camera_type = 'realsense'


# 噪声级别和数据存储
for split in ['seen', 'similar', 'novel']:

    gsnet_ap_split = []
    scale_grasp_ap_split = []
    anygrasp_ap_split = []
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
        
        if 'gsnet_base' in model:
            gsnet_ap_split.append(ap_mean)
        elif 'scale_grasp' in model:
            scale_grasp_ap_split.append(ap_mean)
        elif 'gsnet' in model:
            anygrasp_ap_split.append(ap_mean)
        elif 'ignet_v0.8.2' in model:
            ignet_ap_split.append(ap_mean)
        elif 'ignet_v0.6.2' in model:
            ignet_baseline_ap_split.append(ap_mean)

    gs_split_differences = [g - i for g, i in zip(ignet_ap_split, gsnet_ap_split)]
    scale_split_differences = [g - i for g, i in zip(ignet_ap_split, scale_grasp_ap_split)]
    anygrasp_split_differences = [g - i for g, i in zip(ignet_ap_split, anygrasp_ap_split)]
    
    # 绘图
    plt.figure(figsize=(10, 7))
    try:
        plt.plot(noise_levels, gsnet_ap_split, label='GSNet', marker='o')
    except:
        pass
    try:
        plt.plot(noise_levels, scale_grasp_ap_split, label='ScaleGrasp', marker='^')
    except:
        pass
    try:
        plt.plot(noise_levels, anygrasp_ap_split, label='AnyGrasp', marker='v')
    except:
        pass
    try:
        plt.plot(noise_levels, ignet_baseline_ap_split, label='MMGNet (Baseline)', marker='x')
    except:
        pass
    try:
        plt.plot(noise_levels, ignet_ap_split, label='MMGNet', marker='s')
    except:
        pass
    # plt.plot(noise_levels, differences, label='Difference (MMGNet - GSNet)', linestyle='--', marker='d')
    if noise_type == 'sparse_number':
        plt.gca().invert_xaxis()
    # 为每个点标注差值
    for x, diff, base, ig in zip(noise_levels, gs_split_differences, gsnet_ap_split, ignet_ap_split):
        plt.annotate(f'{diff:.1f}', (x, (base + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    for x, diff, base, ig in zip(noise_levels, scale_split_differences, scale_grasp_ap_split, ignet_ap_split):
        plt.annotate(f'{diff:.1f}', (x, (base + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    for x, diff, base, ig in zip(noise_levels, anygrasp_split_differences, anygrasp_ap_split, ignet_ap_split):
        plt.annotate(f'{diff:.1f}', (x, (base + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        
    # 图例和标题
    plt.title(split + ' $\mathbf{AP}_{mean}$ vs ' + noise_type + ' Noise Level', fontsize=14)
    plt.xlabel('Noise Level', fontsize=12)
    plt.ylabel('$\mathbf{AP}_{mean}$', fontsize=12)
    plt.xticks(noise_levels)
    plt.legend()
    plt.grid(True)
    # plt.show()
    # plt.savefig('{}_ap_mean_vs_{}_noise.svg'.format(split, noise_type), format='svg', dpi=800)
    plt.savefig('{}_ap_mean_vs_{}_noise.png'.format(split, noise_type), dpi=800)


gsnet_ap_mean  = []
scale_grasp_ap_mean = []
anygrasp_ap_mean = []
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
    
    if 'gsnet_base' in model:
        gsnet_ap_mean.append(ap_mean)
    elif 'scale_grasp' in model:
        scale_grasp_ap_mean.append(ap_mean)
    elif 'gsnet' in model:
        anygrasp_ap_mean.append(ap_mean)
    elif 'ignet_v0.6.2' in model:
        ignet_baseline_ap_mean.append(ap_mean)
    elif 'ignet_v0.8.2' in model:
        ignet_ap_mean.append(ap_mean)


gs_differences = [g - i for g, i in zip(ignet_ap_mean, gsnet_ap_mean)]
scale_differences = [g - i for g, i in zip(ignet_ap_mean, scale_grasp_ap_mean)]
anygrasp_differences = [g - i for g, i in zip(ignet_ap_mean, anygrasp_ap_mean)]
# 绘图
plt.figure(figsize=(10, 7))
try:
    plt.plot(noise_levels, gsnet_ap_mean, label='GSNet', marker='o')
except:
    pass
try:
    plt.plot(noise_levels, scale_grasp_ap_mean, label='ScaleGrasp', marker='^')
except:
    pass
try:
    plt.plot(noise_levels, anygrasp_ap_mean, label='AnyGrasp', marker='v')
except:
    pass
try:
    plt.plot(noise_levels, ignet_baseline_ap_mean, label='MMGNet (Baseline)', marker='x')
except:
    pass
try:
    plt.plot(noise_levels, ignet_ap_mean, label='MMGNet', marker='s')
except:
    pass

if noise_type == 'sparse_number':
    plt.gca().invert_xaxis()
# 为每个点标注差值
for x, diff, base, ig in zip(noise_levels, gs_differences, gsnet_ap_mean, ignet_ap_mean):
    plt.annotate(f'{diff:.1f}', (x, (base + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

for x, diff, base, ig in zip(noise_levels, scale_differences, scale_grasp_ap_mean, ignet_ap_mean):
    plt.annotate(f'{diff:.1f}', (x, (base + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
for x, diff, base, ig in zip(noise_levels, anygrasp_differences, anygrasp_ap_mean, ignet_ap_mean):
    plt.annotate(f'{diff:.1f}', (x, (base + ig) / 2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
# 图例和标题
plt.title('$\mathbf{AP}_{mean}$ vs ' + noise_type + ' Noise Level', fontsize=14)
plt.xlabel('Noise Level', fontsize=12)
plt.ylabel('$\mathbf{AP}_{mean}$', fontsize=12)
plt.xticks(noise_levels)
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('ap_mean_vs_{}_noise.svg'.format(noise_type), format='svg', dpi=800)
plt.savefig('ap_mean_vs_{}_noise.png'.format(noise_type), dpi=800)