import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import colorsys

palette = sns.color_palette("muted", n_colors=5)  # 或 "pastel"
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
# plt.rcParams['text.usetex'] = True


# def desaturate(color, factor=0.8):
#     """降低饱和度"""
#     rgb = mcolors.to_rgb(color)
#     h, l, s = colorsys.rgb_to_hls(*rgb)   # 用 colorsys 而不是 matplotlib.colors
#     return colorsys.hls_to_rgb(h, l, s * factor)

# # 获取 tab10 调色板并降低饱和度
# colors = [desaturate(c, 0.7) for c in plt.cm.tab10.colors]
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


# model_list = [
#     'gsnet_base.clear', 'gsnet_base.s5', 'gsnet_base.s15', 'gsnet_base.s29', 
#     'scale_grasp.clear', 'scale_grasp.s5', 'scale_grasp.s15', 'scale_grasp.s29',
#     'gsnet.clear', 'gsnet.s5', 'gsnet.s15', 'gsnet.s29', 
#     'mmgnet_baseline.clear', 'mmgnet_baseline.s5', 'mmgnet_baseline.s15', 'mmgnet_baseline.s29',
#     'mmgnet_scene.clear', 'mmgnet_scene.s5', 'mmgnet_scene.s15', 'mmgnet_scene.s29'
# ]
# noise_levels = [0, 5, 15, 29]
# noise_type = 'smooth'
# bar_font_size = 14

# groups = {
#     'GSNet': ['gsnet_base.clear', 'gsnet_base.s5', 'gsnet_base.s15', 'gsnet_base.s29'],
#     'Scale_grasp': ['scale_grasp.clear', 'scale_grasp.s5', 'scale_grasp.s15', 'scale_grasp.s29'],
#     'Anygrasp': ['gsnet.clear', 'gsnet.s5', 'gsnet.s15', 'gsnet.s29'],
#     'Our Baseline': ['mmgnet_baseline.clear', 'mmgnet_baseline.s5', 'mmgnet_baseline.s15', 'mmgnet_baseline.s29'],
#     'MMGNet': ['mmgnet_scene.clear', 'mmgnet_scene.s5', 'mmgnet_scene.s15', 'mmgnet_scene.s29']
# }



# model_list = [
#     'gsnet_base.clear', 'gsnet_base.0.002', 'gsnet_base.0.005', 'gsnet_base.0.01', 
#     'scale_grasp.clear', 'scale_grasp.0.002', 'scale_grasp.0.005', 'scale_grasp.0.01',
#     'gsnet.clear', 'gsnet.0.002', 'gsnet.0.005', 'gsnet.0.01', 
#     'mmgnet_baseline.clear', 'mmgnet_baseline.0.002', 'mmgnet_baseline.0.005', 'mmgnet_baseline.0.01',
#     'mmgnet_scene.clear', 'mmgnet_scene.0.002', 'mmgnet_scene.0.005', 'mmgnet_scene.0.01'
# ]
# noise_levels = [0, 0.002, 0.005, 0.01]
# noise_type = 'gaussian'
# bar_font_size = 14

# groups = {
# 'GSNet': ['gsnet_base.clear', 'gsnet_base.0.002', 'gsnet_base.0.005', 'gsnet_base.0.01'],
# 'Scale_grasp': ['scale_grasp.clear', 'scale_grasp.0.002', 'scale_grasp.0.005', 'scale_grasp.0.01'],
# 'Anygrasp': ['gsnet.clear', 'gsnet.0.002', 'gsnet.0.005', 'gsnet.0.01'],
# 'Our Baseline': ['mmgnet_baseline.clear', 'mmgnet_baseline.0.002', 'mmgnet_baseline.0.005', 'mmgnet_baseline.0.01'],
# 'MMGNet': ['mmgnet_scene.clear', 'mmgnet_scene.0.002', 'mmgnet_scene.0.005', 'mmgnet_scene.0.01']
# }


# model_list = [
#     # 'gsnet_base.clear', 'gsnet_base.d1', 'gsnet_base.d2', 'gsnet_base.d3', 
#     'scale_grasp.clear', 'scale_grasp.d1', 'scale_grasp.d2', 'scale_grasp.d3',
#     'gsnet.clear', 'gsnet.d1', 'gsnet.d2', 'gsnet.d3', 
#     'mmgnet_baseline.clear', 'mmgnet_baseline.d1', 'mmgnet_baseline.d2', 'mmgnet_baseline.d3',
#     'mmgnet_scene.clear', 'mmgnet_scene.d1', 'mmgnet_scene.d2', 'mmgnet_scene.d3'
# ]
# noise_levels = [0, 1, 2, 3]
# noise_type = 'dropout'
# bar_font_size = 14

# groups = {
#     # 'GSNet': ['gsnet_base.clear', 'gsnet_base.s5', 'gsnet_base.s15', 'gsnet_base.s23'],
#     'Scale_grasp': ['scale_grasp.clear', 'scale_grasp.d1', 'scale_grasp.d2', 'scale_grasp.d3'],
#     'Anygrasp': ['gsnet.clear', 'gsnet.d1', 'gsnet.d2', 'gsnet.d3'],
#     'MMGNet (Baseline)': ['mmgnet_baseline.clear', 'mmgnet_baseline.d1', 'mmgnet_baseline.d2', 'mmgnet_baseline.d3'],
#     'MMGNet': ['mmgnet_scene.clear', 'mmgnet_scene.d1', 'mmgnet_scene.d2', 'mmgnet_scene.d3']
# }


model_list = [
    'gsnet_base.clear', 'gsnet_base.dr0.2', 'gsnet_base.dr0.4', 'gsnet_base.dr0.6', 'gsnet_base.dr0.8', 'gsnet_base.dr1.0',
    'scale_grasp.clear', 'scale_grasp.dr0.2', 'scale_grasp.dr0.4', 'scale_grasp.dr0.6', 'scale_grasp.dr0.8', 'scale_grasp.dr1.0',
    'gsnet.clear', 'gsnet.dr0.2', 'gsnet.dr0.4', 'gsnet.dr0.6', 'gsnet.dr0.8', 'gsnet.dr1.0',
    'mmgnet_baseline.clear', 'mmgnet_baseline.dr0.2', 'mmgnet_baseline.dr0.4', 'mmgnet_baseline.dr0.6', 'mmgnet_baseline.dr0.8', 'mmgnet_baseline.dr1.0',
    'mmgnet_scene.clear', 'mmgnet_scene.dr0.2', 'mmgnet_scene.dr0.4', 'mmgnet_scene.dr0.6', 'mmgnet_scene.dr0.8', 'mmgnet_scene.dr1.0'
]
noise_levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
noise_type = 'controlled_dropout'
bar_font_size = 9

groups = {
    'GSNet': ['gsnet_base.clear', 'gsnet_base.dr0.2', 'gsnet_base.dr0.4', 'gsnet_base.dr0.6', 'gsnet_base.dr0.8', 'gsnet_base.dr1.0'],
    'Scale_grasp': ['scale_grasp.clear', 'scale_grasp.dr0.2', 'scale_grasp.dr0.4', 'scale_grasp.dr0.6', 'scale_grasp.dr0.8', 'scale_grasp.dr1.0'],
    'Anygrasp': ['gsnet.clear', 'gsnet.dr0.2', 'gsnet.dr0.4', 'gsnet.dr0.6', 'gsnet.dr0.8', 'gsnet.dr1.0'],
    'Our Baseline': ['mmgnet_baseline.clear', 'mmgnet_baseline.dr0.2', 'mmgnet_baseline.dr0.4', 'mmgnet_baseline.dr0.6', 'mmgnet_baseline.dr0.8', 'mmgnet_baseline.dr1.0'],
    'MMGNet': ['mmgnet_scene.clear', 'mmgnet_scene.dr0.2', 'mmgnet_scene.dr0.4', 'mmgnet_scene.dr0.6', 'mmgnet_scene.dr0.8', 'mmgnet_scene.dr1.0']
}


# # 读取数据
data = pd.read_csv('{}_noise.csv'.format(noise_type), index_col=0)
data = data.loc[model_list]

# 计算 AP_mean
data['AP_mean'] = data[['AP_seen', 'AP_similar', 'AP_novel']].mean(axis=1)

# # # 获取 MMGNet 的 AP_mean
mmgnet_ap_mean = data.loc[groups['MMGNet'], 'AP_mean'].values * 100.0


# colors = plt.get_cmap('tab10').colors  # tab10常用顺序：蓝、橙、绿、红、紫...

# 创建主图
plt.figure(figsize=(12, 8))

# 存储不同方法的曲线数据
curves = {}

# 绘制每组数据
for i, (group_name, models) in enumerate(groups.items()):
    ap_means = data.loc[models, 'AP_mean'].values * 100.0
    curves[group_name] = ap_means
    if group_name == 'Scale_grasp':
        # label_name = r'Ma \textit{et al.}'
        label_name = 'Ma et al.'
    else:
        label_name = group_name
    # plt.plot(noise_levels, ap_means, label=label_name, marker='o')

    plt.plot(
        noise_levels, ap_means,
        label=label_name,
        # color=colors[i],
        linewidth=4,
        marker=None  # 无marker
    )
# 在两条曲线之间标注差值x
# for group_name, ap_means in curves.items():
#     if group_name != 'MMGNet':  # 跳过 MMGNet 自身
#         for x, ap1, ap2 in zip(noise_levels, ap_means, mmgnet_ap_mean):
#             mid_x = x # 差值位置横坐标
#             mid_y = (ap1 + ap2) / 2  # 差值位置纵坐标
#             diff = ap2 - ap1 # 计算差值
#             plt.annotate(f'{diff:.1f}', (mid_x, mid_y), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=11, color='black')

# 图表美化
# plt.title('$\mathbf{AP}_{mean}$ vs ' + noise_type.capitalize() + ' Noise Level', fontsize=14)
plt.tick_params(axis='x', labelsize=16, color='black')
plt.tick_params(axis='y', labelsize=16, color='black')


plt.tight_layout()
plt.xlabel('Noise Level', fontsize=16, fontweight='bold')
plt.ylabel('$\mathbf{AP}_{mean}$ (%)', fontsize=16, fontweight='bold')
plt.xticks(noise_levels)
# plt.xticks(noise_levels)
# if noise_type == 'gaussian':
    # plt.legend(prop=dict(weight='bold', size=14), loc="upper right")
# elif noise_type == 'controlled_dropout':
#     plt.ylim(51, 66)
plt.grid(True)

# 显示图
plt.tight_layout()
# plt.show(
plt.savefig('ap_mean_vs_{}_noise.png'.format(noise_type), format='png', dpi=600)
plt.savefig('ap_mean_vs_{}_noise.svg'.format(noise_type), format='svg', dpi=600)


# =========================
# Print AP drop in pp (percentage points)
# =========================
print("\n===== AP_mean drop relative to CLEAR (unit: percentage points, pp) =====")
for group_name, models in groups.items():
    clear_pct = data.loc[models[0], 'AP_mean'] * 100.0
    print(f"\n[{group_name}] CLEAR AP_mean = {clear_pct:.2f}%")
    for lvl, model in zip(noise_levels[1:], models[1:]):
        ap_pct = data.loc[model, 'AP_mean'] * 100.0
        drop_pp = clear_pct - ap_pct  # percentage points
        print(f"  level={lvl}: AP_mean={ap_pct:.2f}%, drop={drop_pp:.2f} pp")
print("=======================================================================\n")


# 计算 AP drop
ap_drop_results = {}
for group_name, models in groups.items():
    clear_pct = data.loc[models[0], 'AP_mean'] * 100.0
    ap_drops_pp = [clear_pct - (data.loc[m, 'AP_mean'] * 100.0) for m in models[1:]]
    ap_drop_results[group_name] = ap_drops_pp

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 1 / (5 + 2)
x = np.arange(len(noise_levels[1:]))  # 跳过 clear 方法
# colors = ['blue', 'orange', 'green', 'red', 'purple']


for idx, (group_name, ap_drops) in enumerate(ap_drop_results.items()):
    if group_name == 'Scale_grasp':
        # label_name = r'Ma \textit{et al.}'
        label_name = 'Ma et al.'
    else:
        label_name = group_name
    bars = ax.bar(x + idx * bar_width, ap_drops, bar_width, label=label_name)
    
    # 在每个柱上方显示数字
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=bar_font_size,
            color='black',
            weight='bold'
        )

# 图表美化
# ax.set_title('AP Drop Relative to Clear Method', fontsize=14)
ax.set_xlabel('Noise Level', fontsize=16, fontweight='bold')
# ax.set_ylabel('$\mathbf{AP}_{mean}$ Drop (%)', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$\mathbf{AP}_{mean}$ Drop (pp)', fontsize=16, fontweight='bold')

ax.tick_params(axis='x', labelsize=16, color='black')
ax.tick_params(axis='y', labelsize=16, color='black')

ax.set_xticks(x + (len(ap_drop_results) - 1) * bar_width / 2)
ax.set_xticklabels(noise_levels[1:])  # 跳过 clear 的噪声级别
# ax.set_yticklabels([f'{y:.1f}' for y in ax.get_yticks()], fontsize=16, fontweight='bold')
# ax.legend(title='Methods', fontsize=10)
if noise_type == 'gaussian':
    ax.legend(prop=dict(weight='bold', size=14), loc="upper left")
ax.grid(True, linestyle='--', alpha=0.7)

# 保存图像
plt.tight_layout()
plt.savefig(f'ap_drop_vs_{noise_type}_noise.png', format='png', dpi=600)
plt.savefig(f'ap_drop_vs_{noise_type}_noise.svg', format='svg', dpi=600)

# 计算 AP 相对 drop（relative drop, %）
ap_drop_results = {}
eps = 1e-12  # 防止除零

for group_name, models in groups.items():
    clear_ap = float(data.loc[models[0], 'AP_mean'])  # 0~1
    denom = max(clear_ap, eps)
    drops_rel = []
    for model in models[1:]:
        ap = float(data.loc[model, 'AP_mean'])        # 0~1
        drop_rel = (clear_ap - ap) / denom * 100.0    # 相对下降百分比
        drops_rel.append(drop_rel)
    ap_drop_results[group_name] = drops_rel

# （可选）在终端打印每个方法的相对 drop(%)
print("\n=== AP_mean relative drop (%) ===")
for group_name, drops in ap_drop_results.items():
    print(f"{group_name:>12}: " + ", ".join([f"{d:.2f}%" for d in drops]))

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 1 / (len(ap_drop_results) + 2)
x = np.arange(len(noise_levels[1:]))  # 跳过 clear

for idx, (group_name, drops_rel) in enumerate(ap_drop_results.items()):
    label_name = 'Ma et al.' if group_name == 'Scale_grasp' else group_name
    bars = ax.bar(x + idx * bar_width, drops_rel, bar_width, label=label_name)

    # 在每个柱上方显示相对 drop(%)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f'{h:.2f}',
            ha='center',
            va='bottom',
            fontsize=bar_font_size,
            color='black',
            weight='bold'
        )

# 图表美化
ax.set_xlabel('Noise Level', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$\mathbf{AP}_{mean}$ Relative Drop (%)', fontsize=16, fontweight='bold')

ax.tick_params(axis='x', labelsize=16, color='black')
ax.tick_params(axis='y', labelsize=16, color='black')

ax.set_xticks(x + (len(ap_drop_results) - 1) * bar_width / 2)
ax.set_xticklabels(noise_levels[1:])  # 跳过 clear
ax.grid(True, linestyle='--', alpha=0.7)

ax.legend(prop=dict(weight='bold', size=14), loc="upper left")

plt.tight_layout()
plt.savefig(f'ap_rel_drop_vs_{noise_type}_noise.png', format='png', dpi=600)
plt.savefig(f'ap_rel_drop_vs_{noise_type}_noise.svg', format='svg', dpi=600)