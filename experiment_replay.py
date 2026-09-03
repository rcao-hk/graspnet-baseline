import numpy as np
import os
import pandas as pd

# method = 'ignet_v0.8.1'
# epoch_list = ['40', '45', '50', '55', '60']
# model_list = [method + '_' + i for i in epoch_list]
# model_list = [method]
# experiment_root = 'experiment'
# model_list = ['ignet_v0.8.2.26.2', 'ignet_v0.8.2.26.2_rep',]
# model_list =  ['gsnet_base', 'gsnet_multi', 'mmgnet_scene_24', 'mmgnet_scene_none', 'mmgnet_scene_concat',  'mmgnet_scene_direct', 'mmgnet_scene_gate', 'mmgnet_scene_add', 'mmgnet_scene_intermediate']
# model_list =  ['gsnet.clear', 'gsnet.5120', 'gsnet.2048', 'gsnet.1024', 'gsnet.512', 'gsnet_base.clear', 'gsnet_base.5120', 'gsnet_base.2048', 'gsnet_base.1024', 'gsnet_base.512']
# model_list =  ['mmgnet_baseline.clear', 'mmgnet_baseline.5120', 'mmgnet_baseline.512', 'mmgnet_scene.clear',]
# model_list =  ['scale_grasp.clear', 'scale_grasp.512',  'scale_grasp.512', 'mmgnet_scene.clear', 'mmgnet_scene.512']
# model_list = ['scale_grasp.clear', 'scale_grasp.512', 'scale_grasp.1024', 'scale_grasp.2048', 'scale_grasp.5120']
# model_list = ['mmgnet_scene.clear', 'mmgnet_scene_24', 'mmgnet_scene.none.s0', 'mmgnet_scene.cutout.s1']

# experiment_root = 'experiment'
# camera_type = 'realsense'
# noise_type = 'gaussian'
# model_list = ['gsnet_base.clear','gsnet_base.0.002', 'gsnet_base.0.005', 'gsnet_base.0.008','gsnet_base.0.01',  'gsnet.clear', 'gsnet.0.002', 'gsnet.0.005', 'gsnet.0.008', 'gsnet.0.01', 'scale_grasp.clear', 'scale_grasp.0.002', 'scale_grasp.0.005', 'scale_grasp.0.008', 'scale_grasp.0.01', 'mmgnet_baseline.clear','mmgnet_baseline.0.002', 'mmgnet_baseline.0.005', 'mmgnet_baseline.0.008','mmgnet_baseline.0.01',  'mmgnet_scene_intermediate.clear', 'mmgnet_scene_intermediate.0.002', 'mmgnet_scene_intermediate.0.005', 'mmgnet_scene_intermediate.0.008', 'mmgnet_scene_intermediate.0.01',
# 'mmgnet_scene.clear', 'mmgnet_scene.0.002', 'mmgnet_scene.0.005', 'mmgnet_scene.0.008', 'mmgnet_scene.0.01']

# noise_type = 'smooth'
# model_list = [
#             'gsnet_base.clear','gsnet_base.s5', 'gsnet_base.s15', 'gsnet_base.s29', 
#             'gsnet.clear', 'gsnet.s5', 'gsnet.s15', 'gsnet.s29', 
#             'scale_grasp.clear','scale_grasp.s5', 'scale_grasp.s15', 'scale_grasp.s29', 'mmgnet_baseline.clear','mmgnet_baseline.s5', 'mmgnet_baseline.s15', 'mmgnet_baseline.s29', 
#             'mmgnet_scene_intermediate.clear', 'mmgnet_scene_intermediate.s5', 'mmgnet_scene_intermediate.s15', 'mmgnet_scene_intermediate.s29',
#             'mmgnet_scene.clear', 'mmgnet_scene.s5', 'mmgnet_scene.s15', 'mmgnet_scene.s29'
#             ]
# experiment_root = 'experiment'
# camera_type = 'realsense'

# noise_type = 'control_dropout'
# model_list = [
#             # 'gsnet_base.clear','gsnet_base.s5', 'gsnet_base.s15', 'gsnet_base.s29', 
#             # 'gsnet.clear', 'gsnet.s5', 'gsnet.s15', 'gsnet.s29', 
#               'scale_grasp.clear','scale_grasp.dr0.2', 'scale_grasp.dr0.4', 'scale_grasp.dr0.6', 'scale_grasp.dr0.8', 'scale_grasp.dr1.0', 
#               'mmgnet_baseline.clear','mmgnet_baseline.dr0.2', 'mmgnet_baseline.dr0.4', 'mmgnet_baseline.dr0.6', 'mmgnet_baseline.dr0.8', 'mmgnet_baseline.dr1.0', 
#               'mmgnet_scene.clear', 'mmgnet_scene.dr0.2', 'mmgnet_scene.dr0.4', 'mmgnet_scene.dr0.6','mmgnet_scene.dr0.8', 'mmgnet_scene.dr1.0']
# experiment_root = 'experiment'
# camera_type = 'realsense'


# experiment_root = 'experiment'
# camera_type = 'realsense'
# noise_type = 'sparsity'
# model_list = [
#             'gsnet_base.clear','gsnet_base.5120', 'gsnet_base.2048', 'gsnet_base.1024', 'gsnet_base.512',
#             'gsnet.clear', 'gsnet.5120', 'gsnet.2048', 'gsnet.1024', 'gsnet.512',
#             'scale_grasp.clear', 'scale_grasp.5120', 'scale_grasp.2048', 'scale_grasp.1024', 'scale_grasp.512',
#             'mmgnet_baseline.clear', 'mmgnet_baseline.5120', 'mmgnet_baseline.2048', 'mmgnet_baseline.1024', 'mmgnet_baseline.512',
#             # 'mmgnet_scene.clear', 'mmgnet_scene.5120', 'mmgnet_scene.2048', 'mmgnet_scene.1024', 'mmgnet_scene.512'
#             'mmgnet_scene_pt_early.5120', 'mmgnet_scene_pt_early.5120', 'mmgnet_scene_pt_early.2048', 'mmgnet_scene_pt_early.1024', 'mmgnet_scene_pt_early.512'
#             ]


# noise_type = 'dropout'
# model_list = [
#               'gsnet_base.clear','gsnet_base.dr0.1','gsnet_base.dr0.2', 'gsnet_base.dr0.4', 'gsnet_base.dr0.6',
#               'gsnet.clear','gsnet.dr0.1','gsnet.dr0.2', 'gsnet.dr0.4', 'gsnet.dr0.6',
#               'scale_grasp.clear','scale_grasp.dr0.1', 'scale_grasp.dr0.2', 'scale_grasp.dr0.4', 'scale_grasp.dr0.6', 
#               'mmgnet_baseline.clear','mmgnet_baseline.dr0.1','mmgnet_baseline.dr0.2', 'mmgnet_baseline.dr0.4', 'mmgnet_baseline.dr0.6',
#               'mmgnet_scene_intermediate.clear','mmgnet_scene_intermediate.dr0.1', 'mmgnet_scene_intermediate.dr0.2', 'mmgnet_scene_intermediate.dr0.4', 'mmgnet_scene_intermediate.dr0.6']
# experiment_root = 'experiment'
# camera_type = 'realsense'

# experiment_root = 'experiment'
# camera_type = 'realsense'
# model_list = ['mmgnet_scene_intermediate.clear', 'mmgnet_scene_intermediate.dr0.1', 'mmgnet_scene_intermediate.dr0.2', 'mmgnet_scene_intermediate.dr0.4', 'mmgnet_scene_intermediate.dr0.6']
# noise_type = None

# experiment_root = 'experiment'
# camera_type = 'kinect'
# model_list = ['mmgnet_scene_kinect_new_bp', 'mmgnet_scene_20', 'mmgnet_scene_10', 'mmgnet_scene_pt_early', 'mmgnet_scene_intermediate_bp', 'mmgnet_scene_intermediate_angle_aug', 'mmgnet_scene_intermediate_resnext50', 'mmgnet_scene_intermediate_v0.005']
# noise_type = None

# experiment_root = 'experiment'
# camera_type = 'realsense'
# model_list = ['gsnet_multi','gsnet_multi', 'gsnet_multi_intermediate']
# noise_type = None

# experiment_root = '/data2/robotarm/result/grasp/mmgnet/experiment'
# camera_type = 'realsense'
# model_list = ['mmgnet_scene_intermediate_bp', 'mmgnet_scene_intermediate_angle_aug_20']
# noise_type = None

experiment_root = '/data2/robotarm/result/grasp/mmgnet/experiment'
camera_type = 'realsense'
model_list = ['mmgnet_baseline.clear','mmgnet_scene_intermediate_bp', 'mmgnet_scene_intermediate.none.s0']
noise_type = None

column = ['AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP_mean']
split_data = []
epoch_data = []
for model in model_list:
    root = os.path.join(experiment_root, model)
    data = []
    split_ap = []
    for split in ['seen', 'similar', 'novel']:
    # for split in ['seen']:
        res = np.load(os.path.join(root, 'ap_test_{}_{}.npy'.format(split, camera_type)))

        ap_top50 = np.mean(res[:, :, :50, :])
        print('\nEvaluation Result of Top 50 Grasps:\n----------\n{}, AP {}={:6f}'.format(camera_type, split, ap_top50))

        ap_top50_0dot2 = np.mean(res[..., :50, 0])
        print('----------\n{}, AP0.2 {}={:6f}'.format(camera_type, split, ap_top50_0dot2))

        ap_top50_0dot4 = np.mean(res[..., :50, 1])
        print('----------\n{}, AP0.4 {}={:6f}'.format(camera_type, split, ap_top50_0dot4))

        ap_top50_0dot6 = np.mean(res[..., :50, 2,])
        print('----------\n{}, AP0.6 {}={:6f}'.format(camera_type, split, ap_top50_0dot6))

        ap_top50_0dot8 = np.mean(res[..., :50, 3])
        print('----------\n{}, AP0.8 {}={:6f}'.format(camera_type, split, ap_top50_0dot8))

        split_ap.append(ap_top50)
        data.extend([ap_top50, ap_top50_0dot8, ap_top50_0dot4])

    data.extend([np.mean(split_ap)])
    epoch_data.append(data)
    split_data.append(split_ap)
    
    # split_cf_rate = []
    # for split in ['seen', 'similar', 'novel']:
    #     res = np.load(os.path.join(root, 'ap_test_{}_{}_new.npy'.format(split, camera_type)))

    #     # print(res.shape)
    #     ap_top50 = np.mean(res[:, :, :50, :, 0])
    #     print('\nEvaluation Result of Top 50 Grasps:\n----------\n{}, AP {}={:6f}'.format(camera_type, split, ap_top50))

    #     ap_top50_0dot2 = np.mean(res[..., :50, 0, 0])
    #     print('----------\n{}, AP0.2 {}={:6f}'.format(camera_type, split, ap_top50_0dot2))

    #     ap_top50_0dot4 = np.mean(res[..., :50, 1, 0])
    #     print('----------\n{}, AP0.4 {}={:6f}'.format(camera_type, split, ap_top50_0dot4))

    #     ap_top50_0dot6 = np.mean(res[..., :50, 2, 0])
    #     print('----------\n{}, AP0.6 {}={:6f}'.format(camera_type, split, ap_top50_0dot6))

    #     ap_top50_0dot8 = np.mean(res[..., :50, 3, 0])
    #     print('----------\n{}, AP0.8 {}={:6f}'.format(camera_type, split, ap_top50_0dot8))

    #     collision_free_rate = np.mean(res[..., :50, :, 1])
    #     # ap_top50_cf = np.mean(res[:, :, :50, :, 1])
    #     # print('----------\n{}, AP cf {}={:6f}'.format(camera_type, split, ap_top50_cf))

    #     # ap_top50_0dot2_cf = np.mean(res[..., :50, 0, 1])
    #     # print('----------\n{}, AP0.2 cf {}={:6f}'.format(camera_type, split, ap_top50_0dot2_cf))
        
    #     # ap_top50_0dot4_cf = np.mean(res[..., :50, 1, 1])
    #     # print('----------\n{}, AP0.4 cf {}={:6f}'.format(camera_type, split, ap_top50_0dot4_cf))
        
    #     # ap_top50_0dot6_cf = np.mean(res[..., :50, 2, 1])
    #     # print('----------\n{}, AP0.6 cf {}={:6f}'.format(camera_type, split, ap_top50_0dot6_cf))
        
    #     # ap_top50_0dot8_cf = np.mean(res[..., :50, 3, 1])
    #     # print('----------\n{}, AP0.8 cf {}={:6f}'.format(camera_type, split, ap_top50_0dot8_cf))
        
    #     split_ap.append(ap_top50)
    #     split_cf_rate.append(collision_free_rate)
    #     data.extend([ap_top50, ap_top50_0dot8, ap_top50_0dot4])

    # data.extend([np.mean(split_ap)])
    # data.extend(split_cf_rate)
    # # data.extend([np.mean(split_ap_cf)])
    # epoch_data.append(data)
    

if noise_type is not None:
    save_column = ['AP_seen', 'AP_similar', 'AP_novel']
    data_table = pd.DataFrame(columns=save_column, index=model_list, data=split_data)
    data_table.to_csv(f'{noise_type}_noise.csv')
    
for model_name, data in zip(model_list, epoch_data):
    print(model_name, data)
    print("\t")

# print(split_cf_rate)