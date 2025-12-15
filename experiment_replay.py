import numpy as np
import os
import pandas as pd

# method = 'ignet_v0.8.1'
# epoch_list = ['40', '45', '50', '55', '60']
# model_list = [method + '_' + i for i in epoch_list]
# model_list = [method]
# experiment_root = 'experiment'
model_list = ['ignet_v0.8.2.26.2', 'ignet_v0.8.2.26.2_rep', 'ignet_v0.8.2.26.2_grid_4x4','ignet_v0.8.2.26.2_grid_2x2_2048']
experiment_root = 'experiment'

# model_list = ['ignet_v0.6.2.dr0.2', 'ignet_v0.6.2.dr0.4', 'ignet_v0.6.2.dr0.6', 'ignet_v0.6.2.dr0.8', 'ignet_v0.6.2.dr1.0',
#               'ignet_v0.8.2.dr0.2', 'ignet_v0.8.2.dr0.4', 'ignet_v0.8.2.dr0.6', 'ignet_v0.8.2.dr0.8', 'ignet_v0.8.2.dr1.0',]
# model_list = ['gsnet_virtual_raw_ours_gt', 'gsnet_virtual_raw_ours_restored']

column = ['AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP_mean']
camera_type = 'realsense'
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
    
# data_table = pd.DataFrame(columns=column, index=model_list, data=epoch_data)
# data_table.to_csv('epoch_experiment.csv')
for model_name, data in zip(model_list, epoch_data):
    print(model_name, data)
    print("\t")

# print(split_cf_rate)