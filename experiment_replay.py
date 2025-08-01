import numpy as np
import os
import pandas as pd

# method = 'ignet_v0.8.1'
# epoch_list = ['40', '45', '50', '55', '60']
# model_list = [method + '_' + i for i in epoch_list]
# model_list = [method]
# experiment_root = 'experiment'
experiment_root = '/media/gpuadmin/rcao/result/ignet/experiment'

# model_list = ['ignet_v0.6.2.dr0.2', 'ignet_v0.6.2.dr0.4', 'ignet_v0.6.2.dr0.6', 'ignet_v0.6.2.dr0.8', 'ignet_v0.6.2.dr1.0',
#               'ignet_v0.8.2.dr0.2', 'ignet_v0.8.2.dr0.4', 'ignet_v0.8.2.dr0.6', 'ignet_v0.8.2.dr0.8', 'ignet_v0.8.2.dr1.0',]
model_list = ['gsnet', 'gsnet_base', 'gsnet_multi']

column = ['AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP_mean']
camera_type = 'realsense'
epoch_data = []
for model in model_list:
    root = os.path.join(experiment_root, model)
    data = []
    split_ap = []
    for split in ['seen', 'similar', 'novel']:
        res = np.load(os.path.join(root, 'ap_test_{}_{}.npy'.format(split, camera_type)))

        ap_top50 = np.mean(res[:, :, :50, :])
        print('\nEvaluation Result of Top 50 Grasps:\n----------\n{}, AP {}={:6f}'.format(camera_type, split, ap_top50))

        ap_top50_0dot2 = np.mean(res[..., :50, 0])
        print('----------\n{}, AP0.2 {}={:6f}'.format(camera_type, split, ap_top50_0dot2))

        ap_top50_0dot4 = np.mean(res[..., :50, 1])
        print('----------\n{}, AP0.4 {}={:6f}'.format(camera_type, split, ap_top50_0dot4))

        ap_top50_0dot6 = np.mean(res[..., :50, 2])
        print('----------\n{}, AP0.6 {}={:6f}'.format(camera_type, split, ap_top50_0dot6))

        ap_top50_0dot8 = np.mean(res[..., :50, 3])
        print('----------\n{}, AP0.8 {}={:6f}'.format(camera_type, split, ap_top50_0dot8))

        split_ap.append(ap_top50)
        data.extend([ap_top50, ap_top50_0dot8, ap_top50_0dot4])

    data.extend([np.mean(split_ap)])
    epoch_data.append(data)
    
data_table = pd.DataFrame(columns=column, index=model_list, data=epoch_data)
# data_table.to_csv('epoch_experiment.csv')
for model_name, data in zip(model_list, epoch_data):
    print(model_name, data)
    print("\t")
