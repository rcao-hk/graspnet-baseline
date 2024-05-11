import torch
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
checkpoint_root = '/media/gpuadmin/rcao/result/ignet'
load_path = os.path.join(checkpoint_root, 'ignet_v0.8.2_backup')
save_path = os.path.join(checkpoint_root, 'ignet_v0.8.2')
camera = 'realsense'
checkpoint_list = glob.glob(os.path.join(load_path, camera, '*.tar'))
print(checkpoint_list)
for checkpoint_path in checkpoint_list:
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    print(checkpoint.keys())
    os.makedirs(os.path.join(save_path, camera), exist_ok=True)
    save_name = os.path.join(save_path, camera, os.path.basename(checkpoint_path))
    torch.save(checkpoint['model_state_dict'], save_name)
    print(f'Save checkpoint to {save_name}')