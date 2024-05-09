import numpy as np
import h5py
import os

def convert_npz_to_hdf5(npz_path, hdf5_path):
    # 加载 NPZ 文件
    data = np.load(npz_path)

    # 创建一个新的 HDF5 文件
    with h5py.File(hdf5_path, 'w') as hdf:
        # 遍历所有在 NPZ 文件中的数组
        for key in data.files:
            # 将每个数组保存到 HDF5 文件中
            hdf.create_dataset(key, data=data[key])
    
    print(f"Conversion complete. File saved as {hdf5_path}")

# # 使用示例
# file_root = '/media/gpuadmin/rcao/dataset/graspnet/grasp_label_simplified'

# npz_file_path = 'path/to/your/datafile.npz'  # 替换为你的 .npz 文件路径
# hdf5_file_path = 'path/to/your/datafile.hdf5'  # 替换为你想要保存的 .hdf5 文件路径

# for i in range(0, 88):
#     npz_file_path = os.path.join(file_root, '{:03d}_labels.npz'.format(i))
#     hdf5_file_path = os.path.join(file_root, '{:03d}_labels.hdf5'.format(i))
#     convert_npz_to_hdf5(npz_file_path, hdf5_file_path)

file_root = '/media/gpuadmin/rcao/dataset/graspnet/collision_label'
save_root = '/media/gpuadmin/rcao/dataset/graspnet/collision_label_hdf5'

for i in range(0, 190):
    npz_file_path = os.path.join(file_root, 'scene_{:04d}'.format(i), 'collision_labels.npz')
    save_path = os.path.join(save_root, 'scene_{:04d}'.format(i))
    os.makedirs(save_path, exist_ok=True)
    hdf5_file_path = os.path.join(save_path, 'collision_labels.hdf5')
    convert_npz_to_hdf5(npz_file_path, hdf5_file_path)
