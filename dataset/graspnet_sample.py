import os
from shutil import copyfile

result_root = 'graspnet_sample'
dataset_root = '/media/gpuadmin/rcao/dataset/graspnet'
seg_method = 'uois'
dataset_save_root = os.path.join('/media/gpuadmin/rcao/dataset', result_root)
width = 1280
height = 720

for camera in ['realsense', 'kinect']:
    for scene_idx in range(190):
        for anno_idx in [0, 128, 255]:
            print("camera:{}, scene index:{}, anno index:{}".format(camera, scene_idx, anno_idx))

            rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
            depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
            meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
            label_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
            
            virtual_rgb_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_rgb.png'.format(scene_idx, camera, anno_idx))
            virtual_depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
            virtual_mask_depth = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
            
            if scene_idx in range(100, 190):
                seg_path = os.path.join(dataset_root, '{}_mask/scene_{:04d}/{}/{:04d}.png'.format(seg_method, scene_idx, camera, anno_idx))
                seg_save_root = os.path.join(dataset_save_root, '{}_mask/scene_{:04d}/{}'.format(seg_method, scene_idx, camera))
                os.makedirs(seg_save_root, exist_ok=True)
                copyfile(seg_path, os.path.join(seg_save_root, '{:04d}.png'.format(anno_idx)))
            
            camera_pose_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera))
            camera_pose_save_root = os.path.join(dataset_save_root, 'scenes', 'scene_{:04d}/{}'.format(scene_idx, camera))
            os.makedirs(camera_pose_save_root, exist_ok=True)
            if not os.path.exists(os.path.join(camera_pose_save_root, 'camera_poses.npy')):
                copyfile(camera_pose_path, os.path.join(camera_pose_save_root, 'camera_poses.npy'))

            align_mat_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera))
            align_mat_save_root = os.path.join(dataset_save_root, 'scenes', 'scene_{:04d}/{}'.format(scene_idx, camera))
            os.makedirs(align_mat_save_root, exist_ok=True)
            if not os.path.exists(os.path.join(align_mat_save_root, 'cam0_wrt_table.npy')):
                copyfile(align_mat_path, os.path.join(align_mat_save_root, 'cam0_wrt_table.npy'))
                
            rgb_save_root = os.path.join(dataset_save_root, 'scenes', 'scene_{:04d}/{}/rgb'.format(scene_idx, camera))
            os.makedirs(rgb_save_root, exist_ok=True)
            copyfile(rgb_path, os.path.join(rgb_save_root, '{:04d}.png'.format(anno_idx)))

            depth_save_root = os.path.join(dataset_save_root, 'scenes', 'scene_{:04d}/{}/depth'.format(scene_idx, camera))
            os.makedirs(depth_save_root, exist_ok=True)
            copyfile(depth_path, os.path.join(depth_save_root, '{:04d}.png'.format(anno_idx)))
            
            meta_save_root = os.path.join(dataset_save_root, 'scenes', 'scene_{:04d}/{}/meta'.format(scene_idx, camera))
            os.makedirs(meta_save_root, exist_ok=True)
            copyfile(meta_path, os.path.join(meta_save_root, '{:04d}.mat'.format(anno_idx)))
            
            label_save_root = os.path.join(dataset_save_root, 'scenes', 'scene_{:04d}/{}/label'.format(scene_idx, camera))
            os.makedirs(label_save_root, exist_ok=True)
            copyfile(label_path, os.path.join(label_save_root, '{:04d}.png'.format(anno_idx)))

            if scene_idx in range(100, 130):
                virtual_rgb_save_root = os.path.join(dataset_save_root, 'virtual_scenes', 'scene_{:04d}/{}/rgb'.format(scene_idx, camera))
                os.makedirs(virtual_rgb_save_root, exist_ok=True)
                copyfile(virtual_rgb_path, os.path.join(virtual_rgb_save_root, '{:04d}.png'.format(anno_idx)))
                
                virtual_depth_save_root = os.path.join(dataset_save_root, 'virtual_scenes', 'scene_{:04d}/{}/depth'.format(scene_idx, camera))
                os.makedirs(virtual_depth_save_root, exist_ok=True)
                copyfile(virtual_depth_path, os.path.join(virtual_depth_save_root, '{:04d}.png'.format(anno_idx)))
                
                virtual_mask_save_root = os.path.join(dataset_save_root, 'virtual_scenes', 'scene_{:04d}/{}/label'.format(scene_idx, camera))
                os.makedirs(virtual_mask_save_root, exist_ok=True)
                copyfile(virtual_mask_depth, os.path.join(virtual_mask_save_root, '{:04d}.png'.format(anno_idx)))
            