import os
import numpy as np
import argparse
from PIL import Image
import scipy.io as scio
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import transform_points, CameraInfo, \
    create_point_cloud_from_depth_image, parse_posevector
import cv2
from tqdm import tqdm
import open3d as o3d
from bop_toolkit_lib import renderer

parser = argparse.ArgumentParser()
parser.add_argument('--camera', default='realsense', help='camera to use [kinect | realsense]')
parser.add_argument('--dataset_root', default='/data/rcao/dataset/graspnet', help='where dataset is')
cfgs = parser.parse_args()


def generate_scene_model(dataset_root, scene_name, anno_idx, align=False, camera='realsense'):
    if align:
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[anno_idx]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        camera_pose = np.matmul(align_mat, camera_pose)
    scene_reader = xmlReader(
        os.path.join(dataset_root, 'scenes', scene_name, camera, 'annotations', '%04d.xml' % anno_idx))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    mat_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)
    for obj_idx, pose in zip(obj_list, mat_list):
        plyfile = os.path.join(dataset_root, 'models', '%03d' % obj_idx, 'nontextured.ply')
        # model = o3d.io.read_point_cloud(plyfile)
        # points = np.array(model.points)
        if align:
            pose = np.dot(camera_pose, pose)
        # points = transform_points(points, pose)
        # model.points = o3d.utility.Vector3dVector(points)
        # model = model.voxel_down_sample(0.0005)
        # model_list.append(model)
        pose_list.append(pose)

    return obj_list, pose_list


width, height = 1280, 720
camera = cfgs.camera
dataset_root = cfgs.dataset_root

ren_width, ren_height = 3 * width, 3 * height
ren_cx_offset, ren_cy_offset = width, height
ren = renderer.create_renderer(ren_width, ren_height, 'vispy', mode="depth")

obj_list = range(88)
# Add object models.
for obj_idx in tqdm(obj_list):
    ren.add_object(obj_idx, os.path.join(dataset_root, 'models', '%03d' % obj_idx, 'nontextured.ply'))

visib_info_save_root = os.path.join(dataset_root, 'visib_info')
os.makedirs(visib_info_save_root, exist_ok=True)


for scene_idx in range(100, 130):
    for anno_idx in tqdm(range(256)):
        rgb_path = os.path.join(dataset_root,
                                'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        depth_path = os.path.join(dataset_root,
                                  'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
        mask_path = os.path.join(dataset_root,
                                 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        meta_path = os.path.join(dataset_root,
                                 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))

        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # seg = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.bool)

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))

        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2],
        #                          factor_depth)
        # cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
        # depth_mask = (depth > 0)

        # camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_idx).zfill(4),
        #                                     camera, 'camera_poses.npy'))
        # camera_pose = camera_poses[anno_idx]
        #
        # align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_idx).zfill(4),
        #                                  camera, 'cam0_wrt_table.npy'))
        # trans = np.dot(align_mat, camera_pose)
        # workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        # mask = (depth_mask & workspace_mask)

        # mask = depth_mask
        # cloud_masked = cloud[mask]
        # color_masked = color[mask]

        # scene = o3d.geometry.PointCloud()
        # scene.points = o3d.utility.Vector3dVector(cloud_masked)
        # scene.colors = o3d.utility.Vector3dVector(color_masked)

        obj_list, pose_list = generate_scene_model(dataset_root, 'scene_' + str(scene_idx).zfill(4), anno_idx, align=False, camera=camera)

        # pading_seg = cv2.copyMakeBorder(seg, int(height * 0.5), int(height * 0.5), int(width * 0.5), int(width * 0.5),
        #                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

        info_dict = {}
        for i, (obj_idx, obj_pose) in enumerate(zip(obj_list, pose_list)):
            # model_pc = np.asarray(model_pcd.points)
            # obj_seg_mask = np.where(pading_seg == obj_idx + 1)
            obj_seg_mask = seg == obj_idx + 1
            px_count_visib = obj_seg_mask.astype(np.uint8).sum()

            # obj_seg_mask_np = obj_seg_mask.astype(np.uint8) * 255
            # cv2.imwrite(os.path.join('seg_{:04d}.png'.format(anno_idx)), obj_seg_mask_np)

            # depth_gt = ren.render_object(obj_idx, obj_pose[:3, :3], obj_pose[:3, 3], intrinsics[0][0], intrinsics[1][1],
            #                              intrinsics[0][2], intrinsics[1][2])["depth"]
            depth_gt_large = ren.render_object(
                obj_idx,
                obj_pose[:3, :3],
                obj_pose[:3, 3],
                intrinsics[0][0],
                intrinsics[1][1],
                intrinsics[0][2] + ren_cx_offset,
                intrinsics[1][2] + ren_cy_offset,
            )["depth"]
            obj_mask_gt = depth_gt_large > 0
            px_count_all = np.sum(obj_mask_gt)
            # obj_proj_mask_np = obj_mask_gt.astype(np.uint8) * 255
            # cv2.imwrite(os.path.join('proj_{:04d}.png'.format(anno_idx)), obj_proj_mask_np)
            visib_fract = px_count_visib / float(px_count_all)
            # print(visib_fract)
            # o3d.visualization.draw_geometries([scene, model_pcd])
            info_dict.update({str(obj_idx):{"px_count_all": int(px_count_all),
                                            "px_count_visib": int(px_count_visib),
                                            "visib_fract": float(visib_fract)}})

        visib_info_save_path = os.path.join(visib_info_save_root, 'scene_{:04}'.format(scene_idx), camera)
        os.makedirs(visib_info_save_path, exist_ok=True)
        scio.savemat(os.path.join(visib_info_save_path, '{:04}.mat'.format(anno_idx)), info_dict)