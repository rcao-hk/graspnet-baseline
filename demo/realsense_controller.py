import time
import cv2
import open3d as o3d
import numpy as np
import pyrealsense2 as rs


class RealSenseController:
    def __init__(self):
        self.intrinsic_mat_ = np.zeros((3, 3))
        self.distortion_coeff_ = np.zeros(5)
        self.pipe_ = None
        
    def start_streaming_stereo(self):
        try:
            pipe = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(pipe)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            profile = pipe.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: " , depth_scale)

            clipping_distance_in_meters = 1 # [m]
            clipping_distance = clipping_distance_in_meters / depth_scale

            align_to = rs.stream.color
            self.align_ = rs.align(align_to)

            while True:
                ## Realsense get data
                start = time.time()
                frames = pipe.wait_for_frames()
                aligned_frames = self.align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                ## Visualization
                vis_images = [depth_colormap, color_image]

                # vis_images[]
                cv2.namedWindow('VisAlgoDPT', cv2.WINDOW_NORMAL)
                cv2.imshow('VisAlgoDPT', np.hstack(vis_images))
                

                ## Logging
                print(f"\n vision algo frame rate: {1.0/(time.time() - start)}\n")  

                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    print("\n===== stop pipeline =====")  
                    cv2.destroyAllWindows()
                    break
                if key & 0xFF == ord('s'):
                    print("\n===== save data =====")  
                    cv2.imwrite('data/tmp/color.png', color_image)
                    # cv2.imwrite('data/tmp/depth_mono.png', depth_mono)
                    # cv2.imwrite('data/tmp/depth_filtered.png', th_edge)
                    continue
        finally:
            pipe.stop()

    def stop_streaming(self):
        self.pipe_.stop()

    def get_intrinsics(self):
        print(f'===== get intrinsics =====')
        print(f'intrinsic_mat_: \n{self.intrinsic_mat_}')
        print(f'distortion_coeff_: \n{self.distortion_coeff_}')
        return self.intrinsic_mat_, self.distortion_coeff_

    def config_streaming_color(self, w=1280, h=720, fps=30): # d405 not supported, for it only support stereo mode
        self.pipe_ = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipe_)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Start streaming
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        profile = self.pipe_.start(config)

        # Get intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsic_mat_ = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.distortion_coeff_ = np.array(intr.coeffs)

    def get_stereo_frame(self):
        frames = self.pipe_.wait_for_frames()

        aligned_frames = self.align_.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()


        return color_frame, depth_frame

    def convert_to_pointcloud(self, color_frame, depth_frame):
        color_image = np.asanyarray(color_frame.get_data())

        rgb = color_image[...,::-1]

        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        vertices = np.asanyarray(points.get_vertices(dims=2)) 
        w = depth_frame.get_width()
        h = depth_frame.get_height()
        image_points = np.reshape(vertices, (h, w, 3))

        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        vertices = np.asanyarray(points.get_vertices(dims=2))
        w = depth_frame.get_width()
        image_Points = np.reshape(vertices , (-1,w,3))

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(image_Points.reshape(-1,3))
        point_cloud.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3) / 255.0)
        return point_cloud

    def config_stereo_stream(self, w=1280, h=720, fps=15):
        self.pipe_ = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipe_)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Start streaming
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

        profile = self.pipe_.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # Get intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsic_mat_ = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.distortion_coeff_ = np.array(intr.coeffs)

        self.align_ = rs.align(rs.stream.color)

    def fetch_one_frame(self):
        frames = self.pipe_.wait_for_frames()

        aligned_frames = self.align_.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def start_streaming(self, data_io=None):
        idx = 0
        try:
            while True:
                start = time.time()

                ## Realsense get data
                frames = self.pipe_.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())

                # vis_images[]
                cv2.namedWindow('rs_stream', cv2.WINDOW_NORMAL)
                cv2.imshow('rs_stream', color_image)

                ## Logging
                print(f"\n rs stream frame rate: {1.0/(time.time() - start)}\n")  

                key = cv2.waitKey(1)        
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    print("\n===== stop pipeline =====")  
                    cv2.destroyAllWindows()
                    break
                if key & 0xFF == ord('s'):
                    print("\n===== save data =====")  
                    if data_io is not None:
                        data_io.save_img(color_image, idx)
                        idx += 1
                    else:
                        cv2.imwrite('data/tmp/color.png', color_image)
                    continue
        finally:
            self.stop_streaming()

    def get_color_frame(self):
        frames = self.pipe_.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
    
    def disconnect(self):
        if self.pipe_ is not None:
            self.pipe_.stop()
            self.pipe_ = None
        
if __name__ ==  "__main__":
    np.set_printoptions(precision=7, suppress=True)
    rs_ctrl = RealSenseController()
    # rs_ctrl.config_stereo_stream()
    rs_ctrl.config_streaming_color(1920, 1080)
    rs_ctrl.get_intrinsics()
    from data_io import DataRecorder
    data_recorder = DataRecorder('data/calib_data')
    rs_ctrl.start_streaming(data_recorder)