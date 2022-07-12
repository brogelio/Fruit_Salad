import pyrealsense2 as rs
import numpy as np
import logging
logger = logging.getLogger('rscamera.py')



class RealSenseCamera(object):
    """
    Class for wrapping pyrealsense2 functionalities
    """
    def __init__(self, **kwargs):
        self.color = kwargs['enable_color']
        self.depth = kwargs['enable_depth']
        self.emitter = 1.0 if kwargs['enable_emitter'] else 0.0
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.camera_name = str(self.device.get_info(rs.camera_info.name))
        self.usb_version = str(self.device.get_info(rs.camera_info.usb_type_descriptor))
        self.frame_count = 0
        logger.info(f'Camera Initialized: {self.camera_name}')
        logger.info(f'USB connection detected: USB {self.usb_version}')
        self.running = True

        if self.color:
            self.config.enable_stream(rs.stream.color, kwargs['width'], kwargs['height'], rs.format.bgr8, kwargs['fps'])
        if self.depth:
            self.depth_filters = []
            self.config.enable_stream(rs.stream.depth, kwargs['width']//2, kwargs['height']//2, rs.format.z16, kwargs['fps'])


    def start(self):
        self.profile = self.pipeline.start(self.config)
        color_sensor = self.profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, True)

        depth_sensor = self.profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.emitter_enabled, self.emitter)
        emitter = depth_sensor.get_option(rs.option.emitter_enabled)
        emitter_status = 'ON' if emitter  else 'OFF'
        logger.info(f'IR Emitter is {emitter_status}')
        
        logger.info('Camera started.')
        if self.depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            self.align = rs.align(rs.stream.color)


    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        self.frame_count += 1
        logger.debug(f'Retreived frame {self.frame_count}.')

        if self.depth:
            self.aligned_frames = self.align.process(frames)
            depth_frame = self.aligned_frames.get_depth_frame()
            for filter_ in self.depth_filters:
                depth_frame = filter_.process(depth_frame)
            color_frame = self.aligned_frames.get_color_frame()
            return depth_frame, color_frame

        if self.color:
            color_frame = frames.get_color_frame()
            return color_frame


    def stop(self):
        if self.running:
            self.pipeline.stop()
            self.running = False
            logger.info('Camera stopped.')



def compute_depth_map(depth_image, depth_range = [0, 4000]):
    depth_scaling = (depth_range[1] - depth_range[0])/255
    depth_image = np.where(depth_image >= depth_range[1], depth_range[1], depth_image)
    depth_map = 255 - (depth_image-depth_range[0])/depth_scaling
    depth_map = depth_map.astype(np.uint8)
    return depth_map



if __name__ == '__main__':

    import numpy as np
    import cv2
    import time

    color = True
    depth = True
    emitter = True

    cam = RealSenseCamera(width=1280, height=720, fps=30, enable_color=color, enable_depth=depth, enable_emitter=emitter)

    if depth:
        decimation_filter = rs.decimation_filter(magnitude=2.)
        threshold_filter = rs.threshold_filter(min_dist=0.30, max_dist=1.5)
        depth_to_disparity = rs.disparity_transform(transform_to_disparity=True)
        spatial_filter = rs.spatial_filter(smooth_alpha=0.50, smooth_delta=20., magnitude=2., hole_fill=0)
        temporal_filter = rs.temporal_filter(smooth_alpha=0.80, smooth_delta=20., persistence_control=4)
        hole_filter = rs.hole_filling_filter()
        disparity_to_depth = rs.disparity_transform(transform_to_disparity=False)
        cam.depth_filters.extend([decimation_filter, threshold_filter, depth_to_disparity, spatial_filter, temporal_filter, disparity_to_depth])

    cam.start()
    if color:
        cv2.namedWindow('RGB Frame', cv2.WINDOW_AUTOSIZE)
    if depth:
        cv2.namedWindow('Depth Map', cv2.WINDOW_AUTOSIZE)

    try:
        prevTime = 0
        while True:

            if color and depth:
                depth_frame, color_frame = cam.get_frames()
                if not depth_frame or not color_frame:
                    print('Frame dropped')
                    break
            elif color:
                color_frame = cam.get_frames()

            # Flip image horizontal coordinates
            if color:
                color_image = cv2.flip(np.asanyarray(color_frame.get_data()), 1)
            if depth:
                depth_image = cv2.flip(np.asanyarray(depth_frame.get_data()), 1)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(color_image, f'{fps:.2f} fps', (0, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
            print(f'{fps:.2f} fps')

            if color:
                cv2.imshow('RGB Frame', color_image)

            # Compute a depth map
            if depth:
                depth_map = compute_depth_map(depth_image, depth_range = [525, 910])
                cv2.imshow('Depth Map', depth_map)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:         # Press esc or 'q' to close the image window
                break

    except:
        pass

    finally:
        cv2.destroyAllWindows()
        cam.stop()