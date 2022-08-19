import pyautogui
import cv2
import numpy as np
pyautogui.FAILSAFE = False

class MouseControl(object):
    """
    Class for using MediaPipe hand keypoints for simulating mouse functionalities
    """
    def __init__(self, monitor_resolution=(1920,1080), camera_resolution=(1280,720), ptr_finger="middle"):
        self.monitor_resolution = monitor_resolution
        self.camera_resolution = camera_resolution
        self.working_area = (int(camera_resolution[0]*0.7), int(camera_resolution[1]*0.7))
        if ptr_finger == 'middle':
            self.ptr_lm_index = 12
        if ptr_finger == 'palm':
            self.ptr_lm_index = 9
        self.thumb_lm_index = 4

    def init(self):
        self.keypoints = None
        self.press_state = None
        self.position = None
        self.velocity = None
        self.position_thumb = None
        self.velocity_thumb = None
        self.acceleration = None
        self.mouse_speed = 0.0
        self.mouse_accel = 0.0
        self.left_pressed = False
        self.dragging = False
        self.max_speed = 0
        self.max_accel = 0

    def update(self, keypoints, press_state=None):
        if keypoints is not None:
            self.keypoints = keypoints
            prev_state = self.press_state
            self.press_state = press_state
            if prev_state is not None:
                if self.left_pressed is False and self.press_state is True:
                    # pyautogui.mouseDown()
                    self.left_pressed = True

                if self.left_pressed is True and self.press_state is False:
                    # pyautogui.mouseUp()
                    self.left_pressed = False
                    self.dragging = False

            # For chosen mouse keypoint
            prev_position = self.position
            self.position = self.get_mouse_keypoint()
            prev_velocity = self.velocity
            if prev_position is not None:
                self.velocity = self.position[0]-prev_position[0], self.position[1]-prev_position[1]
                self.mouse_speed = (self.velocity[0]**2 + self.velocity[1]**2) ** 0.5
                if self.mouse_speed > self.max_speed:
                    self.max_speed = self.mouse_speed

            if prev_velocity is not None:
                self.acceleration = self.velocity[0]-prev_velocity[0], self.velocity[1]-prev_velocity[1]
                self.mouse_accel = (self.acceleration[0]**2 + self.acceleration[1]**2) ** 0.5
                if self.mouse_accel > self.max_accel:
                    self.max_accel = self.mouse_accel

            # For thumb
            prev_position_thumb = self.position_thumb
            self.position_thumb = self.get_thumb_keypoint()
            if prev_position_thumb is not None:
                self.velocity_thumb = self.position_thumb[0]-prev_position_thumb[0], self.position_thumb[1]-prev_position_thumb[1]


    def scroll_drag(self, scroll_clicks=50, drag_sensitivity=3, trigger_sensitivity=6, thumbs_up_sensitivity=-7):
        if self.position is None or self.velocity is None:
            return
        if self.left_pressed:
            monitor_delta = self.velocity[0]
            drag_thresh = drag_sensitivity if self.dragging else trigger_sensitivity
            thumb_deltay = self.velocity_thumb[1]*self.monitor_resolution[1]
            # print('drag_x:', int(monitor_delta*self.monitor_resolution[0]), "drag_thresh:", drag_thresh, "self.dragging:", self.dragging)
            # print("thumb_deltay:", thumb_deltay)
            if abs(int(monitor_delta*self.monitor_resolution[0])) > drag_thresh and thumb_deltay > thumbs_up_sensitivity:
                self.dragging = True
                if int(monitor_delta*self.monitor_resolution[0]) > 0:
                    pyautogui.scroll(-scroll_clicks, _pause=False)
                    print('scroll down: prev image')
                else:
                    pyautogui.scroll(scroll_clicks, _pause=False)
                    print('scroll up: next image')
        else:
            self.dragging = False

    def update2(self, keypoints, press_state=None, fing_up = 640):
        if keypoints is not None:
            self.keypoints = keypoints
            self.fing_up = fing_up
            prev_state = self.press_state
            self.press_state = press_state
            if prev_state is not None:
                if self.left_pressed is False and self.press_state is True:
                    pyautogui.mouseDown()
                    self.left_pressed = True

                if self.left_pressed is True and self.press_state is False:
                    pyautogui.mouseUp()
                    self.left_pressed = False

            prev_position = self.position
            self.position = self.get_mouse_keypoint2(fing_up)

            prev_velocity = self.velocity 
            if prev_position is not None:
                self.velocity = self.position[0]-prev_position[0], self.position[1]-prev_position[1]
                self.mouse_speed = (self.velocity[0]**2 + self.velocity[1]**2) ** 0.5
                if self.mouse_speed > self.max_speed:
                    self.max_speed = self.mouse_speed

            if prev_velocity is not None:
                self.acceleration = self.velocity[0]-prev_velocity[0], self.velocity[1]-prev_velocity[1]
                self.mouse_accel = (self.acceleration[0]**2 + self.acceleration[1]**2) ** 0.5
                if self.mouse_accel > self.max_accel:
                    self.max_accel = self.mouse_accel

    def get_thumb_keypoint(self):
        x, y, z = self.keypoints[self.thumb_lm_index]
        return x, y

    def get_mouse_keypoint(self):
        x, y, z = self.keypoints[self.ptr_lm_index]
        return x, y

    def get_mouse_keypoint2(self, fing_up):
        x, y, z = self.keypoints[self.ptr_lm_index]
        return fing_up, x


    def move_mouse(self, monitor_position):
        new_pos_x, new_pos_y = monitor_position
        if new_pos_x < 0: new_pos_x = 0
        if new_pos_x >= self.monitor_resolution[0]: new_pos_x = self.monitor_resolution[0]-1
        if new_pos_y < 0: new_pos_y = 0
        if new_pos_y >= self.monitor_resolution[1]: new_pos_y = self.monitor_resolution[1]-1
        pyautogui.moveTo(new_pos_x, new_pos_y, _pause=False)


    def move_mouse_to_abs_position(self):
        if self.position is None or self.velocity is None:
            return

        monitor_position =  int(self.position[0]*self.camera_resolution[0]), int(self.position[1]*self.camera_resolution[1])
        monitor_delta = (self.velocity[0]**2 + self.velocity[1]**2)**0.5

        if abs(int(monitor_delta*self.monitor_resolution[0])) > 4:
            x, y = monitor_position
            x_ = ((x - (self.camera_resolution[0] - self.working_area[0]) / 2) / self.working_area[0]) * self.monitor_resolution[0]
            y_ = ((y - (self.camera_resolution[1] - self.working_area[1]) / 2) / self.working_area[1]) * self.monitor_resolution[1]
            monitor_position = [int(x_), int(y_)]
            self.move_mouse(monitor_position)


    def move_mouse_to_rel_position(self):
        if self.velocity is None or self.acceleration is None:
            return

        k = 0
        mouse_position = pyautogui.position()
        if self.mouse_speed> 0.001:
            k = 1000
        if self.mouse_speed > 0.005:
            k = 2000
        if self.mouse_speed > 0.01:
            k = 3000
        if self.mouse_speed > 0.02:
            k = 4000
        if self.mouse_speed > 0.1:
            k = 8000

        if self.mouse_accel > 0.01:
            k = k*2
        elif self.mouse_accel > 0.02:
            k = k*3

        if k > 0:
            monitor_position = [int(mouse_position.x + self.velocity[0]*k),
                                int(mouse_position.y + self.velocity[1]*k)]
            self.move_mouse(monitor_position)



if __name__ == '__main__':
    import mediapipe as mp
    import cv2
    from rscamera import RealSenseCamera
    import time
    import numpy as np

    camera_width = 1280
    camera_height = 720
    def get_hand_landmarks(results):
        hands_landmarks = []
        if results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(results.multi_handedness, results.multi_hand_landmarks):
                hand = {}
                lm_list = [] 
                for lm in hand_lms.landmark:
                    px, py, pz = lm.x, lm.y, lm.z
                    lm_list.append([px, py, pz])
                hand["lm_list"] = lm_list
                hand["type"] = hand_type.classification[0].label
                hands_landmarks.append(hand)
        return hands_landmarks


    def palm_pressed(keypoints):
        if keypoints is None:
            return False
        thumb_kp = keypoints[4]
        left_kp = keypoints[13]
        click_value = ( (left_kp[0]-thumb_kp[0])**2 + (left_kp[1]-thumb_kp[1])**2 ) ** 0.5
        return click_value < 0.03

    # MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        max_num_hands = 1, min_detection_confidence = 0.5, model_complexity = 1, min_tracking_confidence = 0.5
    )

    # Camera
    cam = RealSenseCamera(width=1280, height=720, fps=30, enable_color=True, enable_depth=True, enable_emitter=True)
    cam.start()

    # Mouse
    mouse = MouseControl()
    mouse.init()
    currTime = 0

    try:
        while True:
            # Get frame data
            depth_frame, color_frame = cam.get_frames()
            if not depth_frame or not color_frame:
                print('Frame dropped')
                break

            color_image = cv2.flip(np.asanyarray(color_frame.get_data()), 1)

            image = color_image.copy()
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            prevTime = currTime
            results = detector.process(image)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            cv2.putText(annotated_image, f'{fps:.2f} fps', (0, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

            # keypoints_xy = None
            if results.multi_hand_landmarks:
                for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # keypoints = []
                    # keypoints_xy = []
                    # for pt_idx, data_point in enumerate(hand_landmarks.landmark):
                    #     keypoints_xy += [data_point.x, data_point.y]

            # click_state = palm_pressed(keypoints_xy)
            # mouse.update(keypoints_xy, click_state)

            hands = get_hand_landmarks(results)
            lm_list1 = None
            if hands:
                hand1 = hands[0]
                lm_list1 = hand1["lm_list"]

            click_state = palm_pressed(lm_list1)
            mouse.update(lm_list1, click_state)

            if mouse.position is not None:
                cx, cy = mouse.position
                cx, cy = int(cx*image_width), int(cy*image_height)
                cv2.circle(annotated_image, (cx,cy), 5, (0,255,0), thickness=-1)

            vel_text = "Mouse Speed: {:.6f}, Peak: {:6f}".format(mouse.mouse_speed, mouse.max_speed)
            cv2.putText(annotated_image, vel_text, (0, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

            acc_text = "Mouse Accel: {:.6f}, Peak: {:6f}".format(mouse.mouse_accel, mouse.max_accel)
            cv2.putText(annotated_image, acc_text, (0, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

            mouse.move_mouse_to_rel_position()

            cv2.imshow('Display', annotated_image)
            keycode = cv2.waitKey(1)
            if (keycode & 0xFF) == 27 or (keycode & 0xFF) == ord('q'):
                break
            if keycode == ord('m'):
                mouse.init()

    except Exception as e:
        print(e)

    finally:
        cv2.destroyAllWindows()
        cam.stop()
