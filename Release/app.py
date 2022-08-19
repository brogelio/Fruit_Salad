from tkinter import Label, LabelFrame, Button, Toplevel, LEFT, Tk, PhotoImage
import webbrowser
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import pyqtSignal, QThread
import mediapipe as mp
import numpy as np
import cv2
import time
import sys
import os
from utils.rscamera import RealSenseCamera  # Causes PyQt Warning
from utils.timer import Timer
from utils.multithread import ThreadedImageWriter
from utils.mouse import MouseControl
from utils.gestures import *
from utils.overlay import MainWindow
import logging
logger = logging.getLogger('app.py')
import pyautogui
pyautogui.FAILSAFE = False
import keyboard



# Configuration variables
emitter = True  # you can disable RealSense IR emitter
log_level = logging.DEBUG  # log everything
log_dir = 'log'  # log directory (automatically created)
save_frames = False  # you can save captured frames to {log_dir}\run{n}\frames
glove_type = "l_white"  # "nl_green" or "l_white"
dominant_hand = "Right"  # "Right" or "Left"
enable_keypress = True  # you can disable keypress when not needed
monitor_resolution = (1920, 1080) #(2560, 1440)
which_series = 0.3  # 480 or 1440 x-value
scroll_mode = "scroll"  # "scroll" or "drag"
scroll_speed = 50   # not sure about units; doesn't seem to be affected by Windows 10 scroll settings
mouse_mode = "palm"  # "palm" or "middle"
click_mode = "curl"  # "curl" or "palm"
zoom_sensitivity = 3.5  # default value: 3.5
swipe_sensitivity = [5, 1]  # default value: [5, 1], previous value: [5, 0]
swipe_vert_adj = 16/9  # default value: 16/9, previous value: 4/3
swipe_up_adj = 1  # default value: 1, previous value: 1.5
drag_sensitivity = [2, 6, -7]  # default value: [1, 6, -7], previous value: [1, 1, -100]

# Global variables
CAMERA_RESOLUTION = (1280, 720)
OVERLAY_SIZE = (440, 360)
OVERLAY_OPACITY = 0.6
OVERLAY_CROP_X = CAMERA_RESOLUTION[0]//2 - OVERLAY_SIZE[0]
LKBX_SZ = [0.17, 0.4]
BX1_POS = [0.25, 0.3]
BX2_POS = [0.58, 0.3]
UNLOCK_BOX_REL = [[BX1_POS[0], BX1_POS[1]],[BX1_POS[0]+LKBX_SZ[0], BX1_POS[1]+LKBX_SZ[1]]]
LOCK_BOX_REL   = [[BX2_POS[0], BX1_POS[1]],[BX2_POS[0]+LKBX_SZ[0], BX2_POS[1]+LKBX_SZ[1]]]
UNLOCK_BOX_ABS = [[int(UNLOCK_BOX_REL[0][0]*CAMERA_RESOLUTION[0]),int(UNLOCK_BOX_REL[0][1]*CAMERA_RESOLUTION[1])],
                  [int(UNLOCK_BOX_REL[1][0]*CAMERA_RESOLUTION[0]),int(UNLOCK_BOX_REL[1][1]*CAMERA_RESOLUTION[1])]]
LOCK_BOX_ABS   = [[int(LOCK_BOX_REL[0][0]*CAMERA_RESOLUTION[0]),int(LOCK_BOX_REL[0][1]*CAMERA_RESOLUTION[1])],
                  [int(LOCK_BOX_REL[1][0]*CAMERA_RESOLUTION[0]),int(LOCK_BOX_REL[1][1]*CAMERA_RESOLUTION[1])]]
idx_belt = []
zoom_belt = []
rotate_belt = []
FPS = 25
BEGIN_PROGRAM = False



# Clears gesture belt lists if they are not empty
def clear_gesture_belts():
    global idx_belt, zoom_belt, rotate_belt
    if idx_belt:
        idx_belt.clear()
    if zoom_belt:
        zoom_belt.clear()
    if rotate_belt:
        rotate_belt.clear()


# Lock/unlock helper
def is_inside_box(lm_list, box):
    min_x, min_y = CAMERA_RESOLUTION[0], CAMERA_RESOLUTION[1]
    max_x, max_y = -1, -1
    for idx in range(len(lm_list)-1):
        if lm_list[idx][0] < min_x:
            min_x = lm_list[idx][0]
        if lm_list[idx][0] > max_x:
            max_x = lm_list[idx][0]
        if lm_list[idx][1] < min_y:
            min_y = lm_list[idx][1]
        if lm_list[idx][1] > max_y:
            max_y = lm_list[idx][1]
    if min_x > box[0][0] and max_x < box[1][0] and min_y > box[0][1] and max_y < box[1][1]:
        return True
    return False


# Lock/unlock helper
def toggle_lock(lm_list, is_inside, is_locked, box):
    global inside_time
    is_inside_new =  is_inside_box(lm_list, box)
    is_locked_new = is_locked
    if is_inside_new:
        if not is_inside:
            inside_time = time.time()
        else:
            if time.time() - inside_time > 3:
                is_locked_new = not is_locked
                inside_time = time.time() + 3
    return is_inside_new, is_locked_new


# Sleep Timer
def sleeping():
    global sleepy_time
    if time.time() - sleepy_time > 1:  # 1 second sleep time
        return False
    else:
        return True


# Initialize logger
def init_logging(log_dir: str) -> str:
    # Create log_dir if it doesn't exist
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        logger.info(f'Creating path: {log_dir}')

    # Create run directory
    run_num = 1
    while True:
        run_dir = os.path.join(log_dir, f'run{run_num}')
        if os.path.exists(run_dir):
            run_num += 1
        else:
            os.mkdir(run_dir)
            os.mkdir(os.path.join(run_dir, 'frames'))
            logger.info(f'Creating path: {run_dir}')
            break

    logging.basicConfig(filename=f'{run_dir}\\run.log', format='%(asctime)s - %(name)s - %(message)s',
                        encoding='utf-8', level=log_level)
    return run_dir


# RealSense image caputure subroutine
def get_frame(camera):
    depth_frame, color_frame = camera.get_frames()
    if not depth_frame or not color_frame:
        return False
    color_image = cv2.flip(np.asanyarray(color_frame.get_data()), 1)
    depth_image = cv2.flip(np.asanyarray(depth_frame.get_data()), 1)
    return color_image, depth_image


# Generate overlay displayed on the top-right corner
def generate_overlay(color_image, **kwargs):
    results = kwargs['results']
    fps = kwargs['fps']
    depth_image = kwargs['depth_image']
    depth_coords = kwargs['depth_coords']
    gesture_mode = kwargs['gesture_mode']
    gesture_txt = kwargs['gesture_txt']
    is_inside = kwargs['is_inside']
    is_locked = kwargs['is_locked']
    is_waiting = kwargs['is_waiting']
    annotated_image = color_image.copy()

    # Lock/unlock region overlay
    if is_locked:
        box_color = (0, 0, 255) if not is_inside else (0, 255, 255)
        cv2.rectangle(annotated_image, [UNLOCK_BOX_ABS[0][0], UNLOCK_BOX_ABS[0][1]], [UNLOCK_BOX_ABS[1][0],
                        UNLOCK_BOX_ABS[1][1]], box_color, thickness=2)
        cv2.putText(annotated_image, f'LOCKED', (UNLOCK_BOX_ABS[0][0]+20, UNLOCK_BOX_ABS[1][1] + 40),
                        cv2.FONT_HERSHEY_PLAIN, 3, box_color, 2)
    else:
        box_color = (0, 255, 0) if not is_inside else (0, 255, 255)
        cv2.rectangle(annotated_image, [LOCK_BOX_ABS[0][0], LOCK_BOX_ABS[0][1]], [LOCK_BOX_ABS[1][0],
                        LOCK_BOX_ABS[1][1]], box_color, thickness=2)
        cv2.putText(annotated_image, f'UNLOCKED', (LOCK_BOX_ABS[0][0]-10, LOCK_BOX_ABS[1][1] + 40),
                        cv2.FONT_HERSHEY_PLAIN, 3, box_color, 2)

    # Waiting for gesture overlay
    if is_waiting:
        cv2.putText(annotated_image, 'Waiting for Gesture', (340, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    # FPS overlay
    cv2.putText(annotated_image, f'{fps:.0f}fps', (1030, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)

    # Mode overlay
    mode_text = "MODE: GESTURE" if gesture_mode else "MODE: BROWSE"
    cv2.putText(annotated_image, mode_text, (320, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)

    # Gesture overlay
    if gesture_txt is not None:
        if gesture_txt[0:5] != "Panel" and gesture_txt[0:5] != "Thumb":
            warning_overlay = np.zeros_like(annotated_image, np.uint8)
            cv2.rectangle(warning_overlay, [320,180], [960,660], (0,0,255), thickness=cv2.FILLED)
            cv2.putText(annotated_image, gesture_txt, (340, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
            alpha = 0.5
            mask = warning_overlay.astype(bool)
            annotated_image[mask] = cv2.addWeighted(annotated_image, alpha, warning_overlay, 1 - alpha, 0)[mask]
        else:
            cv2.putText(annotated_image, gesture_txt, (340, 310), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

    # MediaPipe landmark overlay
    if results.multi_hand_landmarks:
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Distance overlay
    if depth_coords is not None:
        distance = depth_image[depth_coords[1], depth_coords[0]]
        cv2.putText(annotated_image, "{}mm".format(distance), (depth_coords[0], depth_coords[1] - 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)

    annotated_image = annotated_image[0:720, OVERLAY_CROP_X//2:1280-OVERLAY_CROP_X//2]
    return annotated_image



class MainThread(QThread):
    """
    This class implements a PyQt Thread for the main program loop
    """
    OverlayUpdate = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True
        self.main_loop()

    def stop(self):
        self.ThreadActive = False
        # Note: The thread is terminated at the end of the main_loop()

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pic = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).scaled(OVERLAY_SIZE[0], OVERLAY_SIZE[1])
        self.OverlayUpdate.emit(pic)  # calls the function linked through OverlayUpdate.connect()

    def main_loop(self):
        global run_dir, image_writer, camera, detector, mouse
        global ptr_coords, depth_coords, sleepy_time
        global idx_belt, zoom_belt, rotate_belt

        # Initialize loop variables
        sleepy_time = time.time()
        inside_time = time.time()
        prevTime = time.time() - 1e-3
        gesture_mode = False
        is_inside = False
        is_locked = True
        gesture_txt = None
        ptr_coords = None
        depth_coords = None
        is_waiting = False
        panel1_enable = False
        panel2_enable = False

        logger.info('Entering main program loop.')
        try:
            while self.ThreadActive:
                try:
                    # Compute running FPS for display
                    currTime = time.time()
                    fps = 1 / (currTime - prevTime)
                    prevTime = currTime

                    # Get camera image
                    camera_timer.tic()
                    color_image, depth_image = get_frame(camera)
                    camera_timer.toc()
                    if color_image is False:
                        logger.error('Frame dropped. Exiting program.')
                        break

                    # Save frames to log_dir
                    save_timer.tic()
                    if save_frames:
                        image_writer.save(f'{run_dir}\\frames\\{camera.frame_count}.jpg', color_image)
                    save_timer.toc()
                    save_color = color_image
                    # Hue shift if wearing green latex gloves
                    if glove_type == "nl_green":
                        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                        h_only = hsv_image[:, :, 0]
                        h_only = 180 + h_only
                        hsv_image[:, :, 0] = h_only
                        h_sv = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
                        color_image = cv2.cvtColor(h_sv, cv2.COLOR_HSV2BGR)

                    # Use MediaPipe to detect hand
                    mediapipe_timer.tic()
                    results = detector.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                    hands = get_hand_landmarks_rel(results)
                    mediapipe_timer.toc()

                    if hands:
                        hand1 = hands[0] if hands[0]["type"] == dominant_hand else hands[1]
                        lm_list1 = hand1["lm_list"]  # List of 21 Landmarks
                        handedness = hand1["type"]

                        # Gesture classifier and mouse mode will only work on the specified hand
                        if handedness == dominant_hand:

                            if is_locked:
                                is_inside, is_locked = toggle_lock(lm_list1, is_inside, is_locked, box=UNLOCK_BOX_REL)
                                if not is_locked:
                                    clear_gesture_belts()
                            else:
                                is_inside, is_locked = toggle_lock(lm_list1, is_inside, is_locked, box=LOCK_BOX_REL)
                            
                            may_swipe = not (is_inside or is_locked)

                            ptr_coords = [int(lm_list1[8][0] * CAMERA_RESOLUTION[0]),
                                          int(lm_list1[8][1] * CAMERA_RESOLUTION[1])]
                            depth_coords = [int(lm_list1[9][0] * CAMERA_RESOLUTION[0]),
                                            int(lm_list1[9][1] * CAMERA_RESOLUTION[1])]

                            # Check finger curl
                            fc = hand_curl(lm_list1)
                            
                            is_curl = False
                            may_rotate = False
                            if fc[1] == 1 and fc[2] == 1 and fc[3] == 1 and fc[4] == 1:
                                is_curl = True
                            if fc[3] == 1 or fc[4] == 1:
                                may_rotate = True

                            # Check thumb orientation
                            is_thumbs_up = gesture_flip(lm_list1)

                            is_waiting = False
                            # Gesture Mode
                            if gesture_mode and not sleeping() and not is_locked:
                                gesture_timer.tic()
                                gesture_txt = None
                                is_waiting = True

                                # Belt Filling Station
                                belt_fill(lm_list1, ptr_coords, may_swipe, is_curl, may_rotate, idx_belt, zoom_belt, rotate_belt, FPS)

                                # Check Swipe Gestures
                                # swipe up/down ;; change series ;; left right arrow keys
                                # swipe left/right ;; change img num ;; down up arrow keys
                                is_swipe = False
                                if not is_curl and not may_rotate and len(idx_belt) == FPS:
                                    is_swipe, swipe_dir = swiped(idx_belt, thresh=swipe_sensitivity, vert_adj=swipe_vert_adj, up_adj=swipe_up_adj)
                                    if is_swipe:
                                        if swipe_dir == "Up":
                                            print("i swiped up")
                                            logger.info(f'Gesture: Swipe Up')
                                            gesture_txt = "Swipe Up   "
                                            if enable_keypress:
                                                keyboard.press_and_release('left')
                                        elif swipe_dir == "Down":
                                            print("i swiped down")
                                            logger.info(f'Gesture: Swipe Down')
                                            gesture_txt = "Swipe Down "
                                            if enable_keypress:
                                                keyboard.press_and_release('right')
                                        elif swipe_dir == "Left":
                                            print("i swiped left")
                                            logger.info(f'Gesture: Swipe Left')
                                            gesture_txt = "Swipe Left "
                                            if enable_keypress:
                                                keyboard.press_and_release('up')
                                        elif swipe_dir == "Right":
                                            print("i swiped right")
                                            logger.info(f'Gesture: Swipe Right')
                                            gesture_txt = "Swipe Right"
                                            if enable_keypress:
                                                keyboard.press_and_release('down')
                                        else:
                                            logger.info(f'Error in Swipe Gesture')

                                        clear_gesture_belts()  # Clear arrays
                                        sleepy_time = time.time()  # Sleep time reset


                                # Check Zoom Gestures
                                # zoom in/out ;; ctrl ++ ctrl --
                                is_zoom = False
                                if may_rotate and not is_curl and not is_swipe and len(zoom_belt) == FPS:
                                    is_zoom, zoom_mode = zoomed(zoom_belt, thresh=zoom_sensitivity, out_adj=1.4)
                                    if is_zoom:
                                        if zoom_mode == "zoom_in":
                                            print("i zoomed in")
                                            logger.info(f'Gesture: Zoom In')
                                            gesture_txt = "Zoom In    "
                                            if enable_keypress:
                                                keyboard.press_and_release('ctrl+plus+plus')
                                        elif zoom_mode == "zoom_out":
                                            print("i zoomed out")
                                            logger.info(f'Gesture: Zoom Out')
                                            gesture_txt = "Zoom Out  "
                                            if enable_keypress:
                                                keyboard.press_and_release('ctrl+-+-')
                                        else:
                                            logger.info(f'Error in Zoom Gesture')

                                        clear_gesture_belts()  # Clear arrays
                                        sleepy_time = time.time()  # Sleep time reset


                                # Check Thumbs Up Gesture
                                if is_thumbs_up and is_curl and not is_swipe and not is_zoom:
                                    gesture_mode = False
                                    print("gesture mode now off")
                                    logger.info(f'Gesture: Thumbs Up')
                                    gesture_txt = "Thumbs Up"
                                    mouse.init()
                                    if enable_keypress:
                                        pyautogui.press('b')

                                    clear_gesture_belts()  # Clear arrays
                                    sleepy_time = time.time() + 0.5  # Sleep time reset

                                gesture_timer.toc()
                            # End of Gesture Mode

                            # Browse Mode
                            if not gesture_mode and not sleeping() and not is_locked:
                                browse_timer.tic()
                                gesture_txt = None
                                global which_series

                                idx_is_up = False
                                if lm_list1[8][1] < lm_list1[7][1] < lm_list1[6][1]:
                                    idx_is_up = True
                                mid_is_up = False
                                if lm_list1[12][1] < lm_list1[11][1] < lm_list1[10][1]:
                                    mid_is_up = True

                                # Two-finger detection
                                if fc[1] == 0 and fc[2] == 0 and fc[3] == 1 and fc[4] == 1 and not sleeping() and \
                                                                            idx_is_up and mid_is_up and panel2_enable:
                                    which_series = 0.8
                                    print("2 fing")
                                    gesture_txt = "Panel 2"
                                    logger.info(f'Gesture: Two fingers up')
                                    if scroll_mode == "scroll":
                                        pyautogui.leftClick(2*monitor_resolution[0]//3, monitor_resolution[1]//2, 
                                                            _pause=False)
                                    sleepy_time = time.time()

                                # One-finger detection
                                if fc[1] == 0 and fc[2] == 1 and fc[3] == 1 and fc[4] == 1 and not sleeping() and \
                                                                        idx_is_up and not mid_is_up and panel1_enable:
                                    which_series = 0.3
                                    print("1 fing")
                                    gesture_txt = "Panel 1"
                                    logger.info(f'Gesture: One finger up')
                                    if scroll_mode == "scroll":
                                        pyautogui.leftClick(monitor_resolution[0]//3, monitor_resolution[1]//2,
                                                            _pause=False)
                                    sleepy_time = time.time()

                                # Pre-detection phase
                                if fc[1] == 0 and fc[2] == 0 and fc[3] == 1 and fc[4] == 1 and not sleeping() and \
                                                                                            idx_is_up and mid_is_up:
                                    panel2_enable =True
                                elif fc[1] == 0 and fc[2] == 1 and fc[3] == 1 and fc[4] == 1 and not sleeping() \
                                                                                    and idx_is_up and not mid_is_up:
                                    panel1_enable =True
                                else:
                                    panel1_enable = False
                                    panel2_enable = False

                                if click_mode == "curl":
                                    click_state = curl_click(lm_list1)
                                else:
                                    click_state = thumb_click(lm_list1)

                                # Use scroll wheel commands
                                if scroll_mode == "scroll":
                                    mouse.update(lm_list1, click_state)
                                    mouse.scroll_drag(scroll_clicks=scroll_speed, drag_sensitivity=drag_sensitivity[0],
                                        trigger_sensitivity=drag_sensitivity[1], thumbs_up_sensitivity=drag_sensitivity[2])
                                # Use click and drag commands
                                else:
                                    mouse.update2(lm_list1, click_state, which_series)
                                    mouse.move_mouse_to_abs_position()

                                if mouse.left_pressed:
                                    color_image = cv2.putText(color_image, 'HAND CLOSED', (320, 700),
                                                              cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 255, 255),
                                                              thickness=4)
                                else:
                                    color_image = cv2.putText(color_image, 'HAND OPENED', (320, 700),
                                                              cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 255, 0),
                                                              thickness=4)
                                browse_timer.toc()

                                # Check Thumbs Up Gesture
                                if is_thumbs_up and is_curl and not sleeping():
                                    pyautogui.mouseUp()  ### new
                                    gesture_mode = True
                                    print("gesture mode now on")
                                    logger.info(f'Gesture: Thumbs Up')
                                    gesture_txt = "Thumbs Up"
                                    if enable_keypress:
                                        pyautogui.press('b')

                                    clear_gesture_belts()  # Clear arrays
                                    sleepy_time = time.time() + 0.5  # Sleep time reset
                            # End of Browse Mode

                    overlay_timer.tic()
                    annotated_image = generate_overlay(save_color, results=results, fps=fps, depth_image=depth_image,
                                                       depth_coords=depth_coords, gesture_mode=gesture_mode,
                                                       gesture_txt=gesture_txt, is_inside=is_inside,
                                                       is_locked=is_locked, is_waiting=is_waiting)

                    self.display_image(annotated_image)
                    overlay_timer.toc()

                except:
                    annotated_image = generate_overlay(save_color, results=results, fps=fps, depth_image=depth_image,
                                                       depth_coords=depth_coords, gesture_mode=gesture_mode,
                                                       gesture_txt=gesture_txt, is_inside=is_inside,
                                                       is_locked=is_locked, is_waiting=is_waiting)
                    self.display_image(annotated_image)
                    logger.error('Main program terminated with exception.', exc_info=True)

        except:
            logger.error('Main program terminated with exception.', exc_info=True)

        else:
            logger.info('Main program terminated normally.')

        finally:
            camera.stop()
            camera_timer.summary()
            save_timer.summary()
            mediapipe_timer.summary()
            gesture_timer.summary()
            browse_timer.summary()
            overlay_timer.summary()
            image_writer.stop()
            self.quit()
            QtWidgets.qApp.quit()



# Initial program GUI
class MainProgramGUI(object):
    def __init__(self, title="fruit salad", icon="media/hand.ico"):
        self.root = Tk()
        self.root.title(title)
        self.root.iconbitmap(icon)
        self.root.resizable(0,0)
        gui_w_photo = PhotoImage(file="media/white_gloves.png")
        self.gui_w_photo_rz = gui_w_photo.subsample(8, 8)
        gui_g_photo = PhotoImage(file="media/green_gloves.png")
        self.gui_g_photo_rz = gui_g_photo.subsample(8, 8)
        create_program_gui(self.root, self.gui_w_photo_rz, self.gui_g_photo_rz)
    
    def get_root(self):
        return self.root


def create_program_gui(root, gui_w_photo, gui_g_photo):
    # Button functions
    def gui_glove_color(color):
        print(color)
        global glove_type
        glove_type = "l_white" if color == "WHITE GLOVES" else "nl_green"
        glove_state = Label(gui_color_frame, text=color, font=("Consolas", 20, "bold"))
        glove_state.grid(row=0, column=1, columnspan=2)

    def gui_dominant_hand(hand):
        print(hand)
        global dominant_hand
        dominant_hand = "Right" if hand == "RIGHT HAND" else "Left"
        chosen_hand = Label(gui_hand_frame, text=hand, font=("Consolas", 20, "bold"))
        chosen_hand.grid(row=0, column=1, columnspan=2)

    # Frames
    gui_color_frame = LabelFrame(root, padx=20, pady=20)
    gui_color_frame.grid(row=0, column=0, padx=10, pady=10)

    gui_hand_frame = LabelFrame(root, padx=5, pady=20)
    gui_hand_frame.grid(row=1, column=0)

    glove_state_d = Label(gui_color_frame, text="WHITE GLOVES", font=("Consolas", 20, "bold"))
    glove_state_d.grid(row=0, column=1, columnspan=2)
    chosen_hand_d = Label(gui_hand_frame, text="RIGHT HAND", font=("Consolas", 20, "bold"))
    chosen_hand_d.grid(row=0, column=1, columnspan=2)

    # Buttons Widget; colors can be in HEX color codes
    # Glove colors
    gui_glove_color_label = Label(gui_color_frame, text="Glove Color:")
    gui_glove_color_label.grid(row=0, column=0)
    gui_w_imageButton = Button(gui_color_frame, text="White Gloves", image=gui_w_photo, compound=LEFT,
                                command=lambda: gui_glove_color("WHITE GLOVES"))
    gui_w_imageButton.grid(row=1, column=1)
    gui_g_imageButton = Button(gui_color_frame, text="Green Gloves", image=gui_g_photo, compound=LEFT,
                                command=lambda: gui_glove_color("GREEN GLOVES"))
    gui_g_imageButton.grid(row=1, column=2)

    # Handedness
    gui_dominant_hand_label = Label(gui_hand_frame, text="Dominant Hand:")
    gui_dominant_hand_label.grid(row=0, column=0)
    gui_right_button = Button(gui_hand_frame, text="Right Hand", command=lambda: gui_dominant_hand("RIGHT HAND"),
                                padx=40, pady=20)
    gui_right_button.grid(row=1, column=2)

    gui_left_button = Button(gui_hand_frame, text="Left Hand", command=lambda: gui_dominant_hand("LEFT  HAND"),
                                padx=40, pady=20)
    gui_left_button.grid(row=1, column=1)

    def begin_program(root):
        global BEGIN_PROGRAM
        BEGIN_PROGRAM = True
        root.destroy()

    # Begin
    gui_button_begin = Button(root, text="Begin Gesture Program", command=lambda: begin_program(root))
    gui_button_begin.grid(row=4, column=0, pady=5)

    # About
    def open_link(url):
        webbrowser.open_new(url)

    def create_about_window():
        abt_window = Toplevel(root)
        abt_window.title('About')
        Label(abt_window, text="fruit salad\n \'A Hand Gesture Recognition Program\' \n").pack()
        ez_icon = Label(abt_window, text="App icon created by EasyIcons of icon-icons.com", fg="blue", cursor="hand2")
        ez_icon.pack()
        ez_icon.bind("<Button-1>", lambda e: open_link("https://icon-icons.com/icon/hand-gesture-hands/42574"))

        gl_icon = Label(abt_window, text="Gloves icons created by Smashicons - Flaticon", fg="blue", cursor="hand2")
        gl_icon.pack()
        gl_icon.bind("<Button-1>", lambda e: open_link("https://www.flaticon.com/free-icons/gloves"))

        cc_link = Label(abt_window, text="Licensed under Creative Commons", fg="blue", cursor="hand2")
        cc_link.pack()
        cc_link.bind("<Button-1>", lambda e: open_link("https://creativecommons.org/licenses/by/4.0/"))

    gui_button_abt = Button(root, text="About", command=create_about_window)
    gui_button_abt.grid(row=5, column=0, pady=5)




if __name__ == '__main__':
    # Create log directories for current run
    run_dir = init_logging(log_dir)

    # Start threaded image writer
    image_writer = ThreadedImageWriter().start()

    # Camera
    camera = RealSenseCamera(width=1280, height=720, fps=30, enable_color=True, enable_depth=True,
                            enable_emitter=emitter)
    camera.start()

    # Tk GUI
    init_window = MainProgramGUI()
    root = init_window.get_root()

    # Loop of program window
    root.mainloop()

    if not BEGIN_PROGRAM:
        logger.info("Program exited.")
        sys.exit()
    else:
        print("Program Start")
        logger.info("Program started.")

    # MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    if glove_type == "l_white":
        detector = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.8,
                                            min_tracking_confidence=0.8)
    if glove_type == "nl_green":
        detector = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.4,
                                            min_tracking_confidence=0.4)

    # Mouse
    mouse = MouseControl(ptr_finger=mouse_mode, monitor_resolution=monitor_resolution)
    mouse.init()

    # Timers
    camera_timer = Timer('frame retrieval')
    mediapipe_timer = Timer('keypoint detection')
    save_timer = Timer('frame storage')
    gesture_timer = Timer('gesture mode')
    browse_timer = Timer('browsing mode')
    overlay_timer = Timer('overlay processing')

    # Qt GUI
    app = QApplication([])
    main_window = MainWindow(size=OVERLAY_SIZE, window_opacity=OVERLAY_OPACITY)
    main_window.setLayout(main_window.vbl)
    main_window.show()

    # Spawn QThread
    main_window.main_thread = MainThread()
    main_window.main_thread.start()
    main_window.main_thread.OverlayUpdate.connect(main_window.overlayUpdateCallback)

    # Main Program Loop
    app.exec_()
