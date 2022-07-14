from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import pyqtSignal, QThread
import mediapipe as mp
import numpy as np
import cv2
import time
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
from tkinter import *

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
scroll_mode = "scroll" # "scroll" or "drag"
scroll_speed = 50   # not sure about units; doesn't seem to be affected by Windows 10 scroll settings

# Relative mouse movement
# mouse_mode = "middle"
# click_mode = "palm"

# Absolute mouse movement
mouse_mode = "palm"  # "palm" or "middle"
click_mode = "curl"  # "curl" or "palm"




# Global variables
CAMERA_RESOLUTION = (1280, 720)
idx_belt = []
zoom_belt = []
rotate_belt = []
FPS = 17 # edit(?)

# Sleep Timer
def sleeping():
    global sleepy_time
    if time.time() - sleepy_time > 1:  # 1 second sleep time
        return False
    else:
        return True

# Initialize Logger
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
    ptr_coords = kwargs['ptr_coords']
    depth_coords = kwargs['depth_coords']
    gesture_mode = kwargs['gesture_mode']
    gesture_txt = kwargs['gesture_txt']

    annotated_image = color_image.copy()

    # FPS overlay
    cv2.putText(annotated_image, f'{fps:.2f}fps', (1280 - 300, 0 + 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)

    # Mode overlay
    mode_text = "Gesture" if gesture_mode else " Mouse "
    cv2.putText(annotated_image, mode_text, (1280 - 240, 0 + 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Gesture overlay
    if gesture_mode and gesture_txt is not None:
        cv2.putText(annotated_image, gesture_txt, (1280 - 380, 0 + 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255),
                    5)  # from 320 to 380

    # MediaPipe Landmark Overlay
    if results.multi_hand_landmarks:
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Pointer Overlay
    # if ptr_coords is not None:
    #     if ptr_coords[0] >= CAMERA_RESOLUTION[0]:
    #         ptr_coords[0] = (CAMERA_RESOLUTION[0] - 1)
    #     if ptr_coords[1] >= CAMERA_RESOLUTION[1]:
    #         ptr_coords[1] = (CAMERA_RESOLUTION[1] - 1)
    #     cv2.circle(annotated_image, (ptr_coords[0],ptr_coords[1]), 5, (0,255,0), thickness=-1)

    # Distance Overlay
    if depth_coords is not None:
        distance = depth_image[depth_coords[1], depth_coords[0]]
        cv2.putText(annotated_image, "{}mm".format(distance), (depth_coords[0], depth_coords[1] - 100),
                    # from 40 to 100
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)

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
        pic = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).scaled(640, 360)
        self.OverlayUpdate.emit(pic)  # calls the function linked through OverlayUpdate.connect()

    def main_loop(self):
        global run_dir, image_writer, camera, detector, mouse
        global ptr_coords, depth_coords, sleepy_time
        global idx_belt, zoom_belt, rotate_belt

        # Initialize loop variables
        sleepy_time = time.time()
        prevTime = time.time() - 1e-3
        gesture_mode = True
        gesture_txt = None
        ptr_coords = None
        depth_coords = None

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

                            # Gesture Mode
                            if gesture_mode and not sleeping():
                                gesture_timer.tic()
                                gesture_txt = None
                                color_image = cv2.putText(color_image, 'Waiting for Gesture', (1280 - 640, 0 + 150),
                                                          # from 480 to 520
                                                          cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 255, 0), thickness=5)

                                # Belt Filling Station
                                belt_fill(lm_list1, ptr_coords, is_curl, may_rotate, idx_belt, zoom_belt, rotate_belt,
                                          FPS)

                                # Check Swipe Gestures
                                # swipe up/down ;; change series ;; left right arrow keys
                                # swipe left/right ;; change img num ;; down up arrow keys
                                is_swipe = False
                                if not is_curl and not may_rotate and len(idx_belt) == FPS:
                                    is_swipe, swipe_dir = swiped(idx_belt)
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

                                        idx_belt = []  # Clear arrays
                                        zoom_belt = []
                                        rotate_belt = []
                                        sleepy_time = time.time()  # Sleep time reset

                                # Check Zoom Gestures
                                # zoom in/out ;; ctrl ++ ctrl --
                                is_zoom = False
                                if may_rotate and not is_curl and not is_swipe and len(zoom_belt) == FPS:
                                    is_zoom, zoom_mode = zoomed(zoom_belt)
                                    if is_zoom:
                                        if zoom_mode == "zoom_in":
                                            print("i zoomed in")
                                            logger.info(f'Gesture: Zoom In')
                                            gesture_txt = "   Zoom In "
                                            if enable_keypress:
                                                keyboard.press_and_release('ctrl+plus+plus')
                                        elif zoom_mode == "zoom_out":
                                            print("i zoomed out")
                                            logger.info(f'Gesture: Zoom Out')
                                            gesture_txt = "   Zoom Out"
                                            if enable_keypress:
                                                keyboard.press_and_release('ctrl+-+-')
                                        else:
                                            logger.info(f'Error in Zoom Gesture')

                                        idx_belt = []  # Clear arrays
                                        zoom_belt = []
                                        rotate_belt = []
                                        sleepy_time = time.time()  # Sleep time reset

                                # Check Rotate Gestures
                                is_rotate = False

                                # Rotate removed
                                # if is_curl and not is_swipe and not is_zoom and lm_list1[12][1] < lm_list1[3][1] and len(rotate_belt) == FPS:
                                #     is_rotate, rotate_mode = rotate(rotate_belt)
                                #     if is_rotate:
                                #         if rotate_mode == "rotate_cw":
                                #             print("i rotated cw")
                                #             logger.info(f'Gesture: Rotate Clockwise')
                                #             gesture_txt = " Rotate CW "
                                #             if enable_keypress:
                                #                 pyautogui.hotkey('ctrl', ']')
                                #                 # keyboard.press_and_release('ctrl+]')
                                #         elif rotate_mode == "rotate_ccw":
                                #             print("i rotated ccw")
                                #             logger.info(f'Gesture: Rotate Counter-clockwise')
                                #             gesture_txt = " Rotate CCW"
                                #             if enable_keypress:
                                #                 pyautogui.hotkey('ctrl', '[')
                                #                 # keyboard.press_and_release('ctrl+[')
                                #         else:
                                #             logger.info(f'Error in Rotate Gesture')
                                #
                                #         idx_belt = []   # Clear arrays
                                #         zoom_belt = []
                                #         rotate_belt = []
                                #         sleepy_time = time.time()   # Sleep time reset

                                # Check Thumbs Up Gesture
                                if is_thumbs_up and is_curl and not is_swipe and not is_zoom and not is_rotate:
                                    gesture_mode = False
                                    print("gesture mode now off")
                                    logger.info(f'Gesture: Thumbs Up')
                                    rotate_belt = []
                                    zoom_belt = []
                                    idx_belt = []
                                    mouse.init()
                                    if enable_keypress:
                                        pyautogui.press('b')
                                    sleepy_time = time.time() + 0.5

                                gesture_timer.toc()
                            # End of Gesture Mode

                            # Mouse Mode [Deprecated] hijacked by Browse Mode
                            if not gesture_mode:
                                mouse_timer.tic()
                                global which_series
                                if fc[1] == 0 and fc[2] == 1 and fc[3] == 1 and fc[4] == 1 and not sleeping():
                                    which_series = 0.3
                                    print("1 fing")
                                    logger.info(f'Gesture: one_finger_up')
                                    if scroll_mode == "scroll":
                                        pyautogui.moveTo(monitor_resolution[0]//3, monitor_resolution[1]//2, _pause=False)
                                    sleepy_time = time.time()

                                if fc[1] == 0 and fc[2] == 0 and fc[3] == 1 and fc[4] == 1 and not sleeping():
                                    which_series = 0.8
                                    print("2 fing")
                                    logger.info(f'Gesture: two_finger_up')
                                    if scroll_mode == "scroll":
                                        pyautogui.moveTo(2*monitor_resolution[0]//3, monitor_resolution[1]//2, _pause=False)
                                    sleepy_time = time.time()

                                if click_mode == "curl":
                                    click_state = curl_click(lm_list1)
                                else:
                                    click_state = thumb_click(lm_list1)

                                # Use scroll wheel commands
                                if scroll_mode == "scroll":
                                    mouse.update(lm_list1, click_state)
                                    mouse.scroll_drag(scroll_clicks=scroll_speed)
                                # Use click and drag commands
                                else:
                                    mouse.update2(lm_list1, click_state, which_series)
                                    mouse.move_mouse_to_abs_position()

                                if mouse.left_pressed is True:
                                    color_image = cv2.putText(color_image, 'Left Press', (1280 - 340, 0 + 150),
                                                              # from 280 to 340
                                                              cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 255, 0),
                                                              thickness=5)
                                mouse_timer.toc()

                                # Check Thumbs Up Gesture
                                if is_thumbs_up and is_curl and not sleeping():
                                    pyautogui.mouseUp()  ### new
                                    gesture_mode = True
                                    print("gesture mode now on")
                                    logger.info(f'Gesture: Thumbs Up')
                                    rotate_belt = []
                                    if enable_keypress:
                                        pyautogui.press('b')
                                    sleepy_time = time.time() + 0.5
                            # End of Mouse Mode

                    debug_timer.tic()
                    annotated_image = generate_overlay(color_image, results=results, fps=fps, depth_image=depth_image,
                                                       ptr_coords=ptr_coords, depth_coords=depth_coords,
                                                       gesture_mode=gesture_mode, gesture_txt=gesture_txt)
                    self.display_image(annotated_image)
                    debug_timer.toc()

                except:
                    pass

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
            mouse_timer.summary()
            debug_timer.summary()
            image_writer.stop()
            self.quit()
            QtWidgets.qApp.quit()


# GUI


if __name__ == '__main__':
    # Create log directories for current run
    run_dir = init_logging(log_dir)

    # Start threaded image writer
    image_writer = ThreadedImageWriter().start()

    # Camera
    camera = RealSenseCamera(width=1280, height=720, fps=30, enable_color=True, enable_depth=True,
                             enable_emitter=emitter)
    camera.start()

    # Squeeze config here ####

    # Window Initialization
    root = Tk()
    root.title("fruit salad")
    root.iconbitmap("media/hand.ico")


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

    gui_monitor_frame = LabelFrame(root, padx=143, pady=20)
    gui_monitor_frame.grid(row=2, column=0, padx=10, pady=10)

    # Buttons Widget; colors can be in HEX color codes
    # glove colors
    gui_glove_color_label = Label(gui_color_frame, text="Glove Color:")
    gui_glove_color_label.grid(row=0, column=0)

    gui_w_photo = PhotoImage(file="media/white_gloves.png")
    gui_w_resized_photo = gui_w_photo.subsample(8, 8)
    gui_w_imageButton = Button(gui_color_frame, text="White Gloves", image=gui_w_resized_photo, compound=LEFT,
                               command=lambda: gui_glove_color(
                                   "WHITE GLOVES"))
    gui_w_imageButton.grid(row=1, column=1)

    gui_g_photo = PhotoImage(file="media/green_gloves.png")
    gui_g_resized_photo = gui_g_photo.subsample(8, 8)
    gui_g_imageButton = Button(gui_color_frame, text="Green Gloves", image=gui_g_resized_photo, compound=LEFT,
                               command=lambda: gui_glove_color(
                                   "GREEN GLOVES"))
    gui_g_imageButton.grid(row=1, column=2)

    # handedness
    gui_dominant_hand_label = Label(gui_hand_frame, text="Dominant Hand:")
    gui_dominant_hand_label.grid(row=0, column=0)

    gui_right_button = Button(gui_hand_frame, text="Right Hand", command=lambda: gui_dominant_hand("RIGHT HAND"),
                              padx=40, pady=20, )
    gui_right_button.grid(row=1, column=2)

    gui_left_button = Button(gui_hand_frame, text="Left Hand", command=lambda: gui_dominant_hand("LEFT  HAND"), padx=40,
                             pady=20, )
    gui_left_button.grid(row=1, column=1)

    # Dropdown Menu
    # gui_monitor_label = Label(gui_monitor_frame, text="Monitor Resolution:")
    # gui_monitor_label.grid(row=0, column=0)


    # def gui_show():
    #     global monitor_resolution
    #     monitor_resolution = (int(gui_selected.get().split('x')[0]), int(gui_selected.get().split('x')[1]))
    #     reso_label = Label(gui_monitor_frame, text=gui_selected.get(), font=("Consolas", 11, "bold"))
    #     reso_label.grid(row=3, column=0)
    #
    #
    # gui_selected = StringVar()
    # gui_selected.set("1280 x 1024")
    #
    # gui_drop = OptionMenu(gui_monitor_frame, gui_selected, "1280 x 1024", " 1366 x 768 ", " 1600 x 900 ", "1920 x 1080",
    #                       "1920 x 1200", "2560 x 1440", "3440 x 1440", "3840 x 2160")
    # gui_drop.grid(row=1, column=0)
    #
    # gui_reso_button = Button(gui_monitor_frame, text="Confirm", command=gui_show).grid(row=2, column=0)

    # Quit
    gui_button_quit = Button(root, text="Begin Gesture Program", command=root.destroy)
    gui_button_quit.grid(row=4, column=0, pady=5)

    # Loop of window
    root.mainloop()
    ###
    print("program start")
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
    save_timer = Timer('save image')
    gesture_timer = Timer('gesture recognition')
    mouse_timer = Timer('mouse pointer')
    debug_timer = Timer('visual debugging')

    # Qt GUI
    app = QApplication([])
    main_window = MainWindow()
    main_window.setLayout(main_window.vbl)
    main_window.show()

    # Spawn QThread
    main_window.main_thread = MainThread()
    main_window.main_thread.start()
    main_window.main_thread.OverlayUpdate.connect(main_window.overlayUpdateCallback)

    # Main Program Loop
    app.exec_()
