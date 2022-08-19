import numpy as np



def lm_rel_to_abs(landmark, cam_res=(1280, 720)):
    px, py, pz = landmark
    px = int(px*cam_res[0])
    py = int(py*cam_res[1])
    pz = int(pz*cam_res[0])
    return [px, py, pz]


def belt_fill(lm_list1, ptr_coords, may_swipe, is_curl, may_rotate, idx_belt, zoom_belt, rotate_belt, FPS):

    if not is_curl and may_swipe:
        if len(idx_belt) < FPS:
            idx_belt.append(ptr_coords)
        else:
            idx_belt.pop(0)
            idx_belt.append(ptr_coords)

    else: # if is_curl:
        # conveyor belt for three (3) palm key points
        if len(rotate_belt) < FPS:  # [ [ [], [], [] ], [ [], [], [] ], [ [], [], [] ] ]
            rotate_belt.append([lm_rel_to_abs(lm_list1[5]), lm_rel_to_abs(lm_list1[0]), lm_rel_to_abs(lm_list1[17])])
        else:
            rotate_belt.pop(0)
            rotate_belt.append([lm_rel_to_abs(lm_list1[5]), lm_rel_to_abs(lm_list1[0]), lm_rel_to_abs(lm_list1[17])])

    if may_rotate:
        # conveyor belt for middle finger and thumb
        if len(zoom_belt) < FPS:  # [ [ [], [] ], [ [], [] ], [ [], [] ] ]
            zoom_belt.append([lm_rel_to_abs(lm_list1[12]), lm_rel_to_abs(lm_list1[4])])
        else:
            zoom_belt.pop(0)
            zoom_belt.append([lm_rel_to_abs(lm_list1[12]), lm_rel_to_abs(lm_list1[4])])



# Classifier
def gesture_flip(lm_list):
    thumb_extent = (lm_list[3][1]-lm_list[4][1]) + (lm_list[2][1]-lm_list[3][1]) + (lm_list[1][1]-lm_list[2][1]) + (lm_list[0][1]-lm_list[1][1]) 
    if lm_list[4][1] < lm_list[3][1] < lm_list[2][1] < lm_list[8][1] and lm_list[6][1] > lm_list[5][1] and thumb_extent > 0.14:
        return True
    else:
        return False


def angle_finder(i, j, k):
    a = np.array([i[0], i[1]])
    b = np.array([j[0], j[1]])
    c = np.array([k[0], k[1]])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def hand_curl(landmark_list):
    finger_curl = [0, 0, 0, 0, 0]
    angle0 = angle_finder(landmark_list[4], landmark_list[3], landmark_list[1])
    angle1 = angle_finder(landmark_list[8], landmark_list[6], landmark_list[5])
    angle2 = angle_finder(landmark_list[12], landmark_list[10], landmark_list[9])
    angle3 = angle_finder(landmark_list[16], landmark_list[14], landmark_list[13])
    angle4 = angle_finder(landmark_list[20], landmark_list[18], landmark_list[17])
    if angle0 < 150:
        finger_curl[0] = 1
    if angle1 < 120:
        finger_curl[1] = 1
    if angle2 < 120:
        finger_curl[2] = 1
    if angle3 < 120:
        finger_curl[3] = 1
    if angle4 < 150:
        finger_curl[4] = 1
    # angle_curl = [angle0, angle1, angle2, angle3, angle3]
    # print(angle_curl)
    return finger_curl


def swiped(idx_belt, thresh=[5, 1], vert_adj=16/9, up_adj=1):
    def compute_speed(idx_belt):
        vx, vy = 0.0, 0.0
        ax, ay = 0.0, 0.0
        for idx in range(1, len(idx_belt)-1):
            vx += idx_belt[idx+1][0] - idx_belt[idx][0]  # horizontal speed in pixels per frame
            vy += idx_belt[idx+1][1] - idx_belt[idx][1]  # vertical speed in pixels per frame
            ax += idx_belt[idx+1][0] + idx_belt[idx-1][0] - 2*idx_belt[idx][0]  # horizontal accel in pixels per frame^2
            ay += idx_belt[idx+1][1] + idx_belt[idx-1][1] - 2*idx_belt[idx][1]  # horizontal accel in pixels per frame^2
        vx /= len(idx_belt)-2
        vy /=  len(idx_belt)-2
        ax /=  len(idx_belt)-2
        ay /=  len(idx_belt)-2
        vy *= vert_adj  # vertical compensation
        ay *= vert_adj  # vertical compensation
        if vy < 0:  # swipe_up compensation
            vy *= up_adj
        # speed = (vx**2 + vy**2)**0.5
        speed = max(abs(vx), abs(vy))
        accel = max(abs(ax), abs(ay))
        return vx, vy, speed, ax, ay, accel

    vx, vy, speed, ax, ay, accel = compute_speed(idx_belt)
    print(f"vx={vx:.2f}, vy={vy:.2f}, speed={speed:.2f}, ax={ax:.2f}, ay={ay:.2f}, accel={accel:.2f}")

    if speed > thresh[0] and accel > thresh[1]:
        if abs(vy) > abs(vx):
            if vy < 0:
                return True, "Up"
            else:
                return True, "Down"
        else:
            if vx > 0: # assumes image is flipped
                return True, "Right"
            else:
                return True, "Left"
    else:
        return False, None



def zoomed(zoom_belt, thresh=3.5, out_adj=1.4):  # coords = [[x,y],[x,y]] ;; 94 px
    def compute_angular_speed(zoom_belt):
        w = 0.0
        for idx in range(len(zoom_belt)-1):
            middle0 = np.array([zoom_belt[idx][0][0], zoom_belt[idx][0][1]])
            thumb0 = np.array([zoom_belt[idx][1][0], zoom_belt[idx][1][1]])
            middle1 = np.array([zoom_belt[idx+1][0][0], zoom_belt[idx+1][0][1]])
            thumb1 = np.array([zoom_belt[idx+1][1][0], zoom_belt[idx+1][1][1]])
            phase0 = np.linalg.norm(middle0 - thumb0)
            phase1 = np.linalg.norm(middle1 - thumb1)
            w += phase1 - phase0
        w /= len(zoom_belt)-1
        if w < 0:  # zoom_out compensation
            w *= out_adj
        return w

    w = compute_angular_speed(zoom_belt)
    #print(f"w={w:.2f}")
    if abs(w) > thresh:
        if w > 0:
            return True, "zoom_in"
        else:
            return True, "zoom_out"
    # past_coords, new_coords = zoom_belt[0], zoom_belt[-1]
    # past_coords_idx = np.array((past_coords[0][0], past_coords[0][1]))
    # past_coords_thm = np.array((past_coords[1][0], past_coords[1][1]))
    # new_coords_idx = np.array((new_coords[0][0], new_coords[0][1]))
    # new_coords_thm = np.array((new_coords[1][0], new_coords[1][1]))
    # # print(np.linalg.norm(new_coords_idx - new_coords_thm))
    # if np.linalg.norm(past_coords_idx - past_coords_thm) < 30 and np.linalg.norm(new_coords_idx - new_coords_thm) > 94:
    #     return True, "zoom_in"
    # if np.linalg.norm(past_coords_idx - past_coords_thm) > 80 and np.linalg.norm(new_coords_idx - new_coords_thm) < 40:
    #     return True, "zoom_out"
    return False, None


def rotate(rotate_belt): # coords = [[x,y],[x,y], [x,y] ;; -inf to 20, 21 to 69, 70 to +inf
    past_coords, new_coords = rotate_belt[0], rotate_belt[-1]
    lm_5 = np.array([past_coords[0][0], past_coords[0][1]]) # 5
    lm_0 = np.array([past_coords[1][0], past_coords[1][1]]) # 0
    lm_17 = np.array([past_coords[2][0], past_coords[2][1]]) # 17
    lm_17_ = np.array([new_coords[2][0], new_coords[2][1]]) # 17 prime

    lm0_5 = lm_5 - lm_0
    lm0_17 = lm_17 - lm_0

    cos = np.dot(lm0_5, lm0_17) / (np.linalg.norm(lm0_5) * np.linalg.norm(lm0_17))
    angle = np.arccos(cos)

    lm0_17_ = lm_17_ - lm_0
    cos_ = np.dot(lm0_5, lm0_17_) / (np.linalg.norm(lm0_5) * np.linalg.norm(lm0_17_))
    angle_ = np.arccos(cos_)

    #print(np.degrees(angle), np.degrees(angle_))
    if np.degrees(angle_) > 80:
        return True, "rotate_cw"

    if np.degrees(angle) - np.degrees(angle_) > 20:
        return True, "rotate_ccw"
    else:
        return False, None



# Mouse Mode
def get_hand_landmarks_rel(results):
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


def thumb_click(landmark_list):
    if landmark_list is None:
        return False
    thumb_kp = landmark_list[4]
    left_kp = landmark_list[13]
    click_value = ( (left_kp[0]-thumb_kp[0])**2 + (left_kp[1]-thumb_kp[1])**2 ) ** 0.5
    return click_value < 0.03


def curl_click(landmark_list):
    fc = hand_curl(landmark_list)
    if fc[1] == 1 and fc[2] == 1 and fc[3] == 1 and fc[4] == 1:
        return True
    else:
        return False
