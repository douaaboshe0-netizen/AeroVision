import cv2 as cv
import numpy as np
from ultralytics import YOLO
import winsound 
import os  
import time  

model = YOLO('yolov8n.pt')
clicked_points = []
danger_center = None

classes_list = [0, 1, 2, 3, 5, 7]

det_imgsz = 960            
min_conf_person = 0.30   
min_conf_vehicle = 0.15   
max_box_area = 0.25        
min_seen_frames = 3      
seen_count = {}
camera_zone_as_circle = False  

def select_initial_points(event, x, y, flags, param):
    global clicked_points
    if event == cv.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"Point ({x},{y}) set.")

def select_target_object(event, x, y, flags, param):
    global danger_center
    if event == cv.EVENT_LBUTTONDOWN:
        img_pt = np.array([[[x, y]]], dtype=np.float32)
        transformed_pt = cv.perspectiveTransform(img_pt, H)
        p = transformed_pt[0][0]
        danger_center = np.array([float(p[0]), float(p[1])], dtype=int)
    elif event == cv.EVENT_RBUTTONDOWN:
        danger_center = None  

video_path = r"Videos/ramp.mp4"
if not os.path.exists(video_path) :
    print(f"File not found : {video_path}")
 
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("Error : cannot  be opened !")
    exit()

ret, first_frame = cap.read()
if not ret or first_frame is None :
    print("Error : unable to read the first frame of the video .")
    cap.release()
    exit()

orig_h, orig_w = first_frame.shape[:2]  
sx = 640.0 / orig_w
sy = 480.0 / orig_h
first_frame = cv.resize(first_frame, (640, 480))

cv.namedWindow("AeroVision - select points")
cv.setMouseCallback("AeroVision - select points", select_initial_points)

while len(clicked_points) < 4 :
    temp_frame = first_frame.copy()
    cv.putText(temp_frame, f"Select {len(clicked_points)+1} /4 point on ramp ground : ", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)

    for pt in clicked_points :
        cv.circle(temp_frame, tuple(pt), 6, (0, 0, 255), -1)
    cv.imshow("AeroVision - select points", temp_frame)
    if cv.waitKey(10) & 0xFF == ord('q') :
        break

cv.destroyWindow("AeroVision - select points")

if len(clicked_points) < 4 :
    print("Failed to select all 4 points .Turning off .")
    cap.release()
    exit()

source_points = np.float32(clicked_points)

width_bev, height_bev = 350, 480
distination_points = np.float32([[0, 0], [width_bev, 0], [width_bev, height_bev], [0, height_bev]])
H, _ = cv.findHomography(source_points, distination_points)
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

H_inv = np.linalg.inv(H)
_w_ref = (H_inv @ np.array([width_bev / 2, height_bev / 2, 1.0]))[2]
w_sign = 1.0 if _w_ref >= 0 else -1.0

def to_camera_pts(pts_bev) :
    pts = np.asarray(pts_bev, np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H_inv.T
    valid = (ph[:, 2] * w_sign) > 1e-6
    out = np.zeros((len(pts), 2))
    out[valid] = ph[valid, :2] / ph[valid, 2:3]
    return np.clip(out, -5000, 5000).astype(np.int32), valid

danger_radius = 80
frame_count = 0

fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

print(f"---Video Info---")
print(f"Resolution :{width}x{height}")
print(f"FPS :{fps}")

def no_thing(val) :
    pass

cv.namedWindow("AeroVision - main")
cv.createTrackbar("Danger Radius", "AeroVision - main", danger_radius, 200, no_thing)
cv.setMouseCallback("AeroVision - main", select_target_object)

REAL_WIDTH_METER = 36.0
REAL_HEIGHT_METER = 49.0   
METER_PER_PIXEL = REAL_WIDTH_METER / width_bev
METER_PER_PIXEL_Y = REAL_HEIGHT_METER / height_bev   

try :
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError :
    base_dir = os.getcwd()
tracker_cfg = os.path.join(base_dir, "bytetrack_low.yaml")
with open(tracker_cfg, "w") as f :
    f.write("tracker_type: bytetrack\ntrack_high_thresh: 0.15\ntrack_low_thresh: 0.05\nnew_track_thresh: 0.15\ntrack_buffer: 30\nmatch_thresh: 0.8\nfuse_score: True\n")

prev_min_distance = float('inf')
movement_trend = "NONE"
smoothed_min_distance = None

last_beep = 0
prev_hazard = False   

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video OR failed to read frame")
        break

    frame_count += 1
    orig_frame = frame  
    frame = cv.resize(frame, (640, 480))

    alpha = 1.2  
    beta = 25    
    frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

    bev_frame = cv.warpPerspective(frame, H, (width_bev, height_bev))
    dark_bev = np.full((height_bev, width_bev, 3), 35, dtype=np.uint8)
    grid_size = 40
    for x in range(0, width_bev, grid_size) :
        cv.line(dark_bev, (x, 0), (x, height_bev), (100, 100, 100), 1)
    for y in range(0, height_bev, grid_size) :
        cv.line(dark_bev, (0, y), (width_bev, y), (100, 100, 100), 1)
   
    danger_radius = cv.getTrackbarPos("Danger Radius", "AeroVision - main")
    if danger_radius < 5 :
        danger_radius = 5

    results = model.track(orig_frame, persist=True, conf=0.1, iou=0.5, imgsz=det_imgsz, agnostic_nms=True, classes=classes_list, tracker=tracker_cfg, verbose=False)
    if danger_center is not None :
        cam_zone_color = (0, 0, 255) if prev_hazard else (0, 255, 0)
        radius_m_text = danger_radius * METER_PER_PIXEL
        ang = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        circ = np.stack([danger_center[0] + danger_radius * np.cos(ang), danger_center[1] + danger_radius * np.sin(ang)], axis=1)
        circ_cam, circ_ok = to_camera_pts(circ)
        c_cam, c_ok = to_camera_pts(np.array([danger_center]))
        label_anchor = None
        if camera_zone_as_circle and circ_ok.all() and c_ok[0] :
            ccx, ccy = int(c_cam[0][0]), int(c_cam[0][1])
            rr = int(np.sqrt(cv.contourArea(circ_cam.reshape(-1, 1, 2)) / np.pi))
            zone_layer = frame.copy()
            cv.circle(zone_layer, (ccx, ccy), rr, cam_zone_color, -1)
            frame = cv.addWeighted(zone_layer, 0.2, frame, 0.8, 0)
            cv.circle(frame, (ccx, ccy), rr, cam_zone_color, 3)
            label_anchor = (ccx, ccy - rr)
        else :
            if circ_ok.all() :
                zone_layer = frame.copy()
                cv.fillPoly(zone_layer, [circ_cam], cam_zone_color)
                frame = cv.addWeighted(zone_layer, 0.2, frame, 0.8, 0)
                label_anchor = (int(circ_cam[np.argmin(circ_cam[:, 1])][0]), int(circ_cam[np.argmin(circ_cam[:, 1])][1]))
            for i in range(len(circ_cam)) :
                j = (i + 1) % len(circ_cam)
                if circ_ok[i] and circ_ok[j] :
                    cv.line(frame, (int(circ_cam[i][0]), int(circ_cam[i][1])), (int(circ_cam[j][0]), int(circ_cam[j][1])), cam_zone_color, 3)
        if c_ok[0] :
            cv.drawMarker(frame, (int(c_cam[0][0]), int(c_cam[0][1])), cam_zone_color, cv.MARKER_CROSS, 20, 2)
        if label_anchor is not None :
            zone_text = f"PROTECTED ZONE  R = {radius_m_text:.1f} m"
            (tw, th), _ = cv.getTextSize(zone_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx = int(min(max(label_anchor[0] - tw // 2, 0), 640 - tw - 4))
            ly = int(max(label_anchor[1] - 6, th + 8))
            cv.rectangle(frame, (lx, ly - th - 8), (lx + tw + 4, ly), cam_zone_color, -1)
            cv.putText(frame, zone_text, (lx + 2, ly - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    is_hazard = False

    min_distance = float('inf')
    min_distance_m = None   

    for result in results :
        for box in result.boxes :
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            x1 = int(bx1 * sx)
            y1 = int(by1 * sy)
            x2 = int(bx2 * sx)
            y2 = int(by2 * sy)

            track_id = int(box.id[0]) if box.id is not None else -1
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name == "airplane" :
                continue
            conf_val = float(box.conf[0])
            bw, bh = x2 - x1, y2 - y1
            if bw * bh > max_box_area * 640 * 480 :
                continue                                   
            if class_name == "person" :
                if conf_val < min_conf_person or bw > bh * 1.2 :
                    continue                              
            elif conf_val < min_conf_vehicle :
                continue
            if track_id >= 0 :                             
                seen_count[track_id] = seen_count.get(track_id, 0) + 1
                if seen_count[track_id] < min_seen_frames :
                    continue

            label_text = f"{class_name} #{track_id}" if track_id >= 0 else class_name

            foot_x = (x1 + x2) // 2
            foot_y = y2

            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
            cv.circle(frame, (foot_x, foot_y), 4, (0, 255, 255), -1)

            img_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            bev_point_transformed = cv.perspectiveTransform(img_point, H)
            object_bev_point = bev_point_transformed[0][0].astype(int)

            if danger_center is not None:
                dist = np.linalg.norm(object_bev_point - danger_center)
                d_vec = object_bev_point - danger_center
                obj_dist_m = float(np.hypot(d_vec[0] * METER_PER_PIXEL, d_vec[1] * METER_PER_PIXEL_Y))
                if dist < danger_radius:
                    is_hazard = True  
            else:
                dist = float('inf')
                obj_dist_m = None

            if dist < min_distance :
                min_distance = dist
                min_distance_m = obj_dist_m

            dot_color = (0, 255, 255) if (danger_center is not None and dist < danger_radius) else (0, 165, 255)
            cv.circle(dark_bev, tuple(object_bev_point), 6, dot_color, -1)
            cv.circle(dark_bev, tuple(object_bev_point), 7, (255, 255, 255), 1)
            cv.putText(dark_bev, label_text, (object_bev_point[0] + 8, object_bev_point[1] + 4), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            box_color = (0, 0, 255) if (danger_center is not None and dist < danger_radius) else (255, 150, 0)
            if danger_center is not None and dist < danger_radius :
                cv.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            label_cam = f"{label_text}  {obj_dist_m:.1f}m" if obj_dist_m is not None else label_text
            (tw, th), _ = cv.getTextSize(label_cam, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_bottom = max(y1, th + 8)
            cv.rectangle(frame, (x1, label_bottom - th - 8), (x1 + tw + 4, label_bottom), box_color, -1)
            cv.putText(frame, label_cam, (x1 + 2, label_bottom - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    circle_zone_color = (0, 0, 255) if is_hazard else (0, 255, 0)       
    prev_hazard = is_hazard   

    if min_distance != float('inf') :
        if smoothed_min_distance is None :
            smoothed_min_distance = min_distance
        else :
            smoothed_min_distance = (0.15 * min_distance) + (0.85 * smoothed_min_distance)

        distance_meters = min_distance_m   
        if prev_min_distance == float('inf') :
            movement_trend = "STATIONARY"
        else :
            diff = smoothed_min_distance - prev_min_distance

            if diff < - 1.2 :
                movement_trend = "APPROACHING"
            elif diff > 1.2 :
                movement_trend = "RECEDING"
            else :
                movement_trend = "STATIONARY"

        prev_min_distance = smoothed_min_distance
    else :
        distance_meters = None
        movement_trend = "NONE"
        prev_min_distance = float('inf')
        smoothed_min_distance = None

    pip_w, pip_h = 120, 160
    pip_resized = cv.resize(bev_frame, (pip_w, pip_h))
    x1 = 640 - pip_w - 10
    y1 = 10
    x2 = x1 + pip_w
    y2 = y1 + pip_h

    frame[y1:y2, x1:x2] = pip_resized
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    status_color = (0, 0, 255) if is_hazard else (0, 255, 0)
    alert_text = "HAZARD DETECTED !" if is_hazard else "Ramp zone : clear"
    
    if is_hazard :
        if time.time() - last_beep > 0.3 :   
            winsound.Beep(1200, 150)
            last_beep = time.time()

    if danger_center is not None:
        cx, cy = danger_center[0], danger_center[1]
        cv.line(dark_bev, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 2)
        cv.line(dark_bev, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 2)

        overlay_bev = dark_bev.copy()
        cv.circle(overlay_bev, tuple(danger_center), danger_radius, circle_zone_color, -1)
        cv.addWeighted(overlay_bev, 0.25, dark_bev, 0.75, 0, dark_bev)
        cv.circle(dark_bev, tuple(danger_center), danger_radius, circle_zone_color, 2)
        cv.putText(dark_bev, f"R = {danger_radius * METER_PER_PIXEL:.1f} m", (int(cx) - 35, int(max(15, cy - danger_radius - 8))), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    dashboard = np.zeros((480, 300, 3), dtype=np.uint8) 
    cv.putText(dashboard, "AEROVISION MONITOR ", (15, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.putText(dashboard, f"Frames: {frame_count}", (25, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
    cv.putText(dashboard, f"Radius : {danger_radius * METER_PER_PIXEL:.1f} m", (25, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if distance_meters is not None :
        display_distance_text = f"{distance_meters :.2f} m ({int(min_distance)} px )"
    else :
        display_distance_text = "NOT AVAILABLE"

    trend_color = (0, 0, 255) if movement_trend == "APPROACHING" else (0, 255, 255)
    cv.putText(dashboard, f"Trend : {movement_trend}", (25, 200), cv.FONT_HERSHEY_SIMPLEX, 0.5, trend_color, 1)
    cv.putText(dashboard, f"Distance : {display_distance_text}", (25, 160), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(dashboard, alert_text, (25, 240), cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv.putText(dashboard, "SPACE = pause | Q = quit", (25, 460), cv.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)   

    combined_display = np.hstack((frame, dark_bev, dashboard))

    cv.imshow("AeroVision - main", combined_display)

    key = cv.waitKey(25) & 0xFF
    if key == ord('q') :
        break
    elif key == ord(' ') :   
        while True :
            key2 = cv.waitKey(0) & 0xFF
            if key2 == ord(' ') or key2 == ord('q') :
                break
        if key2 == ord('q') :
            break

cap.release()
cv.destroyAllWindows()