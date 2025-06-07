import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import datetime
import os

# 동영상 선택
def select_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video file",
        filetypes=[("Video files", "*.mp4 *.avi *.webm *.mov")]
    )
    return file_path

# 모자이크
def apply_mosaic(img, x1, y1, x2, y2, ratio=0.07):
    sub_img = img[y1:y2, x1:x2]
    h, w = sub_img.shape[:2]
    if h == 0 or w == 0:
        return img
    mosaic = cv2.resize(sub_img, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic
    return img

paused = False
recording = False
writer = None
warned_seek_during_recording = False
warning_display_time = 0
button_regions = {}

# 마우스
def on_mouse(event, x, y, flags, param):
    global paused, cap, fps, warning_display_time, warned_seek_during_recording, recording

    if event == cv2.EVENT_LBUTTONDOWN:
        for key, (x1, y1, x2, y2) in button_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if recording:
                    warned_seek_during_recording = True
                    warning_display_time = 60
                    return
                if key == "Pause":
                    paused = not paused
                elif key == "Forward":
                    current = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current + 3 * fps)
                elif key == "Backward":
                    current = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - 3 * fps))

# 동영상
video_path = select_video_file()
if not video_path or not os.path.isfile(video_path):
    print("Wrong video file")
    exit()

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

yolo_model = YOLO("yolov8n.pt")

cv2.namedWindow("YOLOv8 - Face Mosaic with Timebar")

def on_trackbar(pos):
    global recording, warning_display_time, warned_seek_during_recording
    if recording:
        if not warned_seek_during_recording:
            warned_seek_during_recording = True
            warning_display_time = 60 
        cv2.setTrackbarPos("Time", "YOLOv8 - Face Mosaic with Timebar", int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        # 시간 표시
        current_time = current_frame / fps
        total_time = total_frames / fps
        time_label = f"{int(current_time // 60):02}:{int(current_time % 60):02} / {int(total_time // 60):02}:{int(total_time % 60):02}"
        cv2.putText(frame, time_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

cv2.createTrackbar("Time", "YOLOv8 - Face Mosaic with Timebar", 0, total_frames - 1, lambda pos: on_trackbar(pos))
cv2.setMouseCallback("YOLOv8 - Face Mosaic with Timebar", on_mouse)

# Main
frame = None
while True:
    if not paused or frame is None:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES))

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Time", "YOLOv8 - Face Mosaic with Timebar", current_frame)
    # 시간 표시
    current_time = current_frame / fps
    total_time = total_frames / fps
    time_label = f"{int(current_time // 60):02}:{int(current_time % 60):02} / {int(total_time // 60):02}:{int(total_time % 60):02}"
    
    cv2.putText(frame, time_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    results = yolo_model(frame, verbose=False)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if yolo_model.names[cls] == "person" and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height = y2 - y1
            fx1, fy1 = x1, int(y1 + height * 0.01)
            fx2, fy2 = x2, int(y1 + height * 0.2)
            fx1, fy1 = max(fx1, 0), max(fy1, 0)
            fx2, fy2 = min(fx2, frame.shape[1]), min(fy2, frame.shape[0])
            frame = apply_mosaic(frame, fx1, fy1, fx2, fy2)
           
    if recording and writer:
        writer.write(frame)

    status = "Recording..." if recording else "Press [Space] to record"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if recording else (255, 255, 255), 2)

    if warning_display_time > 0:
        cv2.putText(frame, "You can't set time when video is recording",
                    (50, frame.shape[0] - 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        warning_display_time -= 1

    # 버튼 표시
    h, w = frame.shape[:2]
    btn_w, btn_h = 100, 40
    margin = 10
    button_regions.clear()

    pause_rect = (margin, h - btn_h - margin, margin + btn_w, h - margin)
    cv2.rectangle(frame, pause_rect[:2], pause_rect[2:], (50, 50, 50), -1)
    cv2.putText(frame, "Pause" if not paused else "Play", (pause_rect[0]+10, pause_rect[1]+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    button_regions["Pause"] = pause_rect

    back_rect = (pause_rect[2] + margin, pause_rect[1], pause_rect[2] + margin + btn_w, pause_rect[3])
    cv2.rectangle(frame, back_rect[:2], back_rect[2:], (80, 80, 80), -1)
    cv2.putText(frame, "-3s", (back_rect[0]+25, back_rect[1]+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    button_regions["Backward"] = back_rect

    fwd_rect = (back_rect[2] + margin, back_rect[1], back_rect[2] + margin + btn_w, back_rect[3])
    cv2.rectangle(frame, fwd_rect[:2], fwd_rect[2:], (80, 80, 80), -1)
    cv2.putText(frame, "+3s", (fwd_rect[0]+25, fwd_rect[1]+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    button_regions["Forward"] = fwd_rect

    cv2.imshow("YOLOv8 - Face Mosaic with Timebar", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Space
        if paused:
            cv2.putText(frame, "You can't record when video is paused",
                        (50, frame.shape[0] - 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            continue
        recording = not recording
        if recording:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{now}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filename, fourcc, fps, (frame.shape[1], frame.shape[0]))
            print(f"녹화 시작: {filename}")
        else:
            if writer:
                writer.release()
                writer = None
                print("녹화 중지")
    elif key != 255 and recording:
        warned_seek_during_recording = True
        warning_display_time = 60  
        
if writer:
    writer.release()
cap.release()
cv2.destroyAllWindows()