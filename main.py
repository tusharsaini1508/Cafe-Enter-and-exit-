import cv2
import csv
import os
import sys
import time
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "yolov8n.pt"  # Using yolov8n as requested
CSV_PATH = os.path.join(os.getcwd(), "entry_data.csv")
INFERENCE_WIDTH = 640  # Resize width for faster inference

# --- Global variable for line selection ---
line_points = []

def draw_line_callback(event, x, y, flags, param):
    """
    Mouse callback function to select two points for the counting line.
    """
    global line_points
    frame = param['frame']

    if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
        line_points.append((x, y))
        print(f"Point {len(line_points)} selected: {(x, y)}")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    elif len(line_points) == 2:
        cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)


def select_counting_line(frame):
    """
    Displays the first frame and lets the user select the counting line.
    Returns (LINE_X, LINE_Y1, LINE_Y2) or None if user quits.
    """
    global line_points
    line_points = []  # Reset points for each call
    
    temp_frame = frame.copy()
    cv2.namedWindow("Select Counting Line")
    cv2.setMouseCallback("Select Counting Line", draw_line_callback, {'frame': temp_frame})

    print("--- Line Selection ---")
    print("Click two points on the frame to define the counting line.")
    print("Press 'c' to confirm.")
    print("Press 'r' to reset points.")
    print("Press 'q' to quit.")

    while True:
        display_frame = temp_frame.copy()
        
        if len(line_points) == 1:
            cv2.circle(display_frame, line_points[0], 5, (0, 255, 0), -1)
        elif len(line_points) == 2:
            cv2.line(display_frame, line_points[0], line_points[1], (0, 0, 255), 2)
            cv2.putText(display_frame, "Press 'c' to confirm, 'r' to reset", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Select Counting Line", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('r'):
            line_points = []  # Reset points
            temp_frame = frame.copy() # Reset frame
            print("Points reset. Please select again.")
        elif key == ord('c') and len(line_points) == 2:
            # Use the average X for the vertical line
            line_x = int((line_points[0][0] + line_points[1][0]) / 2)
            y1 = min(line_points[0][1], line_points[1][1])
            y2 = max(line_points[0][1], line_points[1][1])
            
            print(f"Line confirmed at X={line_x} (from Y={y1} to Y={y2})")
            cv2.destroyAllWindows()
            return (line_x, y1, y2)

def get_video_source():
    """
    Prompts the user to choose between webcam or video file.
    Returns the source (0 for webcam, or file path string).
    """
    print("Select video source:")
    print("  1: Live Webcam")
    print("  2: Video File")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        return 0  # Webcam index
    elif choice == '2':
        path = input("Enter the full path to your video file: ").strip()
        if not os.path.exists(path):
            print(f"Error: File not found at '{path}'")
            sys.exit(1)
        return path
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

def log_event(csv_writer, events_deque, track_id, direction, prev_cx, curr_cx):
    """
    Logs a crossing event to the CSV file and the recent events deque.
    """
    ts = datetime.now().isoformat()
    events_deque.appendleft(f"{datetime.now().strftime('%H:%M:%S')} {direction} ID:{track_id}")
    try:
        csv_writer.writerow([ts, track_id, direction, prev_cx, curr_cx])
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def main():
    # 1. Get Video Source
    video_source = get_video_source()

    # 2. Initialize Model and Video Capture
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model '{MODEL_PATH}': {e}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        sys.exit(1)

    # 3. Select Counting Line
    r, first_frame = cap.read()
    if not r:
        print("Error: Could not read first frame from video source.")
        cap.release()
        sys.exit(1)

    line_coords = select_counting_line(first_frame)
    if line_coords is None:
        print("Line selection cancelled. Exiting.")
        cap.release()
        return
    
    LINE_X, LINE_Y1, LINE_Y2 = line_coords
    
    # Reset video capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 4. Initialize Tracking State
    track_history = {}  # id -> previous center x
    in_count = 0
    out_count = 0
    recent_events = deque(maxlen=10)  # Keep recent events for UI

    # For FPS measurement
    last_time = time.time()
    frame_count_fps = 0
    current_fps = 0.0

    # 5. Open CSV File and Process Video
    file_exists = os.path.exists(CSV_PATH)
    try:
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            # Write header only if the file is new
            if not file_exists or os.path.getsize(CSV_PATH) == 0:
                csv_writer.writerow(["timestamp", "id", "direction", "prev_cx", "curr_cx"])

            # --- Main Processing Loop ---
            while True:
                r, frame = cap.read()
                if not r:
                    print("End of video stream.")
                    break

                orig_h, orig_w = frame.shape[:2]

                # Compute resize scale and resize for faster inference
                scale = INFERENCE_WIDTH / float(orig_w)
                if scale < 1.0:
                    small = cv2.resize(frame, (INFERENCE_WIDTH, int(orig_h * scale)))
                else:
                    small = frame.copy()
                    scale = 1.0

                # Run tracking (classes=[0] -> person)
                results = model.track(small, persist=True, conf=0.5, classes=[0], verbose=False)

                # Process results
                for rdet in results:
                    for box in rdet.boxes:
                        if box.id is None:  # Skip if no track ID
                            continue
                        
                        xy = box.xyxy[0]
                        x1, y1_box, x2, y2_box = int(xy[0].item()), int(xy[1].item()), int(xy[2].item()), int(xy[3].item())

                        # Scale coordinates back to original frame
                        if scale != 1.0:
                            x1, y1_box, x2, y2_box = [int(v / scale) for v in [x1, y1_box, x2, y2_box]]

                        cx = int((x1 + x2) / 2)
                        cy = int((y1_box + y2_box) / 2)
                        track_id = int(box.id.item())

                        # Draw box and id
                        cv2.rectangle(frame, (x1, y1_box), (x2, y2_box), (0, 200, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                        # Initialize history if new track
                        if track_id not in track_history:
                            track_history[track_id] = cx
                            continue

                        prev_cx = track_history[track_id]
                        track_history[track_id] = cx

                        # Detect crossing
                        # Crossing left->right: prev < LINE_X <= cx => IN
                        if prev_cx < LINE_X and cx >= LINE_X:
                            in_count += 1
                            log_event(csv_writer, recent_events, track_id, 'IN', prev_cx, cx)
                        # Crossing right->left: prev > LINE_X and cx <= LINE_X => OUT
                        elif prev_cx > LINE_X and cx <= LINE_X:
                            out_count += 1
                            log_event(csv_writer, recent_events, track_id, 'OUT', prev_cx, cx)

                # --- Draw UI ---
                panel_w = 320
                cv2.line(frame, (LINE_X, LINE_Y1), (LINE_X, LINE_Y2), (255, 0, 255), 3)

                # Right-side panel
                h_frame, w_frame = frame.shape[:2]
                cv2.rectangle(frame, (w_frame - panel_w, 0), (w_frame, h_frame), (30, 30, 30), -1)
                cv2.putText(frame, f"IN: {in_count}", (w_frame - panel_w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                cv2.putText(frame, f"OUT: {out_count}", (w_frame - panel_w + 20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,120,255), 3)

                # FPS
                frame_count_fps += 1
                if time.time() - last_time >= 1.0: # Update every second
                    current_fps = frame_count_fps / (time.time() - last_time)
                    last_time = time.time()
                    frame_count_fps = 0
                
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (w_frame - panel_w + 20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

                # Recent events
                y0 = 210
                for i, ev in enumerate(list(recent_events)[:8]):
                    cv2.putText(frame, ev, (w_frame - panel_w + 10, y0 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)

                # Show frame
                cv2.imshow("People IN/OUT Counter (q=quit, r=reset)", frame)

                # Press 'q' to quit, 'r' to reset counts
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('r'):
                    print("Resetting counts...")
                    in_count = 0
                    out_count = 0
                    track_history.clear()
                    recent_events.clear()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()
