import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(r"C:\Users\Hari ragav K.S\New_folder\optiway ai\video assets\rlv_per1.mp4")

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

STOP_LINE_Y = 270
violating_vehicles = set()
object_positions = {}
vehicle_labels = {2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}

# HSV Ranges
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
lower_yellow = np.array([15, 80, 80])
upper_yellow = np.array([35, 255, 255])

zebra_contour = None
previous_signal_status = "UNKNOWN"
frame_counter = 0

def detect_zebra_crossing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea, default=None)
    return largest

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7, 9])

    signal_status = "UNKNOWN"
    traffic_light_box = None

    if results[0].boxes.cls is not None:
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box, cls_id in zip(boxes, classes):
            if cls_id == 9:
                x1, y1, x2, y2 = map(int, box)
                traffic_light_box = (x1, y1, x2, y2)
                break

    if traffic_light_box:
        x1, y1, x2, y2 = traffic_light_box
        signal_crop = frame[y1:y2, x1:x2]

        if signal_crop.size > 0:
            hsv = cv2.cvtColor(signal_crop, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv, lower_red1, upper_red1),
                cv2.inRange(hsv, lower_red2, upper_red2)
            )
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            red_pixels = cv2.countNonZero(red_mask)
            yellow_pixels = cv2.countNonZero(yellow_mask)
            total_pixels = signal_crop.shape[0] * signal_crop.shape[1]

            red_ratio = red_pixels / total_pixels
            yellow_ratio = yellow_pixels / total_pixels

            if red_ratio > 0.06:
                signal_status = "RED"
            elif yellow_ratio > 0.025:
                signal_status = "YELLOW"
            else:
                signal_status = "GREEN"

            previous_signal_status = signal_status

            color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[signal_status], 2)
            cv2.putText(frame, signal_status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color_map[signal_status], 2)

    # Draw stop line
    cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), (0, 0, 255), 2)

    # Detect zebra crossing once
    if zebra_contour is None:
        zebra_contour = detect_zebra_crossing(frame)

    zebra_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    if zebra_contour is not None:
        # Removed visual drawing of zebra crossing here to avoid unwanted overlay
        cv2.drawContours(zebra_mask, [zebra_contour], -1, 255, -1)

    # Vehicle detection & violation logic
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, obj_id, cls_id in zip(boxes, ids, classes):
            if cls_id not in vehicle_labels:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bottom_y = y2
            curr_pos = (cx, cy)

            moved = False
            direction = None
            if obj_id in object_positions:
                prev_x, prev_y = object_positions[obj_id]
                moved = abs(cy - prev_y) > 2 or abs(cx - prev_x) > 2
                direction = "right_to_left" if cx < prev_x else "left_to_right"
            object_positions[obj_id] = curr_pos

            label_name = vehicle_labels[cls_id]
            is_violation = False

            if signal_status == "RED" and moved and direction == "right_to_left":
                if bottom_y >= STOP_LINE_Y:
                    is_violation = True
                elif zebra_contour is not None:
                    vehicle_mask = np.zeros_like(zebra_mask)
                    cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1)
                    overlap = cv2.bitwise_and(zebra_mask, vehicle_mask)
                    if cv2.countNonZero(overlap) > 50:
                        is_violation = True

            if is_violation:
                violating_vehicles.add(obj_id)
                label_name += " - Violation"

            color = (0, 0, 255) if obj_id in violating_vehicles else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Final overlay text
    cv2.putText(frame, f"Violation Count: {len(violating_vehicles)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Red Light Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Vehicles that violated the red signal:", violating_vehicles)
