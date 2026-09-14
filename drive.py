import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from lelib import doubleMotor

CARD_SERIAL = "0999"
MODEL = "gesture_recognizer.task"

speed = 60
SPEED_STEP = 10
SPEED_REPEAT = 0.4   # seconds between bumps while holding the gesture
MIN_SPEED, MAX_SPEED = 10, 100
TURN_RATIO = 0.0     # inner wheel as a fraction of speed. 0 = arc, -1 = spin

RIGHT_FLIP = -1
LEFT_FLIP = 1
HOLD_FRAMES = 3
SEND_INTERVAL = 0.1
LOST_TIMEOUT = 1.5
SWAP_HANDS = False


def speeds_for(cmd, spd):
    inner = int(spd * TURN_RATIO)
    return {
        "FORWARD":  ( spd,  spd),
        "BACKWARD": (-spd, -spd),
        "LEFT":     ( inner,  spd),
        "RIGHT":    ( spd,  inner),
        "STOP":     (0, 0),
    }[cmd]


def classify(hands):
    left = hands.get("Left")
    right = hands.get("Right")

    # speed modifiers
    if left == "Pointing_Up" and right == "Pointing_Up":
        return "FASTER"
    if left == "Victory" and right == "Victory":
        return "SLOWER"

    # drive commands
    if left == "Closed_Fist" and right == "Closed_Fist":
        return "STOP"
    if left == "Thumb_Down" and right == "Thumb_Down":
        return "FORWARD"
    if left == "Open_Palm" and right == "Open_Palm":
        return "BACKWARD"
    if right == "Open_Palm" and left != "Open_Palm":
        return "RIGHT"
    if left == "Open_Palm" and right != "Open_Palm":
        return "LEFT"
    return None


cams = []
for i in range(2):
    c = cv2.VideoCapture(i)
    if c.isOpened():
        cams.append(i)
        c.release()
print("Available cameras:", cams)
cap = cv2.VideoCapture(int(input("Which camera index? ")))

recognizer = vision.GestureRecognizer.create_from_options(
    vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        num_hands=2,
    )
)

dm = doubleMotor()
print(f"Connecting to double motor {CARD_SERIAL}...")
dm.connect(card_serial=CARD_SERIAL)
print("Connected. q to quit.")

candidate, streak, command = "STOP", 0, "STOP"
last_seen = time.time()
last_speed_change = 0.0
last_send, last_cmd = 0.0, None

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        hands = {}
        for gesture, handed in zip(result.gestures, result.handedness):
            side = handed[0].category_name
            if SWAP_HANDS:
                side = "Left" if side == "Right" else "Right"
            hands[side] = gesture[0].category_name

        detected = None
        if hands:
            detected = classify(hands)
            last_seen = now

        if detected is not None:
            if detected == candidate:
                streak += 1
            else:
                candidate, streak = detected, 1

            if streak >= HOLD_FRAMES:
                if candidate in ("FASTER", "SLOWER"):
                    if now - last_speed_change > SPEED_REPEAT:
                        if candidate == "FASTER":
                            speed = min(MAX_SPEED, speed + SPEED_STEP)
                        else:
                            speed = max(MIN_SPEED, speed - SPEED_STEP)
                        last_speed_change = now
                        last_cmd = None
                else:
                    command = candidate

        if now - last_seen > LOST_TIMEOUT:
            command = "STOP"

        cv2.putText(frame, command, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
        cv2.putText(frame, f"SPEED {speed}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"L: {hands.get('Left', '-')}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"R: {hands.get('Right', '-')}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        left, right = speeds_for(command, speed)
        if now - last_send > SEND_INTERVAL and (left, right) != last_cmd:
            dm.set_speed_left(LEFT_FLIP * left)
            dm.set_speed_right(RIGHT_FLIP * right)
            dm.run_left()
            dm.run_right()
            last_send, last_cmd = now, (left, right)

        cv2.imshow("Gesture drive - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            speed = min(MAX_SPEED, speed + SPEED_STEP)
            last_cmd = None
        elif key in (ord("-"), ord("_")):
            speed = max(MIN_SPEED, speed - SPEED_STEP)
            last_cmd = None
finally:
    dm.stop()
    dm.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")