'''
Turns a MediaPipe GestureRecognizer result into a drive command.
'''

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

SWAP_HANDS = False


def create_recognizer(model_path, num_hands=2):
    return vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=num_hands,
        )
    )


def recognize(recognizer, frame_rgb):
    return recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))


def hands_from_result(result, swap_hands=SWAP_HANDS):
    hands = {}
    for gesture, handed in zip(result.gestures, result.handedness):
        side = handed[0].category_name
        if swap_hands:
            side = "Left" if side == "Right" else "Right"
        hands[side] = gesture[0].category_name
    return hands


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
