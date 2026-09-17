import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from enum import Enum
import time
from datetime import datetime
import os

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

class MouseAction(Enum):
    NEUTRAL = "Neutral"
    MOVE = "Move"
    LEFT_CLICK = "Left Click"
    RIGHT_CLICK = "Right Click"
    DOUBLE_CLICK = "Double Click"
    SCROLL = "Scroll"
    DRAG = "Drag"
    SCREENSHOT = "Screenshot"

class HandGestureController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Screen settings
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # Movement smoothing
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 0.5

        # Action states
        self.current_action = MouseAction.NEUTRAL
        self.is_dragging = False
        self.last_click_time = 0
        self.scroll_start_y = None
        
        # Screenshot directory
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def get_finger_states(self, hand_landmarks):
        """Returns the state of each finger (True if raised)"""
        fingers = []
        # Thumb
        thumb_tip_x = hand_landmarks.landmark[4].x
        thumb_mcp_x = hand_landmarks.landmark[2].x
        fingers.append(thumb_tip_x < thumb_mcp_x)

        # Other fingers
        for tip in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky
            tip_y = hand_landmarks.landmark[tip].y
            pip_y = hand_landmarks.landmark[tip - 2].y
            fingers.append(tip_y < pip_y)

        return fingers

    def determine_action(self, fingers):
        """Determine the mouse action based on finger positions"""
        if fingers == [False, True, False, False, False]:  # Index only
            return MouseAction.MOVE
        elif fingers == [False, True, True, False, False]:  # Index + Middle
            return MouseAction.LEFT_CLICK
        elif fingers == [True, True, False, False, False]:  # Thumb + Index
            return MouseAction.RIGHT_CLICK
        elif fingers == [False, True, True, True, False]:  # Index + Middle + Ring
            return MouseAction.DOUBLE_CLICK
        elif fingers == [False, True, False, False, True]:  # Index + Pinky
            return MouseAction.SCROLL
        elif fingers == [True, True, True, False, False]:  # Thumb + Index + Middle
            return MouseAction.DRAG
        elif fingers == [False, False, False, True, True]:  # Ring + Pinky
            return MouseAction.SCREENSHOT
        return MouseAction.NEUTRAL

    def take_screenshot(self):
        """Take a screenshot and save it with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
        
        # Add visual feedback before taking screenshot
        pyautogui.sleep(0.5)  # Small delay to prepare
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"Screenshot saved: {filename}")
        
        # Add a small delay to prevent multiple screenshots
        time.sleep(1)

    def perform_action(self, action, hand_landmarks):
        """Execute the determined mouse action"""
        if action == MouseAction.MOVE or action == MouseAction.DRAG:
            index_x = hand_landmarks.landmark[8].x
            index_y = hand_landmarks.landmark[8].y
            
            screen_x = int(np.interp(index_x, [0, 1], [0, self.screen_width]))
            screen_y = int(np.interp(index_y, [0, 1], [0, self.screen_height]))
            
            smooth_x = int(self.prev_x + (screen_x - self.prev_x) * self.smoothing)
            smooth_y = int(self.prev_y + (screen_y - self.prev_y) * self.smoothing)
            
            self.prev_x, self.prev_y = smooth_x, smooth_y

            if action == MouseAction.DRAG:
                if not self.is_dragging:
                    pyautogui.mouseDown()
                    self.is_dragging = True
                pyautogui.moveTo(smooth_x, smooth_y)
            else:
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                pyautogui.moveTo(smooth_x, smooth_y)

        elif action == MouseAction.LEFT_CLICK:
            pyautogui.click()
            time.sleep(0.3)  # Prevent multiple clicks

        elif action == MouseAction.RIGHT_CLICK:
            pyautogui.rightClick()
            time.sleep(0.3)

        elif action == MouseAction.DOUBLE_CLICK:
            pyautogui.doubleClick()
            time.sleep(0.3)

        elif action == MouseAction.SCROLL:
            if self.scroll_start_y is None:
                self.scroll_start_y = hand_landmarks.landmark[8].y
            else:
                current_y = hand_landmarks.landmark[8].y
                scroll_amount = int((current_y - self.scroll_start_y) * 100)
                pyautogui.scroll(-scroll_amount)
                self.scroll_start_y = current_y

        elif action == MouseAction.SCREENSHOT:
            self.take_screenshot()

        else:  # NEUTRAL
            self.scroll_start_y = None
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False

    def run(self):
        """Main loop for the hand gesture controller"""
        while True:
            success, img = self.cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_img)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    fingers = self.get_finger_states(hand_landmarks)
                    action = self.determine_action(fingers)
                    self.perform_action(action, hand_landmarks)

                    # Display current action
                    cv2.putText(img, f"Action: {action.value}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Gesture Mouse Control', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()