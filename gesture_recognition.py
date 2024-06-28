# gesture_recognition.py
import collections
import numpy as np

class GestureRecognizer:
    def __init__(self, history_length=20, gesture_threshold=15):
        self.history_length = history_length
        self.gesture_threshold = gesture_threshold
        self.object_movements = collections.defaultdict(list)
        self.gesture_actions = {
            "wave": self.wave_detected,
            "clap": self.clap_detected
        }

    def update_movements(self, object_id, point):
        self.object_movements[object_id].append(point)
        if len(self.object_movements[object_id]) > self.history_length:
            self.object_movements[object_id].pop(0)

    def recognize_gestures(self):
        for object_id, movements in self.object_movements.items():
            if self.is_wave(movements):
                self.gesture_actions["wave"](object_id)
            elif self.is_clap(movements):
                self.gesture_actions["clap"](object_id)

    def is_wave(self, movements):
        if len(movements) < self.history_length:
            return False
        y_diff = [abs(movements[i][1] - movements[i+1][1]) for i in range(len(movements)-1)]
        return sum(y_diff) > self.gesture_threshold

    def is_clap(self, movements):
        if len(movements) < self.history_length:
            return False
        x_diff = [abs(movements[i][0] - movements[i+1][0]) for i in range(len(movements)-1)]
        return sum(x_diff) < self.gesture_threshold / 2

    def wave_detected(self, object_id):
        print(f"Wave gesture detected for {object_id}!")

    def clap_detected(self, object_id):
        print(f"Clap gesture detected for {object_id}!")
