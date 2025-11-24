# core.py
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import time
import os


class ParkingSecuritySystem:
    """Sistema de seguridad para detección de manipulación de vehículos"""

    def __init__(
        self,
        model_path="yolov8l.pt",
        confidence=0.5,
        proximity_threshold=100,
        loitering_time_threshold=5,
    ):
        self.model = YOLO(model_path)

        # Activar tracker
        self.use_tracker = True
        self.tracker_config = "bytetrack.yaml"

        self.confidence = confidence

        # Clases COCO
        self.vehicle_classes = [2, 3, 5, 7]
        self.person_class = 0

        self.proximity_threshold = proximity_threshold
        self.loitering_time_threshold = loitering_time_threshold

        self.suspicious_events = []
        self.last_detection_time = defaultdict(float)
        self.alert_triggered = set()

    # ======================
    #  IOU FUNCTION
    # ======================
    def bbox_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        if boxAArea == 0 or boxBArea == 0:
            return 0
        return interArea / float(boxAArea + boxBArea - interArea)

    # ======================
    # DETECCIÓN DE ACTIVIDAD
    # ======================
    def detect_suspicious_activity(self, persons, vehicles, current_time):
        suspicious = []

        for person_id, person_box in persons:
            for vehicle_id, vehicle_box in vehicles:

                key = f"{person_id}-{vehicle_id}"

                # IOU real
                iou = self.bbox_iou(person_box, vehicle_box)

                # Área expandida del vehículo
                vx1, vy1, vx2, vy2 = vehicle_box
                expanded = [
                    vx1 - 40, vy1 - 40,
                    vx2 + 40, vy2 + 40
                ]
                proximity = self.bbox_iou(person_box, expanded)

                if iou > 0.02 or proximity > 0.1:

                    if key not in self.last_detection_time:
                        self.last_detection_time[key] = current_time

                    time_near = current_time - self.last_detection_time[key]

                    if time_near > self.loitering_time_threshold and key not in self.alert_triggered:
                        suspicious.append({
                            "pair_key": key,
                            "person_box": person_box,
                            "vehicle_box": vehicle_box,
                            "duration": time_near,
                            "iou": iou,
                        })
                        self.alert_triggered.add(key)

                else:
                    # Persistencia de 2s
                    if key in self.last_detection_time:
                        if current_time - self.last_detection_time[key] > 2:
                            del self.last_detection_time[key]
                            if key in self.alert_triggered:
                                self.alert_triggered.remove(key)

        return suspicious  # ← CORRECTO AQUÍ

    # ======================
    # DRAW DETECTIONS
    # ======================
    def draw_detections(self, frame, results, suspicious_activities):
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                if cls == self.person_class:
                    color = (255, 255, 0)
                    label = f"Persona {track_id}"
                elif cls in self.vehicle_classes:
                    color = (0, 255, 0)
                    label = f"Vehículo {track_id}"
                else:
                    continue

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for activity in suspicious_activities:
            person_box = activity["person_box"]
            vehicle_box = activity["vehicle_box"]
            duration = activity["duration"]

            x1, y1, x2, y2 = map(int, person_box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            alert_text = f"ALERTA! {duration:.1f}s cerca del vehículo"
            cv2.putText(annotated_frame, alert_text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated_frame

    # ======================
    # PROCESS FRAME
    # ======================
    def process_frame(self, frame, current_time):

        if self.use_tracker:
            results = self.model.track(
                frame,
                conf=self.confidence,
                iou=0.45,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
        else:
            results = self.model(frame, conf=self.confidence, verbose=False)

        persons = []
        vehicles = []

        for result in results:
            if not hasattr(result, "boxes"):
                continue
            for box in result.boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue

                cls = int(box.cls[0])
                bbox = box.xyxy[0].cpu().numpy()

                if cls == self.person_class:
                    persons.append((track_id, bbox))
                elif cls in self.vehicle_classes:
                    vehicles.append((track_id, bbox))

        suspicious = self.detect_suspicious_activity(persons, vehicles, current_time)

        if suspicious:
            for event in suspicious:
                self.suspicious_events.append({
                    "timestamp": datetime.now().isoformat(),
                    "duration": float(event["duration"]),
                    "iou": float(event["iou"])
                })

        annotated_frame = self.draw_detections(frame, results, suspicious)
        return annotated_frame, len(suspicious)
