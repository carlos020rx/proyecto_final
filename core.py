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
        model_path="yolov8n.pt",
        confidence=0.5,
        proximity_threshold=100,
        loitering_time_threshold=5,
    ):
        # Carga del modelo YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence

        # Clases COCO: car=2, motorcycle=3, bus=5, truck=7, person=0
        self.vehicle_classes = [2, 3, 5, 7]
        self.person_class = 0

        # Parámetros de lógica
        self.proximity_threshold = proximity_threshold
        self.loitering_time_threshold = loitering_time_threshold

        # Estructuras de seguimiento
        self.person_vehicle_proximity = defaultdict(list)
        self.suspicious_events = []
        self.last_detection_time = defaultdict(float)

        # 👉 Para saber qué pares persona-vehículo ya dispararon alerta
        self.alert_triggered = set()

    def calculate_distance(self, box1, box2):
        x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def detect_suspicious_activity(self, persons, vehicles, current_time):
        """
        Devuelve SOLO las alertas NUEVAS de este frame.
        Cada par persona-vehículo genera una única alerta, cuando supera el tiempo umbral.
        """
        suspicious = []

        for person_id, person_box in enumerate(persons):
            for vehicle_id, vehicle_box in enumerate(vehicles):
                distance = self.calculate_distance(person_box, vehicle_box)

                if distance < self.proximity_threshold:
                    key = f"p{person_id}_v{vehicle_id}"

                    # Si es la primera vez que vemos cerca este par, guardamos el tiempo
                    if key not in self.last_detection_time:
                        self.last_detection_time[key] = current_time

                    time_near_vehicle = current_time - self.last_detection_time[key]

                    # Solo disparamos la alerta una vez por par
                    if (
                        time_near_vehicle > self.loitering_time_threshold
                        and key not in self.alert_triggered
                    ):
                        suspicious.append(
                            {
                                "pair_key": key,
                                "person_box": person_box,
                                "vehicle_box": vehicle_box,
                                "duration": time_near_vehicle,
                                "distance": distance,
                            }
                        )
                        # Marcamos que este par ya generó alerta
                        self.alert_triggered.add(key)

                else:
                    # Si quisieras permitir que el mismo par genere otra alerta
                    # después de alejarse y volver, podrías resetear así:
                    #
                    # key = f"p{person_id}_v{vehicle_id}"
                    # if key in self.last_detection_time:
                    #     del self.last_detection_time[key]
                    # if key in self.alert_triggered:
                    #     self.alert_triggered.remove(key)
                    #
                    # Por ahora lo dejamos tal cual (una alerta por par).
                    pass

        return suspicious

    def draw_detections(self, frame, results, suspicious_activities):
        annotated_frame = frame.copy()

        # Dibujar detecciones normales
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls == self.person_class:
                    color = (255, 255, 0)
                    label = f"Persona {conf:.2f}"
                elif cls in self.vehicle_classes:
                    color = (0, 255, 0)
                    label = f"Vehículo {conf:.2f}"
                else:
                    continue

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Dibujar actividades sospechosas (solo las NUEVAS de este frame)
        for activity in suspicious_activities:
            person_box = activity["person_box"]
            vehicle_box = activity["vehicle_box"]
            duration = activity["duration"]

            p_center = (
                int((person_box[0] + person_box[2]) / 2),
                int((person_box[1] + person_box[3]) / 2),
            )
            v_center = (
                int((vehicle_box[0] + vehicle_box[2]) / 2),
                int((vehicle_box[1] + vehicle_box[3]) / 2),
            )

            cv2.line(annotated_frame, p_center, v_center, (0, 0, 255), 3)

            x1, y1, x2, y2 = map(int, person_box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            alert_text = f"ALERTA! {duration:.1f}s cerca del vehículo"
            cv2.putText(
                annotated_frame,
                alert_text,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Texto general de alertas (nuevas en ESTE frame)
        if suspicious_activities:
            status_text = f"ALERTAS NUEVAS: {len(suspicious_activities)}"
            cv2.putText(
                annotated_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

        return annotated_frame

    def process_frame(self, frame, current_time):
        """
        Procesa UN frame, devuelve:
        - annotated_frame: frame con cajas, líneas y textos
        - alert_count: número de alertas NUEVAS en este frame
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        persons = []
        vehicles = []

        # Clasificar cajas en personas o vehículos
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                bbox = box.xyxy[0].cpu().numpy()

                if cls == self.person_class:
                    persons.append(bbox)
                elif cls in self.vehicle_classes:
                    vehicles.append(bbox)

        # Analizar actividad sospechosa (solo nuevas en este frame)
        suspicious = self.detect_suspicious_activity(persons, vehicles, current_time)

        if suspicious:
            for event in suspicious:
                self.suspicious_events.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "duration": float(event["duration"]),
                        "distance": float(event["distance"]),
                    }
                )

        annotated_frame = self.draw_detections(frame, results, suspicious)
        return annotated_frame, len(suspicious)

    # Estos métodos los puedes usar si quieres procesar un video sin GUI
    def process_video(self, video_path, output_path="output_detection.mp4", progress_callback=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"No se encontró el video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Procesando video: {video_path}")
        print(f"Frames totales: {total_frames}, FPS: {fps}")

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            processed_frame, alert_count = self.process_frame(frame, current_time)

            out.write(processed_frame)
            frame_count += 1

            # Cada 30 frames, actualizar progreso
            if total_frames > 0 and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progreso: {progress:.1f}%")

                if progress_callback:
                    progress_callback(
                        progress=progress,
                        frame_count=frame_count,
                        total_frames=total_frames,
                        total_alerts=len(self.suspicious_events),
                    )

        cap.release()
        out.release()
        elapsed_time = time.time() - start_time

        print("\n✔ Procesamiento completado!")
        print(f"Tiempo total: {elapsed_time:.2f}s")
        print(f"Total de alertas: {len(self.suspicious_events)}")

        stats = {
            "total_frames": frame_count,
            "processing_time": elapsed_time,
            "total_alerts": len(self.suspicious_events),
            "output_path": output_path,
        }

        return stats


def analyze_video(
    video_path,
    model_path="yolov8n.pt",
    output_video_path="output_detection.mp4",
    report_path="detection_report.json",
    progress_callback=None,
):
    """
    Función de alto nivel que usa ParkingSecuritySystem,
    genera el video procesado y el JSON de reporte.
    (Opcional si usas solo la UI con Tkinter)
    """
    system = ParkingSecuritySystem(model_path=model_path, confidence=0.5)

    stats = system.process_video(
        video_path=video_path,
        output_path=output_video_path,
        progress_callback=progress_callback,
    )

    report = {
        "execution_date": datetime.now().isoformat(),
        "statistics": stats,
        "configuration": {
            "confidence_threshold": system.confidence,
            "proximity_threshold": system.proximity_threshold,
            "loitering_time_threshold": system.loitering_time_threshold,
        },
        "suspicious_events": system.suspicious_events[-10:],  # últimas 10
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if os.path.exists(output_video_path):
        file_size_bytes = os.path.getsize(output_video_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(
            f"El archivo '{output_video_path}' tiene un tamaño de: {file_size_mb:.2f} MB"
        )

    return stats, output_video_path, report_path
