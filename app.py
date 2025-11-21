# app.py
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import time
import json

from PIL import Image, ImageTk
from core import ParkingSecuritySystem


class ParkingSecurityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Security System")
        self.root.geometry("1150x700")
        self.root.configure(bg="#020617")  # fondo muy oscuro
        self.root.minsize(1000, 600)

        # Estado interno
        self.video_path = None
        self.cap = None
        self.out = None
        self.system = None
        self.start_time = None
        self.frame_count = 0
        self.total_frames = 0
        self.output_video_path = "output_detection.mp4"
        self.report_path = "detection_report.json"

        # Para el reproductor del video procesado
        self.review_cap = None
        self.review_total_frames = 0
        self.review_slider = None
        self.video_controls_frame = None

        # Historial de alertas
        self.last_alert_index = 0

        # Colores
        self.bg_color = "#020617"
        self.panel_color = "#030712"
        self.accent_color = "#3b82f6"
        self.text_color = "#e5e7eb"
        self.subtle_text = "#9ca3af"

        self.build_layout()

    def build_layout(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Panel de video (izquierda)
        video_frame = tk.Frame(main_frame, bg=self.panel_color, bd=0, highlightthickness=0)
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        title_bar = tk.Frame(video_frame, bg=self.panel_color)
        title_bar.pack(fill="x", pady=(10, 5), padx=10)

        title_label = tk.Label(
            title_bar,
            text="Vista en tiempo real",
            bg=self.panel_color,
            fg=self.text_color,
            font=("Segoe UI", 13, "bold"),
        )
        title_label.pack(side="left")

        subtitle_label = tk.Label(
            title_bar,
            text="Detección de personas cerca de vehículos con YOLO",
            bg=self.panel_color,
            fg=self.subtle_text,
            font=("Segoe UI", 9),
        )
        subtitle_label.pack(side="left", padx=(10, 0))

        # Contenedor del video con tamaño fijo
        self.video_container = tk.Frame(
            video_frame,
            bg="black",
            width=800,
            height=450,
        )
        self.video_container.pack(padx=10, pady=(0, 10))
        self.video_container.pack_propagate(False)  # el contenedor no cambia de tamaño

        self.video_label_widget = tk.Label(self.video_container, bg="black")
        self.video_label_widget.pack(fill="both", expand=True)

        # Aquí pondremos los controles del video resultante (slider)
        self.video_controls_frame = None

        # Panel lateral derecho (controles e info)
        side_frame = tk.Frame(main_frame, bg=self.panel_color, bd=0, highlightthickness=0, width=300)
        side_frame.pack(side="right", fill="y", padx=(8, 0))
        side_frame.pack_propagate(False)

        panel_title = tk.Label(
            side_frame,
            text="Panel de control",
            bg=self.panel_color,
            fg=self.text_color,
            font=("Segoe UI", 14, "bold"),
        )
        panel_title.pack(pady=(15, 5), padx=15, anchor="w")

        panel_subtitle = tk.Label(
            side_frame,
            text="Carga un video, inicia el análisis y monitorea las alertas.",
            bg=self.panel_color,
            fg=self.subtle_text,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=260,
        )
        panel_subtitle.pack(pady=(0, 15), padx=15, anchor="w")

        # Info de video
        self.video_info_label = tk.Label(
            side_frame,
            text="Ningún video seleccionado",
            bg=self.panel_color,
            fg=self.subtle_text,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=260,
        )
        self.video_info_label.pack(pady=(0, 10), padx=15, anchor="w")

        # Botones
        buttons_frame = tk.Frame(side_frame, bg=self.panel_color)
        buttons_frame.pack(pady=(0, 15), padx=15, anchor="w")

        self.select_button = tk.Button(
            buttons_frame,
            text="Seleccionar video",
            command=self.select_video,
            bg=self.accent_color,
            fg="white",
            activebackground="#2563eb",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=4,
        )
        self.select_button.grid(row=0, column=0, padx=(0, 5))

        self.run_button = tk.Button(
            buttons_frame,
            text="Iniciar análisis",
            command=self.run_analysis,
            bg="#111827",
            fg=self.text_color,
            activebackground="#1f2937",
            activeforeground=self.text_color,
            relief="flat",
            font=("Segoe UI", 10),
            padx=12,
            pady=4,
            state="disabled",
        )
        self.run_button.grid(row=0, column=1, padx=(5, 0))

        # Estado y progreso
        self.status_label = tk.Label(
            side_frame,
            text="Estado: esperando selección de video",
            bg=self.panel_color,
            fg=self.subtle_text,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=260,
        )
        self.status_label.pack(pady=(0, 5), padx=15, anchor="w")

        self.progress_label = tk.Label(
            side_frame,
            text="Progreso: 0.0%",
            bg=self.panel_color,
            fg=self.subtle_text,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=260,
        )
        self.progress_label.pack(pady=(0, 5), padx=15, anchor="w")

        # Estado de alertas activas (texto pequeño)
        self.alert_status_label = tk.Label(
            side_frame,
            text="Sin alertas activas",
            bg=self.panel_color,
            fg="#f97316",  # naranja
            font=("Segoe UI", 9, "bold"),
            justify="left",
            wraplength=260,
        )
        self.alert_status_label.pack(pady=(8, 4), padx=15, anchor="w")

        # Título historial de alertas
        history_title = tk.Label(
            side_frame,
            text="Historial de alertas",
            bg=self.panel_color,
            fg=self.text_color,
            font=("Segoe UI", 10, "bold"),
            justify="left",
        )
        history_title.pack(pady=(4, 2), padx=15, anchor="w")

        # Contenedor scrollable para las tarjetas de alertas
        alerts_container = tk.Frame(side_frame, bg=self.panel_color)
        alerts_container.pack(fill="both", expand=True, padx=15, pady=(0, 10), anchor="nw")

        self.alerts_canvas = tk.Canvas(
            alerts_container,
            bg=self.panel_color,
            highlightthickness=0,
            bd=0,
        )
        self.alerts_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(
            alerts_container,
            orient="vertical",
            command=self.alerts_canvas.yview,
        )
        scrollbar.pack(side="right", fill="y")

        self.alerts_canvas.configure(yscrollcommand=scrollbar.set)

        # Frame interno donde van las tarjetas
        self.alerts_frame = tk.Frame(self.alerts_canvas, bg=self.panel_color)
        self.alerts_canvas.create_window((0, 0), window=self.alerts_frame, anchor="nw")

        # Actualizar región scroll cuando cambie el contenido
        self.alerts_frame.bind(
            "<Configure>",
            lambda e: self.alerts_canvas.configure(
                scrollregion=self.alerts_canvas.bbox("all")
            ),
        )

        # Resultado final
        self.result_label = tk.Label(
            side_frame,
            text="",
            bg=self.panel_color,
            fg=self.text_color,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=260,
        )
        self.result_label.pack(pady=(5, 5), padx=15, anchor="w")

    # -------------------------------------------------
    # Lógica de UI
    # -------------------------------------------------
    def select_video(self):
        filetypes = [
            ("Videos", "*.mp4 *.avi *.mov *.mkv"),
            ("Todos los archivos", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Selecciona un video", filetypes=filetypes
        )

        if path:
            self.video_path = path
            filename = os.path.basename(path)
            self.video_info_label.config(text=f"Video seleccionado:\n{filename}")
            self.status_label.config(text="Estado: listo para analizar")
            self.progress_label.config(text="Progreso: 0.0%")
            self.alert_status_label.config(text="Sin alertas activas")
            self.result_label.config(text="")

            # Limpiar historial de alertas
            for w in self.alerts_frame.winfo_children():
                w.destroy()
            self.last_alert_index = 0

            # Limpiar controles de reproducción del video resultante (si existían)
            if self.video_controls_frame:
                self.video_controls_frame.destroy()
                self.video_controls_frame = None
            if self.review_cap:
                self.review_cap.release()
                self.review_cap = None

            self.run_button.config(state="normal")

    def run_analysis(self):
        if not self.video_path:
            messagebox.showwarning("Video no seleccionado", "Primero selecciona un video.")
            return

        # Reiniciar estado
        self.frame_count = 0
        self.total_frames = 0
        self.output_video_path = "output_detection.mp4"
        self.report_path = "detection_report.json"
        self.last_alert_index = 0

        # Limpiar historial
        for w in self.alerts_frame.winfo_children():
            w.destroy()

        # Cerrar reproductor previo, si lo hay
        if self.review_cap:
            self.review_cap.release()
            self.review_cap = None
        if self.video_controls_frame:
            self.video_controls_frame.destroy()
            self.video_controls_frame = None

        # Crear sistema YOLO
        self.system = ParkingSecuritySystem(model_path="yolov8n.pt", confidence=0.5)

        # Abrir video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video seleccionado.")
            self.cap = None
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))

        self.start_time = time.time()

        # Actualizar UI
        self.run_button.config(state="disabled")
        self.select_button.config(state="disabled")
        self.status_label.config(text="Estado: procesando video en tiempo real...")
        self.progress_label.config(text="Progreso: 0.0%")
        self.alert_status_label.config(text="Sin alertas activas")
        self.result_label.config(text="")
        self.root.update_idletasks()

        # Comenzar bucle de frames
        self.update_frame()

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Fin del video
            self.finish_analysis()
            return

        current_time = time.time()
        processed_frame, alert_count = self.system.process_frame(frame, current_time)

        # Guardar en el video de salida
        if self.out:
            self.out.write(processed_frame)

        self.frame_count += 1

        # Mostrar en la interfaz (convertir BGR -> RGB -> ImageTk)
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Redimensionar para que quepa en el contenedor de video (contenedor fijo)
        label_width = self.video_container.winfo_width() or 800
        label_height = self.video_container.winfo_height() or 450
        img.thumbnail((label_width, label_height), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label_widget.config(image=imgtk)
        self.video_label_widget.image = imgtk  # evitar GC

        # Progreso
        if self.total_frames > 0:
            progress = (self.frame_count / self.total_frames) * 100
        else:
            progress = 0.0

        self.progress_label.config(
            text=f"Progreso: {progress:.1f}%  |  Frames: {self.frame_count}/{self.total_frames or '?'}"
        )

        # Alertas activas (para el frame actual)
        total_alerts = len(self.system.suspicious_events)
        if alert_count > 0:
            self.alert_status_label.config(
                text=f"ALERTAS ACTIVAS: {alert_count} (Total acumuladas: {total_alerts})"
            )
        else:
            self.alert_status_label.config(
                text=f"Sin alertas activas (Total acumuladas: {total_alerts})"
            )

        # Actualizar historial de alertas (tarjetas verdes nuevas)
        if self.system and len(self.system.suspicious_events) > self.last_alert_index:
            new_events = self.system.suspicious_events[self.last_alert_index :]
            for ev in new_events:
                dur = ev.get("duration", 0.0)
                dist = ev.get("distance", 0.0)
                ts = ev.get("timestamp", "")
                text = (
                    f"{ts}\n"
                    f"Persona cerca de vehículo\n"
                    f"Duración: {dur:.1f}s  |  Distancia: {dist:.1f}"
                )
                card = tk.Label(
                    self.alerts_frame,
                    text=text,
                    bg="#16a34a",  # verde
                    fg="white",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=240,
                    anchor="w",
                    padx=6,
                    pady=4,
                )
                card.pack(fill="x", pady=2, anchor="w")
            self.last_alert_index = len(self.system.suspicious_events)

        # Programar siguiente frame
        self.root.after(1, self.update_frame)

    def finish_analysis(self):
        # Cerrar recursos
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.out:
            self.out.release()
            self.out = None

        elapsed_time = time.time() - self.start_time if self.start_time else 0.0
        total_alerts = len(self.system.suspicious_events) if self.system else 0

        stats = {
            "total_frames": self.frame_count,
            "processing_time": elapsed_time,
            "total_alerts": total_alerts,
            "output_path": self.output_video_path,
        }

        # Crear reporte JSON
        report = {
            "execution_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "statistics": stats,
            "configuration": {
                "confidence_threshold": self.system.confidence if self.system else 0.5,
                "proximity_threshold": self.system.proximity_threshold if self.system else 100,
                "loitering_time_threshold": self.system.loitering_time_threshold if self.system else 5,
            },
            "suspicious_events": (self.system.suspicious_events[-10:] if self.system else []),
        }

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Info de archivo
        if os.path.exists(self.output_video_path):
            size_bytes = os.path.getsize(self.output_video_path)
            size_mb = size_bytes / (1024 * 1024)
            size_txt = f"{size_mb:.2f} MB"
        else:
            size_txt = "desconocido"

        fps_promedio = (
            self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        )

        # Actualizar UI
        self.status_label.config(text="Estado: análisis completado ✔")
        self.result_label.config(
            text=(
                f"Frames: {self.frame_count} | Alertas: {total_alerts}\n"
                f"Tiempo: {elapsed_time:.2f}s | FPS promedio: {fps_promedio:.2f}\n"
                f"Tamaño video: {size_txt}\n"
                f"Video: {os.path.abspath(self.output_video_path)}\n"
                f"Reporte JSON: {os.path.abspath(self.report_path)}"
            )
        )

        self.run_button.config(state="normal")
        self.select_button.config(state="normal")

        # Configurar reproductor del video generado (slider para adelantar/atrasar)
        self.setup_result_player()

        messagebox.showinfo(
            "Análisis completado",
            "El procesamiento ha terminado.\n\n"
            f"Video generado:\n{os.path.abspath(self.output_video_path)}\n\n"
            f"Reporte JSON:\n{os.path.abspath(self.report_path)}",
        )

    # -------------------------------------------------
    # Reproductor del video procesado (slider manual)
    # -------------------------------------------------
    def setup_result_player(self):
        # Si no existe el video, nada
        if not os.path.exists(self.output_video_path):
            return

        # Cerrar reproductor previo
        if self.review_cap:
            self.review_cap.release()

        self.review_cap = cv2.VideoCapture(self.output_video_path)
        self.review_total_frames = int(self.review_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Crear frame de controles debajo del video
        if self.video_controls_frame:
            self.video_controls_frame.destroy()

        self.video_controls_frame = tk.Frame(
            self.video_container.master, bg=self.panel_color
        )
        self.video_controls_frame.pack(fill="x", padx=10, pady=(0, 10))

        slider_label = tk.Label(
            self.video_controls_frame,
            text="Revisión del video procesado (mueve el control para adelantar/atrasar):",
            bg=self.panel_color,
            fg=self.subtle_text,
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
        )
        slider_label.pack(anchor="w", pady=(0, 2))

        self.review_slider = tk.Scale(
            self.video_controls_frame,
            from_=0,
            to=max(self.review_total_frames - 1, 0),
            orient="horizontal",
            length=700,
            showvalue=False,
            command=self.on_review_slider_change,
            bg=self.panel_color,
            fg=self.text_color,
            troughcolor="#111827",
            highlightthickness=0,
        )
        self.review_slider.pack(anchor="w")

        # Mostrar primer frame
        self.show_review_frame(0)

    def on_review_slider_change(self, value):
        # Llamado cuando el usuario mueve el slider
        idx = int(float(value))
        self.show_review_frame(idx)

    def show_review_frame(self, frame_index):
        if not self.review_cap:
            return

        # Posicionar el video en el frame deseado
        self.review_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.review_cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        label_width = self.video_container.winfo_width() or 800
        label_height = self.video_container.winfo_height() or 450
        img.thumbnail((label_width, label_height), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label_widget.config(image=imgtk)
        self.video_label_widget.image = imgtk  # evitar GC


if __name__ == "__main__":
    root = tk.Tk()
    app = ParkingSecurityApp(root)
    root.mainloop()
