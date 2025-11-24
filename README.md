# 🛡️ Parking Security System – Detección de Actividad Sospechosa

Sistema de visión por computadora diseñado para detectar interacciones sospechosas entre personas y vehículos en parqueaderos residenciales.  
A partir del modelo YOLOv8 y el seguimiento persistente ByteTrack, el sistema identifica presencia humana cerca de vehículos, evalúa proximidad en el tiempo y emite alertas cuando se detecta posible manipulación no autorizada.

Incluye:
- Motor de detección basado en deep learning
- Interfaz gráfica en Tkinter
- Reproductor de video procesado
- Generación automática de reportes JSON
- Anotaciones visuales en tiempo real

---

## 📌 Características principales

- Detección de personas y vehículos en video usando **YOLOv8**.
- Seguimiento robusto mediante **ByteTrack**.
- Medición de proximidad usando **IoU real y expandido**.
- Generación de alertas con persistencia temporal.
- Interfaz gráfica intuitiva desarrollada con **Tkinter + PIL**.
- Producción automática de:
  - Video procesado y anotado.
  - Reporte JSON con estadísticas y eventos.
- Barra de progreso, FPS, estadísticas finales y reproductor integrado.

---

## 📦 Estructura del Proyecto

####  📁 ParkingSecuritySystem
#### │
#### ├── core.py # Motor de detección (YOLO + ByteTrack + IoU)
#### ├── app.py # Interfaz gráfica en Tkinter
#### ├── modelos/ # Ubicación recomendada del archivo yolov8n.pt
#### ├── videos/ # Videos de entrada
#### ├── resultados/ # Video procesado + reporte JSON
#### ├── README.md # Este documento
#### └── requirements.txt # Dependencias (opcional)

---

## ⚙️ Requerimientos

- Python **3.8+**
- Windows / Linux / macOS
- CPU o GPU compatible (opcional)

---

# 🚀 Instalación

### **1. Clonar el repositorio o Descargar el respositorio**



---
### **2. Crear un entorno virtual (recomendado)**

python -m venv venv

### **3. Activar el entorno virtual**

venv\Scripts\activate

### **4. Instalar dependencias**

pip install ultralytics opencv-python numpy

---

## ▶️ Uso del sistema

### **1. Ejecutar la aplicación**

python app.py 

### 2. Dentro de la interfaz

#### 1. Seleccionar video desde tu equipo.

Haciendo uso del boton "seleccionar video"

<img width="463" height="91" alt="image" src="https://github.com/user-attachments/assets/8406c994-79e7-428c-8d17-591a59dc3790" />


#### 2. Iniciar análisis para:

  - Ver detecciones en tiempo real.

  - Monitorear alertas sospechosas.

  - Revisar barra de progreso, FPS y estadísticas.

#### 3. Al finalizar, el sistema generará:

  - Un video procesado con anotaciones.

  - Un reporte JSON con eventos y parámetros.

#### 4. Usar el reproductor integrado para avanzar manualmente por el video usando el slider.


## 📊 Salidas generadas

En la carpeta resultados/ encontrarás:

🎥 video_procesado.mp4

Video con:

  - Detecciones
  - Alertas confirmadas
  - Anotaciones visuales
  - Identificadores únicos por objeto

📄 reporte.json

Incluye:

  - Tiempos de procesamiento
  - FPS promedio
  - Parámetros usados
  - Número de alertas detectadas
  - Lista de eventos sospechosos


