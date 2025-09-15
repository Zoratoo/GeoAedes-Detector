# AI-Powered Environmental Surveillance for Mapping Dengue Breeding Sites

This repository contains the source code for my **Graduation Project**. The project aims to provide a low-cost, technology-driven solution to identify, classify, and geolocate potential breeding sites for the *Aedes aegypti* mosquito, the primary vector for diseases like Dengue, Zika, and Chikungunya.

The core of this project is a desktop application that analyzes urban environments through images and videos to detect objects that can accumulate stagnant water, such as discarded tires, bottles, and open containers.

---

## ✨ Core Features

-   **Dual Processing Modes:** The application is organized into two main tabs for analyzing either static **images** or **videos**.
-   **Hybrid AI Pipeline:**
    -   **YOLO:** For fast and efficient primary object detection.
    -   **EfficientNetV2:** As a secondary, fine-grained classifier to verify low-confidence detections from YOLO, increasing overall accuracy.
-   **Real-Time Video Analysis:** Utilizes a multi-threaded producer-consumer architecture to provide a smooth video playback experience while performing continuous object detection and tracking in the background.
-   **Intelligent Object Tracking:** Implements OpenCV's CSRT tracker to maintain object bounding boxes across frames where AI inference is skipped, creating a fluid and persistent visual output.
-   **User-Adjustable Controls:** The UI includes interactive sliders to dynamically change the confidence thresholds for both AI models and performance mode selectors to balance processing speed and detection frequency in videos.
-   **Automated Summary Reports:** After processing a video, the application generates a complete summary, counting the total number of each class of object detected.

## ⚙️ How It Works

The solution is designed around a two-part system:

1.  **Hardware (Proposed):** A mobile data-capturing unit, prototyped with an Arduino and GPS module, synchronized with a camera. This device is intended to be mounted on a vehicle to collect georeferenced images and videos of urban streets.
2.  **Software (This Repository):** The desktop application, which takes the collected visual data as input and uses the AI pipeline to identify and classify potential breeding sites. The final output is structured data that can be used to generate heatmaps on platforms like Google Earth, providing valuable insights for public health agents.

## 🛠️ Technology Stack

-   **Object Detection:** YOLOv11s
-   **Classification:** EfficientNetV2
-   **Video Object Tracking:** OpenCV (CSRT Legacy Tracker)
-   **GUI:** Python, CustomTkinter
-   **Core Libraries:** PyTorch, TensorFlow, OpenCV, NumPy, Pillow
-   **Hardware (Prototype):** Arduino, GPS Module
