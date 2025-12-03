import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import os
import threading
import time
from datetime import datetime, timedelta

# --- CONFIGURAÇÕES GERAIS ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

YOLO_MODEL_PATH = 'models/yolo_detector.pt'
EFFICIENTNET_MODEL_PATH = 'models/efficientnet_classifier.keras'
EFFICIENTNET_CLASSES = ['plastic_bottle', 'tire', 'can', 'open_container', 'garbage_bag', 'sofa', 'plastic_bag']
COLORS = {'plastic_bottle': (255, 179, 0), 'tire': (0, 255, 255), 'can': (255, 0, 102), 'open_container': (0, 204, 0), 'garbage_bag': (170, 0, 255), 'sofa': (0, 102, 204), 'plastic_bag': (204, 204, 0)}

EFFICIENTNET_IMG_SIZE = (224, 224)
DISPLAY_FPS = 30 

PERFORMANCE_MODES = {
    "Precisão": 5,
    "Equilíbrio": 15,
    "Performance": 30
}

GPS_TIMEZONE_OFFSET_HOURS = 3
DETECTION_DEBOUNCE_SECONDS = 3

class ObjectDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Mapeamento Geográfico de Objetos v8.2")
        self.after(0, lambda: self.state('zoomed'))

        # ... (O resto do __init__ é idêntico)
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.processing_video = False
        self.video_source = None
        self.processing_thread = None
        self.latest_processed_frame = None
        self.frame_lock = threading.Lock()
        self.video_object_counts = {}
        self.current_image_path = None
        self.image_processing_mode = ctk.StringVar(value="yolo_effnet")
        self.video_performance_mode = ctk.StringVar(value="Equilíbrio")
        self.individual_trackers = []
        self.tracked_objects_info = [] 
        self.gps_data = []
        self.detected_points = []
        self.last_detection_timestamp = None
        self.video_start_time = None
        self.video_fps = 0
        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=1)
        self.tabview = ctk.CTkTabview(self.main_content_frame, command=self._on_tab_change)
        self.tabview.grid(row=0, column=0, sticky="nsew")
        self.video_tab = self.tabview.add("Processamento de Vídeo")
        self.image_tab = self.tabview.add("Processamento de Imagem")
        self.video_display_label = ctk.CTkLabel(self.video_tab, text="1. Selecione um arquivo GPS\n2. Selecione um vídeo para começar.", text_color="gray", font=ctk.CTkFont(size=18))
        self.video_display_label.pack(expand=True, fill="both", padx=10, pady=10)
        self.image_display_label = ctk.CTkLabel(self.image_tab, text="Selecione uma imagem na barra lateral para começar.", text_color="gray", font=ctk.CTkFont(size=18))
        self.image_display_label.pack(expand=True, fill="both", padx=10, pady=10)
        self._setup_sidebar()
        self.load_models()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._on_tab_change()

    def _setup_sidebar(self):
        # ... (código idêntico)
        self.sidebar_frame = ctk.CTkFrame(self)
        self.sidebar_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        ctk.CTkLabel(self.sidebar_frame, text="Controles e Análise", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=20)
        self._create_legend(self.sidebar_frame, EFFICIENTNET_CLASSES).grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.controls_container = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.controls_container.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.video_controls_frame = ctk.CTkFrame(self.controls_container, fg_color="transparent")
        self.btn_select_gps = ctk.CTkButton(self.video_controls_frame, text="1. Selecionar Arquivo GPS (.txt)", command=self.select_gps_file)
        self.btn_select_gps.pack(pady=5, fill="x")
        self.btn_select_video = ctk.CTkButton(self.video_controls_frame, text="2. Selecionar Vídeo e Iniciar", command=self.select_video, state="disabled")
        self.btn_select_video.pack(pady=5, fill="x")
        self.btn_stop_video = ctk.CTkButton(self.video_controls_frame, text="Parar Processamento", command=self.stop_processing, state="disabled")
        self.btn_stop_video.pack(pady=5, fill="x")
        self.btn_generate_kml = ctk.CTkButton(self.video_controls_frame, text="3. Gerar Mapa KML", command=self.generate_kml_file, state="disabled")
        self.btn_generate_kml.pack(pady=(10, 5), fill="x")
        performance_frame = ctk.CTkFrame(self.video_controls_frame)
        performance_frame.pack(pady=10, fill="x")
        ctk.CTkLabel(performance_frame, text="Modo:").pack(side="left", padx=(10, 5))
        mode_selector = ctk.CTkSegmentedButton(performance_frame, variable=self.video_performance_mode, values=list(PERFORMANCE_MODES.keys()))
        mode_selector.pack(side="left", expand=True, fill="x", padx=(0, 10))
        self.image_controls_frame = ctk.CTkFrame(self.controls_container, fg_color="transparent")
        self.btn_select_image = ctk.CTkButton(self.image_controls_frame, text="Selecionar Imagem", command=self.select_image_for_processing)
        self.btn_select_image.pack(pady=5, fill="x")
        model_selection_frame = ctk.CTkFrame(self.image_controls_frame)
        model_selection_frame.pack(pady=10, fill="x")
        ctk.CTkLabel(model_selection_frame, text="Modelo:").pack(side="left", padx=10)
        self.radio_yolo = ctk.CTkRadioButton(model_selection_frame, text="YOLO", variable=self.image_processing_mode, value="yolo", command=self._process_selected_image_if_present)
        self.radio_yolo.pack(side="left", padx=5, expand=True)
        self.radio_effnet = ctk.CTkRadioButton(model_selection_frame, text="EfficientNet", variable=self.image_processing_mode, value="effnet", command=self._process_selected_image_if_present)
        self.radio_effnet.pack(side="left", padx=5, expand=True)
        self.radio_yolo_effnet = ctk.CTkRadioButton(model_selection_frame, text="YOLO+EffNet", variable=self.image_processing_mode, value="yolo_effnet", command=self._process_selected_image_if_present)
        self.radio_yolo_effnet.pack(side="left", padx=5, expand=True)
        thresholds_frame = ctk.CTkFrame(self.sidebar_frame)
        thresholds_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        thresholds_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(thresholds_frame, text="Limiar YOLO p/ EfficientNet").grid(row=0, column=0, sticky="w")
        self.yolo_threshold_label = ctk.CTkLabel(thresholds_frame, text="60%")
        self.yolo_threshold_label.grid(row=0, column=1, sticky="e")
        self.yolo_threshold_slider = ctk.CTkSlider(thresholds_frame, from_=0.0, to=1.0, command=self._update_yolo_threshold_label)
        self.yolo_threshold_slider.set(0.6)
        self.yolo_threshold_slider.grid(row=1, column=0, columnspan=2, sticky="ew")
        ctk.CTkLabel(thresholds_frame, text="Confiança Mínima EfficientNet").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.effnet_threshold_label = ctk.CTkLabel(thresholds_frame, text="50%")
        self.effnet_threshold_label.grid(row=2, column=1, sticky="e", pady=(10, 0))
        self.effnet_threshold_slider = ctk.CTkSlider(thresholds_frame, from_=0.0, to=1.0, command=self._update_effnet_threshold_label)
        self.effnet_threshold_slider.set(0.5)
        self.effnet_threshold_slider.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.log_textbox = ctk.CTkTextbox(self.sidebar_frame, state="disabled", wrap="word")
        self.log_textbox.grid(row=4, column=0, padx=20, pady=20, sticky="nsew")

    def _update_yolo_threshold_label(self, value): self.yolo_threshold_label.configure(text=f"{value:.0%}")
    def _update_effnet_threshold_label(self, value): self.effnet_threshold_label.configure(text=f"{value:.0%}")
        
    def _on_tab_change(self):
        selected_tab = self.tabview.get()
        if selected_tab == "Processamento de Vídeo":
            self.image_controls_frame.pack_forget()
            self.video_controls_frame.pack(fill="x", expand=True)
        else:
            self.video_controls_frame.pack_forget()
            self.image_controls_frame.pack(fill="x", expand=True)

    def _create_legend(self, parent_frame, classes_list):
        # ... (código idêntico)
        legend_frame = ctk.CTkFrame(parent_frame)
        legend_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(legend_frame, text="Legenda", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        for i, class_name in enumerate(classes_list):
            color_bgr = COLORS.get(class_name, (255,255,255))
            color_hex = f'#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}'
            ctk.CTkLabel(legend_frame, text="", width=20, height=20, fg_color=color_hex, corner_radius=5).grid(row=i+1, column=0, padx=10, pady=2)
            ctk.CTkLabel(legend_frame, text=class_name.replace("_", " ").title()).grid(row=i+1, column=1, padx=10, pady=2, sticky="w")
        return legend_frame

    def add_to_log(self, message):
        # ... (código idêntico)
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.configure(state="disabled")
        self.log_textbox.see("end")

    def load_models(self):
        # ... (código idêntico)
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH)
            self.add_to_log("INFO: Modelos carregados com sucesso.")
        except Exception as e:
            messagebox.showerror("Erro ao Carregar Modelos", f"Não foi possível carregar os modelos.\n\nErro: {e}")
            self.destroy()

    def on_closing(self):
        if self.processing_video: self.stop_processing()
        self.destroy()

    def select_gps_file(self):
        # ... (código idêntico)
        file_path = filedialog.askopenfilename(filetypes=[("Arquivos de Texto", "*.txt")])
        if file_path:
            self.add_to_log(f"INFO: Arquivo GPS selecionado: {os.path.basename(file_path)}")
            self._parse_gps_data(file_path)

    def _parse_gps_data(self, file_path):
        # ... (código idêntico)
        self.gps_data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try: next(f)
                except StopIteration: pass
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8: continue
                    lat_str, lon_str, date_str, time_str, alt_str = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip(), parts[4].strip()
                    timestamp_str = f"{date_str} {time_str}"
                    timestamp_utc = datetime.strptime(timestamp_str, '%d/%m/%Y %H:%M:%S')
                    timestamp_brt = timestamp_utc - timedelta(hours=GPS_TIMEZONE_OFFSET_HOURS)
                    self.gps_data.append({'timestamp': timestamp_brt, 'lat': float(lat_str), 'lon': float(lon_str), 'alt': float(alt_str), 'original_line': line.strip()})
            if not self.gps_data:
                self.add_to_log("AVISO: Nenhum ponto de GPS válido foi encontrado no arquivo."); self.btn_select_video.configure(state="disabled"); return
            self.add_to_log(f"INFO: {len(self.gps_data)} pontos de GPS carregados e ajustados.")
            self.btn_select_video.configure(state="normal")
        except Exception as e:
            self.add_to_log(f"ERRO: Falha ao ler o arquivo GPS. Verifique o formato. Erro: {e}")
            self.btn_select_video.configure(state="disabled")

    def _find_gps_for_timestamp(self, frame_time):
        # ... (código idêntico)
        if not self.gps_data: return None
        closest_point = min(self.gps_data, key=lambda point: abs((point['timestamp'] - frame_time).total_seconds()))
        return closest_point

    # --- FUNÇÃO PRINCIPALMENTE ALTERADA ---
    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi *.mov")])
        if file_path:
            # Reseta contadores e listas
            self.video_object_counts = {class_name: 0 for class_name in EFFICIENTNET_CLASSES}
            self.detected_points = []
            self.last_detection_timestamp = None
            self.individual_trackers = []
            self.tracked_objects_info = []

            # --- CORREÇÃO DE SINCRONIZAÇÃO AQUI ---
            # Define o tempo de início do vídeo como o timestamp do PRIMEIRO ponto do GPS
            if self.gps_data:
                self.video_start_time = self.gps_data[0]['timestamp']
            else:
                self.add_to_log("ERRO: Não há dados de GPS carregados. Impossível sincronizar o vídeo.")
                messagebox.showerror("Erro de Sincronização", "Carregue um arquivo GPS válido antes de selecionar o vídeo.")
                return

            self.video_source = cv2.VideoCapture(file_path)
            self.video_fps = self.video_source.get(cv2.CAP_PROP_FPS)
            if self.video_fps == 0:
                self.add_to_log("AVISO: Não foi possível ler o FPS do vídeo. Usando valor padrão de 30.")
                self.video_fps = 30.0

            self.add_to_log(f"INFO: Vídeo iniciado. Sincronizando com o timestamp inicial do GPS: {self.video_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Atualiza a UI e inicia a thread
            self.processing_video = True
            self.btn_select_gps.configure(state="disabled")
            self.btn_select_video.configure(state="disabled")
            self.btn_stop_video.configure(state="normal")
            self.btn_generate_kml.configure(state="disabled")
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            self._display_loop()

    # ... (O resto do código é idêntico à v8.1)
    def stop_processing(self):
        self.processing_video = False
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join()
        if self.video_source is not None:
            self.video_source.release()
            self.video_source = None
        self.btn_select_gps.configure(state="normal")
        self.btn_select_video.configure(state="normal")
        self.btn_stop_video.configure(state="disabled")
        self.add_to_log("\n" + "="*40 + "\n      RESUMO DA DETECÇÃO DO VÍDEO\n" + "="*40)
        total_objects = sum(self.video_object_counts.values())
        if total_objects == 0:
            self.add_to_log("Nenhum objeto foi detectado.")
        else:
            for class_name, count in self.video_object_counts.items():
                if count > 0: self.add_to_log(f"- {class_name.replace('_', ' ').title()}: {count}")
            self.add_to_log(f"\nTOTAL DE OBJETOS: {total_objects}")
            self.add_to_log(f"Total de Pontos GPS salvos: {len(self.detected_points)}")
            if self.detected_points:
                self.btn_generate_kml.configure(state="normal")
        self.add_to_log("="*40 + "\nINFO: Processamento de vídeo finalizado.")
    def _processing_loop(self):
        frame_counter = 0
        mode = self.video_performance_mode.get()
        frame_skip = PERFORMANCE_MODES.get(mode, 15)
        self.add_to_log(f"INFO: Modo '{mode}'. Re-detecção a cada {frame_skip} frames.")
        while self.processing_video:
            if self.video_source is None: break
            ret, frame = self.video_source.read()
            if not ret: break
            image_to_draw_on = frame.copy()
            current_frame_time = self.video_start_time + timedelta(seconds=frame_counter / self.video_fps)
            if frame_counter % frame_skip == 0:
                detections = self.detect_objects_in_frame(frame)
                self.individual_trackers = []
                self.tracked_objects_info = []
                for det in detections:
                    bbox = det['bbox']
                    bbox_tracker_format = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, bbox_tracker_format)
                    self.individual_trackers.append(tracker)
                    self.tracked_objects_info.append(det)
                for obj in self.tracked_objects_info:
                    now = time.time()
                    if self.last_detection_timestamp is None or (now - self.last_detection_timestamp) > DETECTION_DEBOUNCE_SECONDS:
                        gps_point = self._find_gps_for_timestamp(current_frame_time)
                        if gps_point:
                            self.detected_points.append({'class_name': obj['class_name'], 'gps_data': gps_point})
                            self.add_to_log(f"PONTO SALVO: {obj['class_name']} em {gps_point['timestamp'].strftime('%H:%M:%S')}")
                            self.last_detection_timestamp = now
                    if obj['class_name'] in self.video_object_counts:
                        self.video_object_counts[obj['class_name']] += 1
            else:
                if self.individual_trackers:
                    new_trackers, new_tracked_objects_info = [], []
                    for tracker, info in zip(self.individual_trackers, self.tracked_objects_info):
                        success, new_box = tracker.update(frame)
                        if success:
                            x, y, w, h = [int(v) for v in new_box]
                            if w > 0 and h > 0:
                                info['bbox'] = (x, y, x + w, y + h)
                                new_trackers.append(tracker)
                                new_tracked_objects_info.append(info)
                    self.individual_trackers = new_trackers
                    self.tracked_objects_info = new_tracked_objects_info
            for obj in self.tracked_objects_info:
                color = COLORS.get(obj['class_name'], (255, 255, 255))
                self._draw_bounding_box_and_label(image_to_draw_on, obj['bbox'], obj['label'], color)
            with self.frame_lock: self.latest_processed_frame = image_to_draw_on
            frame_counter += 1
        self.add_to_log("INFO: Thread de processamento finalizada.")
    def detect_objects_in_frame(self, frame):
        detections = []
        original_for_cropping = frame.copy()
        model_class_names_yolo = self.yolo_model.names
        yolo_thresh = self.yolo_threshold_slider.get()
        effnet_thresh = self.effnet_threshold_slider.get()
        yolo_results = self.yolo_model(frame, verbose=False)[0]
        for result in yolo_results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            bbox = tuple(map(int, [x1, y1, x2, y2]))
            class_id = int(class_id)
            if class_id >= len(model_class_names_yolo): continue
            detected_class_name_yolo = model_class_names_yolo[class_id]
            final_class_name, source, confidence = "", "", 0.0
            if score >= yolo_thresh:
                final_class_name, source, confidence = detected_class_name_yolo, "YOLO", score
            else:
                crop = original_for_cropping[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if crop.size > 0:
                    preprocessed_crop = self.preprocess_for_efficientnet(crop)
                    predictions = self.efficientnet_model.predict(preprocessed_crop, verbose=0)[0]
                    effnet_confidence = np.max(predictions)
                    if effnet_confidence >= effnet_thresh:
                        effnet_class_id = np.argmax(predictions)
                        final_class_name = EFFICIENTNET_CLASSES[effnet_class_id]
                        source = "EfficientNet"
                        confidence = effnet_confidence
                    else:
                        self.add_to_log(f"  - Descartado. Conf. EfficientNet ({effnet_confidence:.0%}) < {effnet_thresh:.0%}")
            if final_class_name:
                label = f"{final_class_name} {confidence:.0%} ({source})"
                detections.append({'bbox': bbox, 'label': label, 'class_name': final_class_name})
        return detections
    def _display_loop(self):
        if self.processing_video and (self.processing_thread is None or not self.processing_thread.is_alive()): self.stop_processing(); return
        if not self.processing_video: return
        with self.frame_lock: frame_to_display = self.latest_processed_frame
        if frame_to_display is not None: self._update_display_label(self.video_display_label, frame_to_display)
        self.after(int(1000 / DISPLAY_FPS), self._display_loop)
    def select_image_for_processing(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            self._process_selected_image_if_present()
    def _process_selected_image_if_present(self):
        if self.current_image_path:
            self.log_textbox.configure(state="normal")
            self.log_textbox.delete("1.0", "end")
            self.log_textbox.configure(state="disabled")
            self.add_to_log("-" * 40 + f"\nINFO: Processando imagem: {os.path.basename(self.current_image_path)} com modo '{self.image_processing_mode.get().upper()}'")
            image = cv2.imread(self.current_image_path)
            if image is None: self.add_to_log("ERRO: Não foi possível carregar a imagem."); return
            processed_image = self.process_single_image(image.copy())
            self._update_display_label(self.image_display_label, processed_image)
    def process_single_image(self, image):
        processing_mode = self.image_processing_mode.get()
        if processing_mode in ["yolo", "yolo_effnet"]:
            detections = self.detect_objects_in_frame(image)
            if not detections: self.add_to_log("Nenhuma detecção feita com YOLO.")
            for det in detections:
                color = COLORS.get(det['class_name'], (255, 255, 255))
                self._draw_bounding_box_and_label(image, det['bbox'], det['label'], color)
        elif processing_mode == "effnet":
            self.add_to_log("  - Classificando imagem inteira com EfficientNet...")
            preprocessed_image = self.preprocess_for_efficientnet(image)
            predictions = self.efficientnet_model.predict(preprocessed_image, verbose=0)[0]
            effnet_confidence = np.max(predictions)
            if effnet_confidence >= self.effnet_threshold_slider.get():
                effnet_class_id = np.argmax(predictions)
                final_class_name = EFFICIENTNET_CLASSES[effnet_class_id]
                label = f"{final_class_name} {effnet_confidence:.0%} (EfficientNet)"
                self.add_to_log(f"    - Resultado: {label}")
                color_bgr = COLORS.get(final_class_name, (255, 255, 255))
                h, w, _ = image.shape
                self._draw_bounding_box_and_label(image, (0, 0, w, h), label, color_bgr, full_image_mode=True)
            else:
                self.add_to_log(f"    - Resultado Descartado. Confiança ({effnet_confidence:.0%}) abaixo do limiar.")
        return image
    def preprocess_for_efficientnet(self, image_crop):
        img_resized = cv2.resize(image_crop, EFFICIENTNET_IMG_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb)
        img_array_float32 = img_array.astype(np.float32)
        img_batch = np.expand_dims(img_array_float32, axis=0)
        return img_batch
    def _draw_bounding_box_and_label(self, image_cv, bbox, label, color_bgr, full_image_mode=False):
        x1, y1, x2, y2 = bbox
        (h_img, w_img, _) = image_cv.shape
        font_scale = min(w_img, h_img) / 1300.0
        thickness_line = max(1, int(font_scale * 3))
        thickness_text = max(1, int(font_scale * 2))
        font_size_cv = max(0.4, font_scale * 0.8)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color_bgr, thickness_line)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size_cv, thickness_text)
        if full_image_mode:
            text_x = x1 + int(thickness_line * 1.5); text_y = y1 + text_h + int(thickness_line * 1.5)
            cv2.rectangle(image_cv, (x1, y1), (x1 + text_w + int(thickness_line * 3), y1 + text_h + int(thickness_line * 3)), color_bgr, -1)
            cv2.putText(image_cv, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_size_cv, (255, 255, 255), thickness_text)
        else:
            text_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10
            cv2.rectangle(image_cv, (x1, y1 - text_h - 15), (x1 + text_w, y1), color_bgr, -1)
            cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size_cv, (255, 255, 255), thickness_text)
    def _update_display_label(self, target_label, frame_cv):
        parent_w = target_label.master.winfo_width()
        parent_h = target_label.master.winfo_height()
        if parent_w <= 1 or parent_h <= 1: parent_w = 1280; parent_h = 720
        image_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        img_w, img_h = pil_image.size
        ratio = min(parent_w / img_w, parent_h / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        resized_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=resized_image, dark_image=resized_image, size=(new_w, new_h))
        target_label.configure(image=photo, text="")
        target_label.image = photo
    def generate_kml_file(self):
        if not self.detected_points:
            self.add_to_log("AVISO: Nenhum ponto detectado para gerar o mapa.")
            messagebox.showwarning("Geração de KML", "Nenhum ponto de detecção foi salvo para gerar o arquivo de mapa.")
            return
        
        # Mapeamento de cores para KML (AABBGGRR format) - Cores otimizadas para objetos mais frequentes
        color_mapping = {
            'plastic_bottle': 'FF0000FF',  # Vermelho puro - Muito visível
            'tire': 'FF00FF00',  # Verde puro
            'can': 'FFFF0000',  # Azul puro
            'open_container': 'FF00CCFF',  # Amarelo brilhante - Muito contrastante
            'garbage_bag': 'FF0080FF',  # Laranja - Muito visível
            'sofa': 'FF808000',  # Ciano/Azul
            'plastic_bag': 'FFFF00FF'  # Magenta brilhante - Muito contrastante
        }
        
        # Agrupar pontos por classe
        points_by_class = {}
        for point in self.detected_points:
            class_name = point['class_name']
            if class_name not in points_by_class:
                points_by_class[class_name] = []
            points_by_class[class_name].append(point)
        
        # Iniciar documento KML
        kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Pontos de Detecção de Objetos</name>
    <Description>Objetos acumuladores de água detectados pelo sistema de visão computacional.</Description>

"""
        
        # Adicionar estilos para cada classe
        for class_name in points_by_class.keys():
            style_name = class_name.replace('_', ' ').title().replace(' ', '_')
            color = color_mapping.get(class_name, 'FFFFFFFF')
            kml_content += f"""    <Style id="style_{style_name}">
      <IconStyle>
        <color>{color}</color>
        <scale>1.2</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href>
        </Icon>
      </IconStyle>
      <LabelStyle>
        <scale>0</scale>
      </LabelStyle>
    </Style>
"""
        
        # Adicionar folders para cada classe
        for class_name, points in points_by_class.items():
            class_title = class_name.replace('_', ' ').title()
            style_name = class_title.replace(' ', '_')
            kml_content += f"""    <Folder>
      <name>{class_title} ({len(points)})</name>
      <open>1</open>
"""
            
            # Adicionar placemarks para cada ponto
            for point in points:
                gps_info = point['gps_data']
                timestamp = gps_info['timestamp'].strftime('%d/%m/%Y %H:%M:%S')
                kml_content += f"""      <Placemark>
        <name>{class_title}</name>
        <description>
        
          &lt;b&gt;Objeto:&lt;/b&gt; {class_title}&lt;br/&gt;
          &lt;b&gt;Data/Hora:&lt;/b&gt; {timestamp}&lt;br/&gt;
          &lt;b&gt;Latitude:&lt;/b&gt; {gps_info['lat']}&lt;br/&gt;
          &lt;b&gt;Longitude:&lt;/b&gt; {gps_info['lon']}&lt;br/&gt;
          &lt;b&gt;Altitude:&lt;/b&gt; {gps_info['alt']:.2f} m
        
        </description>
        <Point>
          <coordinates>{gps_info['lon']},{gps_info['lat']},{gps_info['alt']}</coordinates>
        </Point>
        <styleUrl>#style_{style_name}</styleUrl>
      </Placemark>

"""
            
            kml_content += "    </Folder>"
        
        kml_content += """  </Document>
</kml>
"""
        
        file_path = filedialog.asksaveasfilename(defaultextension=".kml", filetypes=[("Google Earth KML", "*.kml")], title="Salvar arquivo de mapa KML")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f: 
                    f.write(kml_content)
                self.add_to_log(f"INFO: Mapa KML salvo com sucesso em: {os.path.basename(file_path)}")
                messagebox.showinfo("Sucesso", "Arquivo de mapa KML gerado e salvo com sucesso!")
            except Exception as e:
                self.add_to_log(f"ERRO: Falha ao salvar o arquivo KML. Erro: {e}")
                messagebox.showerror("Erro de Salvamento", f"Não foi possível salvar o arquivo KML.\n\nErro: {e}")

if __name__ == "__main__":
    app = ObjectDetectorApp()
    app.mainloop()