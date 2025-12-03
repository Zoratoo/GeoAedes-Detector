import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONFIGURAÇÕES ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

YOLO_MODEL_PATH = 'models/yolo_detector.pt'
EFFICIENTNET_MODEL_PATH = 'models/efficientnet_classifier.keras'
EFFICIENTNET_IMG_SIZE = (224, 224)
EFFICIENTNET_CLASSES = ['plastic_bottle', 'tire', 'can', 'open_container', 'garbage_bag', 'sofa', 'plastic_bag']

class BenchmarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Ferramenta de Benchmark: YOLO vs Pipeline Híbrido")
        self.geometry("1000x700")
        
        self.yolo_model = None
        self.efficientnet_model = None
        self.video_path = None
        self.is_running = False
        
        self._setup_ui()
        self._load_models()

    def _setup_ui(self):
        # Layout principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Painel de Controle
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.btn_select_video = ctk.CTkButton(self.control_frame, text="Selecionar Vídeo de Teste", command=self.select_video)
        self.btn_select_video.pack(side="left", padx=10, pady=10)
        
        self.lbl_video_path = ctk.CTkLabel(self.control_frame, text="Nenhum vídeo selecionado")
        self.lbl_video_path.pack(side="left", padx=10)
        
        self.btn_start = ctk.CTkButton(self.control_frame, text="INICIAR BENCHMARK", command=self.start_benchmark, state="disabled", fg_color="green")
        self.btn_start.pack(side="right", padx=10, pady=10)

        # Área de Resultados e Logs
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)
        
        # Logs (Esquerda)
        self.log_box = ctk.CTkTextbox(self.content_frame, width=400)
        self.log_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.log_box.insert("end", ">>> Aguardando seleção de vídeo...\n")
        
        # Gráfico (Direita)
        self.graph_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.graph_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.progress_bar.set(0)

    def _load_models(self):
        self.log("Carregando modelos... (Isso pode demorar um pouco)")
        self.update()
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH)
            self.log("INFO: Modelos carregados com sucesso.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar modelos: {e}")
            self.log(f"ERRO CRÍTICO: {e}")

    def log(self, text):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.lbl_video_path.configure(text=os.path.basename(path))
            self.btn_start.configure(state="normal")
            self.log(f"INFO: Vídeo selecionado: {os.path.basename(path)}")

    def start_benchmark(self):
        if not self.video_path: return
        self.btn_start.configure(state="disabled")
        self.btn_select_video.configure(state="disabled")
        self.is_running = True
        
        # Inicia thread para não travar a interface
        threading.Thread(target=self._run_tests, daemon=True).start()

    def _run_tests(self):
        results = {}
        
        # --- TESTE A: Apenas YOLO ---
        self.log("\n" + "="*30)
        self.log("INICIANDO TESTE A: Apenas YOLO")
        self.log("="*30)
        fps_yolo, latencia_yolo = self._process_video_loop(mode="yolo")
        results['YOLO'] = {'fps': fps_yolo, 'ms': latencia_yolo}
        
        # --- TESTE B: Pipeline Híbrido ---
        self.log("\n" + "="*30)
        self.log("INICIANDO TESTE B: Pipeline Híbrido")
        self.log("="*30)
        fps_hibrido, latencia_hibrido = self._process_video_loop(mode="pipeline")
        results['Hibrido'] = {'fps': fps_hibrido, 'ms': latencia_hibrido}
        
        self.log("\n" + "="*30)
        self.log("BENCHMARK CONCLUÍDO")
        self.log("="*30)
        
        # Atualiza GUI com gráficos
        self.after(0, lambda: self._generate_report(results))
        self.after(0, lambda: self.btn_start.configure(state="normal"))
        self.after(0, lambda: self.btn_select_video.configure(state="normal"))
        self.is_running = False

    def _process_video_loop(self, mode):
        cap = cv2.VideoCapture(self.video_path)

        # --- CONFIGURAÇÃO: Apenas 1100 frames para teste rápido ---
        MAX_FRAMES_TO_TEST = 1100
        
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Garante que não vai tentar ler mais frames do que o vídeo tem
        total_frames = min(total_frames_video, MAX_FRAMES_TO_TEST)
        
        processed_frames = 0
        start_time = time.time()
        total_inference_time = 0
        
        self.log(f"--> Iniciando teste de {total_frames} frames...")

        # Pular alguns frames iniciais para evitar tela preta (se o vídeo for longo)
        if total_frames_video > 100:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)

        while cap.isOpened():
            if processed_frames >= total_frames:
                break

            ret, frame = cap.read()
            if not ret: break
            
            t_start_frame = time.time()
            
            # 1. Inferência YOLO
            results = self.yolo_model(frame, verbose=False)[0]
            
            if mode == "pipeline":
                boxes = results.boxes.data.tolist()
                
                # --- DEBUG: Ver quantos objetos ele achou no primeiro frame ---
                if processed_frames == 0:
                    self.log(f"   [DEBUG] Frame 1: YOLO detectou {len(boxes)} objetos.")
                
                for box in boxes:
                    # --- CORREÇÃO DO ERRO AQUI ---
                    # Extração direta e segura, sem usar map() + list()
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    # (Ignoramos score e class_id pois não precisamos deles para medir a velocidade)
                    
                    # Recorte seguro (evita erro se a caixa sair da imagem)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Se o recorte for válido, roda a EfficientNet
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        
                        # Pré-processamento e Inferência EfficientNet
                        img_resized = cv2.resize(crop, EFFICIENTNET_IMG_SIZE)
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        # Adiciona dimensão do batch (1, 224, 224, 3)
                        img_input = np.expand_dims(img_rgb.astype(np.float32), axis=0)
                        
                        self.efficientnet_model.predict(img_input, verbose=0)
            
            t_end_frame = time.time()
            frame_duration = t_end_frame - t_start_frame
            total_inference_time += frame_duration
            
            processed_frames += 1
            
            # Atualiza barra e log a cada frame para você ver que está rodando
            self.log(f"Frame {processed_frames}/{total_frames} processado em {frame_duration:.3f}s")
            self.progress_bar.set(processed_frames / total_frames)
            self.update() # Mantém a janela responsiva

        cap.release()
        
        if processed_frames == 0: return 0, 0

        # Cálculos Finais
        avg_fps = processed_frames / total_inference_time
        avg_latency_ms = (total_inference_time / processed_frames) * 1000
        
        self.log(f"--> RESULTADO FINAL DO MODO: {avg_fps:.2f} FPS")
        
        return avg_fps, avg_latency_ms

    def _generate_report(self, results):
        # Limpa gráficos anteriores
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        yolo_fps = results['YOLO']['fps']
        hybrid_fps = results['Hibrido']['fps']
        
        # Cria figura matplotlib
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        models = ['YOLO Isolado', 'Pipeline Híbrido']
        fps_values = [yolo_fps, hybrid_fps]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = ax.bar(models, fps_values, color=colors)
        ax.set_ylabel('FPS (Quadros por Segundo)')
        ax.set_title('Comparativo de Velocidade de Processamento')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adiciona valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f} FPS',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Canvas para Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Texto para copiar
        summary = (
            f"\n--- DADOS FINAIS PARA O TRABALHO ---\n"
            f"Hardware Utilizado: [Seu PC]\n"
            f"Vídeo de Entrada: {os.path.basename(self.video_path)}\n\n"
            f"TABELA 1: Comparativo de Desempenho\n"
            f"| Abordagem | FPS Médio | Latência (ms) |\n"
            f"| :--- | :---: | :---: |\n"
            f"| YOLO Isolado | {yolo_fps:.2f} FPS | {results['YOLO']['ms']:.2f} ms |\n"
            f"| Pipeline Híbrido | {hybrid_fps:.2f} FPS | {results['Hibrido']['ms']:.2f} ms |\n"
            f"--------------------------------------"
        )
        self.log(summary)

if __name__ == "__main__":
    app = BenchmarkApp()
    app.mainloop()