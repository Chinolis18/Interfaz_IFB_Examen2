import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from scipy import signal
from scipy.stats import entropy


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Procesamiento de Señales IMU")
        self.root.geometry("800x600")
        self.root.configure(bg="#2C3E50")
        
        # Frame principal
        main_frame = tk.Frame(root, bg="#2C3E50")
        main_frame.pack(expand=True, fill="both", padx=50, pady=50)
        
        # Logo (placeholder - puedes reemplazar con tu imagen)
        logo_frame = tk.Frame(main_frame, bg="#314457")
        logo_frame.pack(pady=20)
        
        try:
            # Intenta cargar un logo si existe
            logo_img = Image.open("D:/Descargas/universidad-nacional-autonoma-de-mexico-logo-png_seeklogo-387362.png")
            logo_img = logo_img.resize((160, 160))
            logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(logo_frame, image=logo_photo, bg="#2C3E50")
            logo_label.image = logo_photo
            logo_label.pack()
        except:
            # Si no existe el logo, muestra un placeholder
            logo_label = tk.Label(logo_frame, text="🎯", font=("Arial", 72), 
                                bg="#2C3E50", fg="#ECF0F1")
            logo_label.pack()
        
        # Título
        title_label = tk.Label(main_frame, 
                              text="Sistema de Procesamiento\nde Señales en una IU",
                              font=("Arial", 27, "bold"),
                              bg="#2C3E50", fg="#ECF3F5")
        title_label.pack(pady=20)
        
        # Subtítulo
        subtitle_label = tk.Label(main_frame, 
                                 text=" Examen Parcial II: Módulo II ",
                                 font=("Arial", 15),
                                 bg="#2C3E50", fg="#D8DDE0")
        subtitle_label.pack(pady=5)
        
        # Autor
        author_label = tk.Label(main_frame, 
                               text="Alumna: Valeria Rodriguez Ponce",
                               font=("Arial", 13, "italic"),
                               bg="#2C3E50", fg="#BAC3C4")
        author_label.pack(pady=10)
        
        # Frame para botones
        button_frame = tk.Frame(main_frame, bg="#2C3E50")
        button_frame.pack(pady=40)
        
        # Botón Módulo A
        btn_modulo_a = tk.Button(button_frame,
                                text=" MÓDULO A \n Cargar y procesar archivo",
                                font=("Arial", 14, "bold"),
                                bg="#C94CFA", fg="white",
                                activebackground="#7F3CEC",
                                width=25, height=4,
                                cursor="hand2",
                                command=self.open_module_a)
        btn_modulo_a.grid(row=0, column=0, padx=20)
        
        # Botón Módulo B
        btn_modulo_b = tk.Button(button_frame,
                                text=" MÓDULO B\nAdquisición en Vivo",
                                font=("Arial", 14, "bold"),
                                bg="#C52DCA", fg="white",
                                activebackground="#DA2DAE",
                                width=25, height=4,
                                cursor="hand2",
                                command=self.open_module_b)
        btn_modulo_b.grid(row=0, column=1, padx=20)
        
        # Footer
        footer_label = tk.Label(main_frame, 
                               text="Es la v3.0 - 2025 porque en MATLAB no se pudo",
                               font=("Arial", 9),
                               bg="#2C3E50", fg="#7F8C8D")
        footer_label.pack(side="bottom", pady=10)
    
    def open_module_a(self):
        module_a_window = tk.Toplevel(self.root)
        ModuleA(module_a_window)
    
    def open_module_b(self):
        module_b_window = tk.Toplevel(self.root)
        ModuleB(module_b_window)

#-----------------------Modulo A ----------------------------------------------
class ModuleA:
    def __init__(self, window):
        self.window = window
        self.window.title("Módulo A - Cargar y Procesar Archivo")
        self.window.geometry("1600x900")
        self.window.configure(bg="#ECF0F1")
        
        self.data = None
        self.filtered_data = None
        self.features = None
        self.predictions = None
        self.sampling_rate = 200  # Hz
        
        # Header
        header = tk.Frame(window, bg="#3498DB", height=80)
        header.pack(fill="x")
        
        title = tk.Label(header, text="📁 MÓDULO A: Cargar y Procesar Archivo",
                        font=("Arial", 20, "bold"), bg="#3498DB", fg="white")
        title.pack(pady=25)
        
        # Main container
        main_container = tk.Frame(window, bg="#ECF0F1")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Control and Tables
        left_frame = tk.Frame(main_container, bg="#ECF0F1")
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Control Panel
        control_frame = tk.Frame(left_frame, bg="#ECF0F1")
        control_frame.pack(fill="x", pady=5)
        
        btn_load = tk.Button(control_frame, text="Cargar Archivo CSV/EXCEL/TXT",
                            font=("Arial", 11), bg="#2ECC71", fg="white",
                            command=self.load_file, width=22)
        btn_load.pack(side="left", padx=5)
        
        btn_process = tk.Button(control_frame, text="Procesar Datos",
                               font=("Arial", 11), bg="#E74C3C", fg="white",
                               command=self.process_data, width=22)
        btn_process.pack(side="left", padx=5)
        
        self.status_label = tk.Label(control_frame, text="Estado: Esperando archivo...",
                                     font=("Arial", 10), bg="#19444E", fg="white")
        self.status_label.pack(side="left", padx=15)
        
        # Notebook para tabs
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill="both", expand=True, pady=5)
        
        # Tab 1: Vista de datos crudos
        self.tab_raw = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_raw, text="Datos Crudos")
        
        # Tab 2: Características
        self.tab_features = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_features, text="Características")
        
        # Tab 3: Resultados
        self.tab_results = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_results, text="Resultados")
        
        self.setup_tabs()
        
        # Right side - Graphs
        right_frame = tk.Frame(main_container, bg="white", width=600)
        right_frame.pack(side="right", fill="both", expand=False, padx=5)
        right_frame.pack_propagate(False)
        
        self.setup_graphs(right_frame)
    
    def setup_tabs(self):
        # Tab de datos crudos
        self.raw_text = tk.Text(self.tab_raw, wrap="none", height=20, font=("Courier", 9))
        raw_scroll_y = tk.Scrollbar(self.tab_raw, command=self.raw_text.yview)
        raw_scroll_x = tk.Scrollbar(self.tab_raw, orient="horizontal", 
                                    command=self.raw_text.xview)
        self.raw_text.configure(yscrollcommand=raw_scroll_y.set,
                               xscrollcommand=raw_scroll_x.set)
        
        raw_scroll_y.pack(side="right", fill="y")
        raw_scroll_x.pack(side="bottom", fill="x")
        self.raw_text.pack(fill="both", expand=True)
        
        # Tab de características
        self.features_text = tk.Text(self.tab_features, wrap="none", height=20, font=("Courier", 9))
        feat_scroll_y = tk.Scrollbar(self.tab_features, 
                                     command=self.features_text.yview)
        feat_scroll_x = tk.Scrollbar(self.tab_features, orient="horizontal",
                                     command=self.features_text.xview)
        self.features_text.configure(yscrollcommand=feat_scroll_y.set,
                                    xscrollcommand=feat_scroll_x.set)
        
        feat_scroll_y.pack(side="right", fill="y")
        feat_scroll_x.pack(side="bottom", fill="x")
        self.features_text.pack(fill="both", expand=True)
        
        # Tab de resultados
        self.results_text = tk.Text(self.tab_results, wrap="word", height=20,
                                   font=("Courier", 10))
        results_scroll = tk.Scrollbar(self.tab_results, 
                                     command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        results_scroll.pack(side="right", fill="y")
        self.results_text.pack(fill="both", expand=True)
    
    def setup_graphs(self, parent):
        # Frame para las gráficas
        graph_frame = tk.Frame(parent, bg="white")
        graph_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Crear figura con dos subplots
        self.fig = Figure(figsize=(8, 10), facecolor='white')
        
        # Gráfica 1: Datos crudos
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Acelerómetro - Datos Crudos", fontsize=12, fontweight='bold')
        self.ax1.set_xlabel("Tiempo (s)", fontsize=10)
        self.ax1.set_ylabel("Aceleración (g)", fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(['Ax', 'Ay', 'Az'], loc='upper right', fontsize=8)
        
        # Gráfica 2: Datos filtrados
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Acelerómetro - Datos Filtrados (Notch 50Hz)", fontsize=12, fontweight='bold')
        self.ax2.set_xlabel("Tiempo (s)", fontsize=10)
        self.ax2.set_ylabel("Aceleración (g)", fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(['Ax', 'Ay', 'Az'], loc='upper right', fontsize=8)
        
        self.fig.tight_layout(pad=3.0)
        
        # Canvas para matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=[
                ("Archivos compatibles", "*.csv;*.html;*.xls;*.xlsx;*.txt"),
                ("Archivos CSV", "*.csv"),
                ("Archivos HTML", "*.html"),
                ("Archivos Excel", "*.xls;*.xlsx"),
                ("Archivos de texto", "*.txt"),
                ("Todos los archivos", "*.*")
            ]
        )

        if filename:
            try:
                # Detectar tipo de archivo
                if filename.endswith('.csv'):
                    self.data = pd.read_csv(filename)
                elif filename.endswith('.html'):
                    tables = pd.read_html(filename)
                    if len(tables) > 0:
                        self.data = tables[0]
                    else:
                        raise ValueError("No se encontraron tablas en el archivo HTML")
                elif filename.endswith(('.xls', '.xlsx')):
                    self.data = pd.read_excel(filename)
                elif filename.endswith('.txt'):
                    # Detectar separador automáticamente (coma, tab, punto y coma)
                    with open(filename, 'r') as f:
                        sample = f.read(1024)
                    if '\t' in sample:
                        sep = '\t'
                    elif ';' in sample:
                        sep = ';'
                    else:
                        sep = ','
                    self.data = pd.read_csv(filename, sep=sep)
                else:
                    raise ValueError("Formato de archivo no compatible")

                # Verificar columnas necesarias
                required_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
                if not all(col in self.data.columns for col in required_cols):
                    messagebox.showwarning(
                        "Advertencia",
                        f"El archivo debe contener las columnas: {', '.join(required_cols)}"
                    )
                    return

                # Mostrar datos crudos
                self.raw_text.delete(1.0, tk.END)
                self.raw_text.insert(1.0, self.data.head(100).to_string())

                # Graficar datos crudos
                self.plot_raw_data()

                self.status_label.config(
                    text=f"Estado: Archivo cargado - {len(self.data)} muestras"
                )
                messagebox.showinfo("Éxito", "Archivo cargado correctamente")

            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar archivo:\n{str(e)}")

    def plot_raw_data(self):
        """Grafica los datos crudos del acelerómetro"""
        if self.data is not None:
            time_vector = np.arange(len(self.data)) / self.sampling_rate
            
            self.ax1.clear()
            self.ax1.plot(time_vector, self.data['Ax'], 'r-', label='Ax', linewidth=0.8)
            self.ax1.plot(time_vector, self.data['Ay'], 'g-', label='Ay', linewidth=0.8)
            self.ax1.plot(time_vector, self.data['Az'], 'b-', label='Az', linewidth=0.8)
            self.ax1.set_title("Acelerómetro - Datos Crudos", fontsize=12, fontweight='bold')
            self.ax1.set_xlabel("Tiempo (s)", fontsize=10)
            self.ax1.set_ylabel("Aceleración (g)", fontsize=10)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend(loc='upper right', fontsize=8)
            
            self.canvas.draw()
    
    def plot_filtered_data(self):
        """Grafica los datos filtrados del acelerómetro"""
        if self.filtered_data is not None:
            time_vector = np.arange(len(self.filtered_data)) / self.sampling_rate
            
            self.ax2.clear()
            self.ax2.plot(time_vector, self.filtered_data['Ax'], 'r-', label='Ax', linewidth=0.8)
            self.ax2.plot(time_vector, self.filtered_data['Ay'], 'g-', label='Ay', linewidth=0.8)
            self.ax2.plot(time_vector, self.filtered_data['Az'], 'b-', label='Az', linewidth=0.8)
            self.ax2.set_title("Acelerómetro - Datos Filtrados (Notch 50Hz)", fontsize=12, fontweight='bold')
            self.ax2.set_xlabel("Tiempo (s)", fontsize=10)
            self.ax2.set_ylabel("Aceleración (g)", fontsize=10)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend(loc='upper right', fontsize=8)
            
            self.canvas.draw()
    
    def process_data(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar un archivo")
            return
        
        try:
            self.status_label.config(text="Estado: Procesando...")
            self.window.update()
            
            # 1. Aplicar filtro Notch de 50 Hz
            self.filtered_data = self.apply_notch_filter(self.data)
            self.plot_filtered_data()
            
            # 2. Segmentación en ventanas de 50 muestras (0.25s a 200Hz)
            windows = self.segment_windows(self.filtered_data, window_size=50)
            
            # 3. Extracción de características
            self.features = self.extract_features(windows)
            
            # 4. Clasificación basada en máximos y mínimos
            self.predictions = self.classify(self.features)
            
            # Mostrar características
            self.display_features()
            
            # Mostrar resultados
            self.show_results()
            
            self.status_label.config(text="Estado: Procesamiento completado")
            messagebox.showinfo("Éxito", "Datos procesados correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el procesamiento:\n{str(e)}")
            self.status_label.config(text="Estado: Error en procesamiento")
    
    def apply_notch_filter(self, data):
        """Aplica filtro Notch de 50 Hz a todas las señales"""
        filtered_data = data.copy()
        
        # Frecuencia de corte del notch
        notch_freq = 50.0
        quality_factor = 30.0
        
        # Diseñar filtro notch
        b, a = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
        
        # Aplicar filtro a cada eje
        for col in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
            if col in filtered_data.columns:
                filtered_data[col] = signal.filtfilt(b, a, filtered_data[col])
        
        return filtered_data
    
    def segment_windows(self, data, window_size=50):
        """Segmenta los datos en ventanas de 50 muestras (sin solapamiento)"""
        windows = []
        num_windows = len(data) // window_size
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window = data.iloc[start_idx:end_idx]
            windows.append(window)
        
        return windows
    
    def extract_features(self, windows):
        """Extrae características específicas de cada ventana"""
        features_list = []
        
        for idx, window in enumerate(windows):
            # Extraer datos de cada eje
            ax = window['Ax'].values
            ay = window['Ay'].values
            az = window['Az'].values
            gx = window['Gx'].values
            gy = window['Gy'].values
            gz = window['Gz'].values
            
            # Calcular características
            features = {
                'Ventana': idx + 1,
                # Media
                'Media_Ax': np.mean(ax),
                'Media_Ay': np.mean(ay),
                'Media_Az': np.mean(az),
                
                # RMS en Ax
                'RMS_Ax': np.sqrt(np.mean(ax**2)),
                
                # Varianza en Ay y Gy
                'Varianza_Ay': np.var(ay),
                'Varianza_Gy': np.var(gy),
                
                # Energía Total (suma de energías de todos los ejes)
                'Energia_Total': np.sum(ax**2) + np.sum(ay**2) + np.sum(az**2),
                
                # Jerk promedio (derivada de la aceleración)
                'Jerk_Promedio_Ax': np.mean(np.abs(np.diff(ax))),
                'Jerk_Promedio_Ay': np.mean(np.abs(np.diff(ay))),
                'Jerk_Promedio_Az': np.mean(np.abs(np.diff(az))),
                
                # Entropía en Az
                'Entropia_Az': self.calculate_entropy(az),
                
                # Energía rotacional (suma de energías del giroscopio)
                'Energia_Rotacional': np.sum(gx**2) + np.sum(gy**2) + np.sum(gz**2),
                
                # Máximos y Mínimos (para clasificación)
                'Max_Ax': np.max(ax),
                'Min_Ax': np.min(ax),
                'Max_Ay': np.max(ay),
                'Min_Ay': np.min(ay),
                'Max_Az': np.max(az),
                'Min_Az': np.min(az),
                'Max_Gx': np.max(gx),
                'Min_Gx': np.min(gx),
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def calculate_entropy(self, signal_data):
        """Calcula la entropía de una señal"""
        # Discretizar la señal en bins
        hist, _ = np.histogram(signal_data, bins=20, density=True)
        # Normalizar y evitar log(0)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        return entropy(hist)
    
    def classify(self, features):
        """
        Clasifica en 'Saltando' o 'Estático' basado en máximos y mínimos
        Criterios simples de clasificación
        """
        predictions = []
        
        for idx, row in features.iterrows():
            # Calcular rangos de variación
            rango_ax = row['Max_Ax'] - row['Min_Ax']
            rango_ay = row['Max_Ay'] - row['Min_Ay']
            rango_az = row['Max_Az'] - row['Min_Az']
            rango_gx = row['Max_Gx'] - row['Min_Gx']
            
            # Criterio de clasificación (puedes ajustar estos umbrales)
            # Si hay alta variación en aceleración y rotación -> Saltando
            # Si hay baja variación -> Estático
            
            if (rango_ax > 0.5 or rango_ay > 0.5 or rango_az > 1.0) and rango_gx > 50:
                predictions.append('Saltando')
            else:
                predictions.append('Estático')
        
        return predictions
    
    def display_features(self):
        """Muestra las características en el tab"""
        self.features_text.delete(1.0, tk.END)
        
        # Mostrar solo las primeras 50 ventanas para no saturar
        display_features = self.features.head(50)
        self.features_text.insert(1.0, display_features.to_string(index=False))
        
        if len(self.features) > 50:
            self.features_text.insert(tk.END, f"\n\n... ({len(self.features) - 50} ventanas más)")
    
    def show_results(self):
        """Muestra resultados y métricas"""
        self.results_text.delete(1.0, tk.END)
        
        # Contar predicciones
        unique, counts = np.unique(self.predictions, return_counts=True)
        pred_dict = dict(zip(unique, counts))
        
        results = f"""
╔═══════════════════════════════════════════════════════════════════╗
║              RESULTADOS DEL PROCESAMIENTO                         ║
╚═══════════════════════════════════════════════════════════════════╝

📊 PIPELINE APLICADO:
─────────────────────────────────────────────────────────────────────
1. ✓ Filtro Notch de 50 Hz aplicado
2. ✓ Segmentación en ventanas de 50 muestras (0.25s @ 200Hz)
3. ✓ Extracción de características:
   • Media (Ax, Ay, Az)
   • RMS (Ax)
   • Varianza (Ay, Gy)
   • Energía Total
   • Jerk Promedio (Ax, Ay, Az)
   • Entropía (Az)
   • Energía Rotacional
   • Máximos y Mínimos (todos los ejes)

4. ✓ Clasificación basada en máximos y mínimos

📈 RESUMEN DE RESULTADOS:
─────────────────────────────────────────────────────────────────────
Total de muestras procesadas: {len(self.data)}
Total de ventanas analizadas: {len(self.predictions)}
Duración total: {len(self.data)/self.sampling_rate:.2f} segundos

🏷️  DISTRIBUCIÓN DE CLASES:
─────────────────────────────────────────────────────────────────────
"""
        
        for clase in ['Estático', 'Saltando']:
            count = pred_dict.get(clase, 0)
            percentage = (count / len(self.predictions)) * 100 if len(self.predictions) > 0 else 0
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            results += f"{clase:12s}: {count:4d} ventanas ({percentage:5.1f}%) {bar}\n"
        
        results += f"""
─────────────────────────────────────────────────────────────────────
📊 ESTADÍSTICAS DE CARACTERÍSTICAS (Promedio):
─────────────────────────────────────────────────────────────────────
RMS Ax:                {self.features['RMS_Ax'].mean():.4f} g
Energía Total:         {self.features['Energia_Total'].mean():.2f}
Energía Rotacional:    {self.features['Energia_Rotacional'].mean():.2f}
Jerk Promedio Ax:      {self.features['Jerk_Promedio_Ax'].mean():.4f}
Entropía Az:           {self.features['Entropia_Az'].mean():.4f}

🎯 CRITERIOS DE CLASIFICACIÓN:
─────────────────────────────────────────────────────────────────────
• ESTÁTICO: Baja variación en aceleración y rotación
• SALTANDO: Alta variación en aceleración (>0.5g) y rotación (>50°/s)

💡 Nota: Los umbrales pueden ajustarse según tus datos específicos
"""
        
        self.results_text.insert(1.0, results)


#---------------MODULO B ------------------------------------------------------------------------
class ModuleB:
    def __init__(self, window):
        self.window = window
        self.window.title("Módulo B - Adquisición en Vivo")
        self.window.geometry("1200x800")
        self.window.configure(bg="#ECF0F1")
        
        self.is_acquiring = False
        self.data_buffer = []
        
        # Header
        header = tk.Frame(window, bg="#2ECC71", height=80)
        header.pack(fill="x")
        
        title = tk.Label(header, text="📡 MÓDULO B: Adquisición en Vivo",
                        font=("Arial", 20, "bold"), bg="#2ECC71", fg="white")
        title.pack(pady=25)
        
        # Control Panel
        control_frame = tk.Frame(window, bg="#ECF0F1")
        control_frame.pack(fill="x", padx=20, pady=10)
        
        self.btn_start = tk.Button(control_frame, text="▶ Iniciar Adquisición",
                                   font=("Arial", 12), bg="#27AE60", fg="white",
                                   command=self.start_acquisition, width=20)
        self.btn_start.pack(side="left", padx=5)
        
        self.btn_stop = tk.Button(control_frame, text="⏹ Detener",
                                  font=("Arial", 12), bg="#E74C3C", fg="white",
                                  command=self.stop_acquisition, width=20,
                                  state="disabled")
        self.btn_stop.pack(side="left", padx=5)
        
        btn_save = tk.Button(control_frame, text="💾 Guardar Datos",
                            font=("Arial", 12), bg="#3498DB", fg="white",
                            command=self.save_data, width=20)
        btn_save.pack(side="left", padx=5)
        
        btn_process = tk.Button(control_frame, text="⚙ Procesar",
                               font=("Arial", 12), bg="#9B59B6", fg="white",
                               command=self.process_acquired_data, width=20)
        btn_process.pack(side="left", padx=5)
        
        # Status
        self.status_label = tk.Label(control_frame, 
                                     text="Estado: Listo para adquirir",
                                     font=("Arial", 10, "bold"), bg="#ECF0F1")
        self.status_label.pack(side="left", padx=20)
        
        # Gráficas en tiempo real
        self.setup_plots()
    
    def setup_plots(self):
        plot_frame = tk.Frame(self.window, bg="white")
        plot_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.fig = Figure(figsize=(12, 6))
        
        # Acelerómetro
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Acelerómetro (Ax)", fontsize=12, fontweight='bold')
        self.ax1.set_ylabel("Aceleración (g)")
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2)
        
        # Giroscopio
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Giroscopio (Gx)", fontsize=12, fontweight='bold')
        self.ax2.set_xlabel("Muestras")
        self.ax2.set_ylabel("Velocidad Angular (°/s)")
        self.ax2.grid(True, alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2)
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def start_acquisition(self):
        self.is_acquiring = True
        self.data_buffer = []
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_label.config(text="Estado: Adquiriendo datos...", fg="green")
        
        # Iniciar thread de adquisición
        self.acquisition_thread = threading.Thread(target=self.acquire_data)
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()
    
    def acquire_data(self):
        """
        AQUÍ IMPLEMENTA LA CONEXIÓN CON EL SENSOR WT9011DCL
        Este es un ejemplo simulado
        """
        while self.is_acquiring:
            # Simulación de datos del sensor
            # Reemplaza esto con la lectura real del sensor
            ax = np.sin(len(self.data_buffer) * 0.1) + np.random.normal(0, 0.1)
            ay = np.cos(len(self.data_buffer) * 0.1) + np.random.normal(0, 0.1)
            az = 1.0 + np.random.normal(0, 0.05)
            gx = np.sin(len(self.data_buffer) * 0.05) * 50 + np.random.normal(0, 5)
            gy = np.cos(len(self.data_buffer) * 0.05) * 50 + np.random.normal(0, 5)
            gz = np.random.normal(0, 10)
            
            sample = {
                'Ax': ax, 'Ay': ay, 'Az': az,
                'Gx': gx, 'Gy': gy, 'Gz': gz,
                'timestamp': time.time()
            }
            
            self.data_buffer.append(sample)
            
            # Actualizar gráficas
            self.window.after(0, self.update_plots)
            
            time.sleep(0.01)  # 100 Hz de frecuencia de muestreo
    
    def update_plots(self):
        if len(self.data_buffer) > 0:
            # Mostrar últimas 500 muestras
            window_size = 500
            data_to_plot = self.data_buffer[-window_size:]
            
            ax_data = [d['Ax'] for d in data_to_plot]
            gx_data = [d['Gx'] for d in data_to_plot]
            
            self.line1.set_data(range(len(ax_data)), ax_data)
            self.line2.set_data(range(len(gx_data)), gx_data)
            
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            self.canvas.draw()
    
    def stop_acquisition(self):
        self.is_acquiring = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_label.config(text=f"Estado: Detenido - {len(self.data_buffer)} muestras capturadas")
        messagebox.showinfo("Info", f"Adquisición detenida\nMuestras capturadas: {len(self.data_buffer)}")
    
    def save_data(self):
        if not self.data_buffer:
            messagebox.showwarning("Advertencia", "No hay datos para guardar")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.data_buffer)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Éxito", f"Datos guardados en:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar:\n{str(e)}")
    
    def process_acquired_data(self):
        if not self.data_buffer:
            messagebox.showwarning("Advertencia", "No hay datos para procesar")
            return
        
        # Procesar con el mismo pipeline del Módulo A
        messagebox.showinfo("Procesamiento", 
                          "Procesando datos adquiridos con el pipeline...\n"
                          "Implementa aquí la misma lógica del Módulo A")


# Ejecutar aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()