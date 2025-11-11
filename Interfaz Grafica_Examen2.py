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
import scipy.io as sio
from pipeline_module import DataPipeline  # ← Pipeline externo




class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Procesamiento de Señales IMU")
        self.root.geometry("900x700")
        self.root.configure(bg="#2C3E50")
        
        # Frame principal
        main_frame = tk.Frame(root, bg="#2C3E50")
        main_frame.pack(expand=True, fill="both", padx=50, pady=50)
        
        # Logo (placeholder - puedes reemplazar con tu imagen)
        logo_frame = tk.Frame(main_frame, bg="#314457")
        logo_frame.pack(pady=15)
        
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
                                bg="#D348D8", fg="white",
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
        self.window.configure(bg="#144855")
        self.data = None
        self.sampling_rate = 200  # Definido para evitar errores
        # self.pipeline = DataPipeline(sampling_rate=200)  # Comentado ya que no está disponible

        # Header
        header = tk.Frame(window, bg="#5634F1", height=80)
        header.pack(fill="x")
        title = tk.Label(header, text="MÓDULO A: Cargar y Procesar Archivo",
                         font=("Arial", 20, "bold"), bg="#5C22E2", fg="white")
        title.pack(pady=25)

        # Main container
        main_container = tk.Frame(window, bg="#1A6072")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left section
        left_frame = tk.Frame(main_container, bg="#17677A")
        left_frame.pack(side="left", fill="both", expand=True, padx=5)

        # Controls
        control_frame = tk.Frame(left_frame, bg="#127E99")
        control_frame.pack(fill="x", pady=5)

        tk.Button(control_frame, text="Cargar Archivo",
                  font=("Arial", 11), bg="#723FE7", fg="white",
                  command=self.load_file, width=22).pack(side="left", padx=5)
        tk.Button(control_frame, text="Procesar Datos",
                  font=("Arial", 11), bg="#CB3CE7", fg="white",
                  command=self.process_data, width=22).pack(side="left", padx=5)

        self.status_label = tk.Label(control_frame, text="Estado: Esperando archivo...",
                                     font=("Arial", 10), bg="#19444E", fg="white")
        self.status_label.pack(side="left", padx=15)

        # Tabs
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill="both", expand=True, pady=5)
        self.tab_raw = tk.Frame(self.notebook, bg="white")
        self.tab_features = tk.Frame(self.notebook, bg="white")
        self.tab_results = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_raw, text="Datos Crudos")
        self.notebook.add(self.tab_features, text="Características")
        self.notebook.add(self.tab_results, text="Resultados")

        self.setup_tabs()

        # Graph area
        right_frame = tk.Frame(main_container, bg="white", width=600)
        right_frame.pack(side="right", fill="both", expand=False, padx=5)
        right_frame.pack_propagate(False)
        self.setup_graphs(right_frame)

    # --- UI Setup ---

    def setup_tabs(self):
        self.raw_text = tk.Text(self.tab_raw, wrap="none", height=20, font=("Courier", 9))
        self.raw_text.pack(fill="both", expand=True)
        self.features_text = tk.Text(self.tab_features, wrap="none", height=20, font=("Courier", 9))
        self.features_text.pack(fill="both", expand=True)
        self.results_text = tk.Text(self.tab_results, wrap="word", height=20, font=("Courier", 10))
        self.results_text.pack(fill="both", expand=True)

    def setup_graphs(self, parent):
        self.fig = Figure(figsize=(8, 10), facecolor='white')
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Datos Crudos", fontsize=10)
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Datos Filtrados", fontsize=10)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Funciones principales ---

    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo",
            parent=self.window,
            filetypes=[
                ("Archivos compatibles", "*.csv;*.xlsx;*.txt;*.mat"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx"),
                ("Texto", "*.txt"),
                ("MATLAB", "*.mat")
            ]
        )

        if not filename:
            return

        try:
            if filename.endswith('.csv'):
                # Leer el archivo CSV con manejo de metadatos
                with open(filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Buscar la línea que contiene los encabezados reales
                data_start_line = 0
                for i, line in enumerate(lines):
                    if 'address' in line and 'Time(s)' in line and 'ax(g)' in line:
                        data_start_line = i
                        break
                    elif len(line.split('\t')) >= 7:  # Buscar líneas con suficientes columnas
                        data_start_line = i
                        break
                
                # Leer los datos saltando las líneas de metadatos
                self.data = pd.read_csv(filename, 
                                      skiprows=data_start_line, 
                                      sep='\t',
                                      encoding='utf-8')
                
                # Limpiar nombres de columnas
                self.data.columns = self.data.columns.str.strip()
                
                # Estandarizar nombres de columnas
                self.standardize_columns()

            elif filename.endswith('.xlsx'):
                # Para Excel, leer todo y luego buscar dónde empiezan los datos
                raw_data = pd.read_excel(filename, header=None)
                
                # Buscar la fila que contiene los encabezados
                header_row = 0
                for i, row in raw_data.iterrows():
                    if any('ax(g)' in str(cell) for cell in row.values):
                        header_row = i
                        break
                    elif any('address' in str(cell) for cell in row.values):
                        header_row = i
                        break
                
                # Leer nuevamente con la fila correcta como encabezado
                self.data = pd.read_excel(filename, header=header_row)
                
                # Limpiar nombres de columnas
                self.data.columns = self.data.columns.str.strip()
                
                # Estandarizar nombres de columnas
                self.standardize_columns()

            elif filename.endswith('.txt'):
                self.data = pd.read_csv(filename, sep='\t', engine='python')
                self.standardize_columns()
            elif filename.endswith('.mat'):
                mat_data = sio.loadmat(filename)
                keys = [k for k in mat_data.keys() if not k.startswith("__")]
                if len(keys) == 1:
                    raw = mat_data[keys[0]]
                else:
                    raw = next((mat_data[k] for k in keys if isinstance(mat_data[k], np.ndarray)), None)
                if raw is None:
                    raise ValueError("No se encontró matriz de datos válida en el archivo .mat")
                self.data = pd.DataFrame(raw, columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])

            # Limpiar datos (eliminar filas con NaN)
            self.data = self.data.dropna()

            # Mostrar información del archivo cargado
            self.raw_text.delete(1.0, tk.END)
            self.raw_text.insert(1.0, f"Archivo: {filename}\n")
            self.raw_text.insert(tk.END, f"Columnas: {', '.join(self.data.columns)}\n")
            self.raw_text.insert(tk.END, f"Total de muestras: {len(self.data)}\n\n")
            self.raw_text.insert(tk.END, "Primeras 200 filas:\n")
            self.raw_text.insert(tk.END, self.data.head(200).to_string())
            
            self.plot_raw_data()
            self.status_label.config(text=f"Archivo cargado ({len(self.data)} muestras)")
            messagebox.showinfo("Éxito", "Archivo cargado correctamente", parent=self.window)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{str(e)}", parent=self.window)

    def standardize_columns(self):
        """Estandariza los nombres de columnas para el procesamiento"""
        if self.data is None:
            return
            
        # Mapeo de nombres de columnas esperados
        column_mapping = {
            'Ax': ['ax(g)', 'ax', 'aceleracion_x', 'acel_x'],
            'Ay': ['ay(g)', 'ay', 'aceleracion_y', 'acel_y'], 
            'Az': ['az(g)', 'az', 'aceleracion_z', 'acel_z'],
            'Gx': ['wx(deg/s)', 'wx', 'giroscopio_x', 'gyro_x'],
            'Gy': ['wy(deg/s)', 'wy', 'giroscopio_y', 'gyro_y'],
            'Gz': ['wz(deg/s)', 'wz', 'giroscopio_z', 'gyro_z']
        }
        
        available_columns = self.data.columns.tolist()
        renamed_columns = {}
        
        # Buscar coincidencias para cada columna esperada
        for new_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in available_columns:
                    renamed_columns[possible_name] = new_name
                    break
        
        # Renombrar las columnas que encontramos
        if renamed_columns:
            self.data = self.data.rename(columns=renamed_columns)
            print(f"Columnas renombradas: {renamed_columns}")
        
        # Verificar que tenemos las columnas mínimas requeridas
        required_columns = ['Ax', 'Ay', 'Az']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            available_cols = ", ".join(self.data.columns)
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}\nColumnas disponibles: {available_cols}")

    def plot_raw_data(self):
        if self.data is None:
            return
        
        # Crear vector de tiempo basado en la frecuencia de muestreo
        time = np.arange(len(self.data)) / self.sampling_rate
        
        self.ax1.clear()
        
        # Graficar datos de aceleración
        if 'Ax' in self.data.columns and 'Ay' in self.data.columns and 'Az' in self.data.columns:
            self.ax1.plot(time, self.data['Ax'], label='Ax', linewidth=1)
            self.ax1.plot(time, self.data['Ay'], label='Ay', linewidth=1)
            self.ax1.plot(time, self.data['Az'], label='Az', linewidth=1)
            self.ax1.legend()
            self.ax1.set_xlabel("Tiempo (s)")
            self.ax1.set_ylabel("Aceleración (g)")
            self.ax1.set_title("Datos Crudos de Aceleración")
        else:
            # Si no tenemos las columnas estandarizadas, graficar las primeras columnas numéricas
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            for i, col in enumerate(numeric_columns[:3]):  # Tomar máximo 3 columnas
                self.ax1.plot(time, self.data[col], label=col, linewidth=1)
            self.ax1.legend()
            self.ax1.set_xlabel("Tiempo (s)")
            self.ax1.set_ylabel("Valor")
            self.ax1.set_title("Datos Crudos (primeras 3 columnas numéricas)")

        self.canvas.draw()

    def process_data(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Carga un archivo antes de procesar", parent=self.window)
            return
        
        self.status_label.config(text="Procesando...")
        self.window.update()
        
        try:
            # Verificar que tenemos las columnas necesarias
            required_columns = ['Ax', 'Ay', 'Az']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Faltan columnas necesarias: {missing_columns}\nColumnas disponibles: {list(self.data.columns)}")
            
            print(f"Columnas disponibles para procesamiento: {list(self.data.columns)}")
            print(f"Primeras filas:\n{self.data.head()}")

            # Procesamiento básico
            results = self.run_basic_pipeline(self.data)
            
            # Mostrar resultados
            self.features_text.delete(1.0, tk.END)
            self.features_text.insert(1.0, results['features'].to_string())
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results['summary'])
            
            # Graficar datos filtrados
            self.ax2.clear()
            filtered_data = results['filtered']
            time_filtered = np.arange(len(filtered_data)) / self.sampling_rate
            
            self.ax2.plot(time_filtered, filtered_data['Ax'], label='Ax', linewidth=1)
            self.ax2.plot(time_filtered, filtered_data['Ay'], label='Ay', linewidth=1)
            self.ax2.plot(time_filtered, filtered_data['Az'], label='Az', linewidth=1)
            self.ax2.legend()
            self.ax2.set_xlabel("Tiempo (s)")
            self.ax2.set_ylabel("Aceleración (g)")
            self.ax2.set_title("Datos Filtrados")
            
            self.canvas.draw()
            self.status_label.config(text="Procesamiento completado")
            
        except Exception as e:
            error_msg = f"Error durante el procesamiento:\n{str(e)}\n\nColumnas disponibles: {list(self.data.columns) if self.data is not None else 'No data'}"
            messagebox.showerror("Error", error_msg, parent=self.window)
            self.status_label.config(text="Error en procesamiento")
            print(f"Error detallado: {e}")

    def run_basic_pipeline(self, data):
        """Función básica de procesamiento para reemplazar el pipeline externo"""
        # Aplicar filtro notch
        filtered_data = self.apply_notch_filter(data)
        
        # Segmentar en ventanas
        windows = self.segment_windows(filtered_data)
        
        # Extraer características
        features = self.extract_features(windows)
        
        # Clasificar
        predictions = self.classify(features)
        features['Prediccion'] = predictions
        
        # Crear resumen
        unique, counts = np.unique(predictions, return_counts=True)
        pred_dict = dict(zip(unique, counts))
        
        summary = f"RESUMEN DEL PROCESAMIENTO\n"
        summary += f"Ventanas procesadas: {len(windows)}\n"
        summary += f"Muestras totales: {len(data)}\n"
        summary += f"Tasa de muestreo: {self.sampling_rate} Hz\n\n"
        summary += "DISTRIBUCIÓN DE CLASES:\n"
        for clase, count in pred_dict.items():
            percentage = (count / len(predictions)) * 100
            summary += f"{clase}: {count} ventanas ({percentage:.1f}%)\n"
        
        return {
            'filtered': filtered_data,
            'features': features,
            'summary': summary
        }

    def apply_notch_filter(self, data):
        """Aplica filtro Notch de 50 Hz a todas las señales"""
        filtered_data = data.copy()
        
        # Frecuencia de corte del notch
        notch_freq = 50.0
        quality_factor = 30.0
        
        # Diseñar filtro notch
        b, a = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
        
        # Aplicar filtro a cada eje de aceleración
        for col in ['Ax', 'Ay', 'Az']:
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
            # Extraer datos de cada eje (solo aceleración si no hay giroscopio)
            ax = window['Ax'].values
            ay = window['Ay'].values
            az = window['Az'].values
            
            features = {
                'Ventana': idx + 1,
                # Media
                'Media_Ax': np.mean(ax),
                'Media_Ay': np.mean(ay),
                'Media_Az': np.mean(az),
                
                # RMS
                'RMS_Ax': np.sqrt(np.mean(ax**2)),
                'RMS_Ay': np.sqrt(np.mean(ay**2)),
                'RMS_Az': np.sqrt(np.mean(az**2)),
                
                # Varianza
                'Varianza_Ax': np.var(ax),
                'Varianza_Ay': np.var(ay),
                'Varianza_Az': np.var(az),
                
                # Energía Total
                'Energia_Total': np.sum(ax**2) + np.sum(ay**2) + np.sum(az**2),
                
                # Jerk promedio (derivada de la aceleración)
                'Jerk_Promedio_Ax': np.mean(np.abs(np.diff(ax))),
                'Jerk_Promedio_Ay': np.mean(np.abs(np.diff(ay))),
                'Jerk_Promedio_Az': np.mean(np.abs(np.diff(az))),
                
                # Entropía
                'Entropia_Ax': self.calculate_entropy(ax),
                'Entropia_Ay': self.calculate_entropy(ay),
                'Entropia_Az': self.calculate_entropy(az),
                
                # Máximos y Mínimos
                'Max_Ax': np.max(ax),
                'Min_Ax': np.min(ax),
                'Max_Ay': np.max(ay),
                'Min_Ay': np.min(ay),
                'Max_Az': np.max(az),
                'Min_Az': np.min(az),
            }
            
            # Agregar características de giroscopio si están disponibles
            if 'Gx' in window.columns:
                gx = window['Gx'].values
                gy = window['Gy'].values
                gz = window['Gz'].values
                
                features.update({
                    'Media_Gx': np.mean(gx),
                    'Media_Gy': np.mean(gy),
                    'Media_Gz': np.mean(gz),
                    'Varianza_Gy': np.var(gy),
                    'Energia_Rotacional': np.sum(gx**2) + np.sum(gy**2) + np.sum(gz**2),
                    'Max_Gx': np.max(gx),
                    'Min_Gx': np.min(gx),
                })
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def calculate_entropy(self, signal_data):
        """Calcula la entropía de una señal"""
        if len(signal_data) == 0:
            return 0
        # Discretizar la señal en bins
        hist, _ = np.histogram(signal_data, bins=20, density=True)
        # Normalizar y evitar log(0)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        return entropy(hist)
    
    def classify(self, features):
        """
        Clasifica en 'Saltando' o 'Estático' basado en características
        """
        predictions = []
        
        for idx, row in features.iterrows():
            # Calcular rangos de variación
            rango_ax = row['Max_Ax'] - row['Min_Ax']
            rango_ay = row['Max_Ay'] - row['Min_Ay']
            rango_az = row['Max_Az'] - row['Min_Az']
            
            # Criterio de clasificación simplificado
            # Si hay alta variación en aceleración -> Saltando
            umbral_variacion = 0.3  # Ajusta este valor según tus datos
            
            if (rango_ax > umbral_variacion or 
                rango_ay > umbral_variacion or 
                rango_az > umbral_variacion * 2):
                predictions.append('Saltando')
            else:
                predictions.append('Estático')
        
        return predictions

#-----------------------Modulo B ----------------------------------------------
class ModuleB:
    def __init__(self, window):
        self.window = window
        self.window.title("Módulo B - Adquisición en Vivo")
        self.window.geometry("1200x800")
        self.window.configure(bg="#ECF0F1")
        
        # Header
        header = tk.Frame(window, bg="#D348D8", height=80)
        header.pack(fill="x")
        title = tk.Label(header, text="MÓDULO B: Adquisición en Vivo",
                         font=("Arial", 20, "bold"), bg="#DA2DAE", fg="white")
        title.pack(pady=25)
        
        # Contenido simple para el módulo B
        content = tk.Frame(window, bg="#ECF0F1")
        content.pack(expand=True, fill="both", padx=50, pady=50)
        
        label = tk.Label(content, text="Módulo B - Adquisición en Vivo\n\n(Esta funcionalidad está en desarrollo)",
                        font=("Arial", 16), bg="#ECF0F1", fg="#2C3E50")
        label.pack(expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()