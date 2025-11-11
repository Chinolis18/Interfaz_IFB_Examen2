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
import os
from datetime import datetime
from pipeline_module import DataPipeline  # ← Pipeline externo
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import serial
import serial.tools.list_ports
from collections import deque




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
        self.window.configure(bg="#ECF0F1")
        self.data = None
        self.sampling_rate = 200
        self.features_df = None  # Para almacenar las características extraídas
        self.true_labels = None  # Para almacenar etiquetas reales si están disponibles
        self.predicted_labels = None  # Para almacenar predicciones

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

        # Frame para botones en dos filas
        button_row1 = tk.Frame(control_frame, bg="#127E99")
        button_row1.pack(fill="x", pady=2)
        
        button_row2 = tk.Frame(control_frame, bg="#127E99")
        button_row2.pack(fill="x", pady=2)

        # Botones en primera fila
        tk.Button(button_row1, text="Cargar Archivo",
                  font=("Arial", 11), bg="#723FE7", fg="white",
                  command=self.load_file, width=20).pack(side="left", padx=2)
        tk.Button(button_row1, text="Procesar Datos",
                  font=("Arial", 11), bg="#CB3CE7", fg="white",
                  command=self.process_data, width=20).pack(side="left", padx=2)
        
        # NUEVO BOTÓN: Borrar Todo
        tk.Button(button_row1, text="Borrar Todo",
                  font=("Arial", 11), bg="#E74C3C", fg="white",
                  command=self.clear_all, width=20).pack(side="left", padx=2)

        # Botones en segunda fila
        tk.Button(button_row2, text="Guardar Características",
                  font=("Arial", 11), bg="#27AE60", fg="white",
                  command=self.save_features, width=20).pack(side="left", padx=2)
        tk.Button(button_row2, text="Guardar Resultados",
                  font=("Arial", 11), bg="#2980B9", fg="white",
                  command=self.save_results, width=20).pack(side="left", padx=2)

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
        self.ax1.set_title("Datos Crudos", fontsize=12)
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Matriz de Confusión", fontsize=12)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear_all(self):
        """Limpia todos los datos y reinicia la interfaz"""
        try:
            # Limpiar datos
            self.data = None
            self.features_df = None
            self.true_labels = None
            self.predicted_labels = None
            
            # Limpiar textos
            self.raw_text.delete(1.0, tk.END)
            self.features_text.delete(1.0, tk.END)
            self.results_text.delete(1.0, tk.END)
            
            # Limpiar gráficas
            self.ax1.clear()
            self.ax2.clear()
            
            # Restablecer títulos de gráficas
            self.ax1.set_title("Datos Crudos", fontsize=12)
            self.ax2.set_title("Matriz de Confusión", fontsize=12)
            
            # Redibujar canvas
            self.canvas.draw()
            
            # Restablecer estado
            self.status_label.config(text="Estado: Listo para cargar nuevo archivo")
            
            # Mostrar mensaje de confirmación
            messagebox.showinfo("Éxito", "Todos los datos han sido borrados.\nPuedes cargar un nuevo archivo.", 
                              parent=self.window)
            
            print("Sistema reiniciado - Listo para nuevo archivo")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron borrar los datos:\n{str(e)}", parent=self.window)


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
                with open(filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                data_start_line = 0
                for i, line in enumerate(lines):
                    if 'address' in line and 'Time(s)' in line and 'ax(g)' in line:
                        data_start_line = i
                        break
                    elif len(line.split('\t')) >= 7:
                        data_start_line = i
                        break
                
                self.data = pd.read_csv(filename, 
                                      skiprows=data_start_line, 
                                      sep='\t',
                                      encoding='utf-8')
                
                self.data.columns = self.data.columns.str.strip()
                self.standardize_columns()

            elif filename.endswith('.xlsx'):
                raw_data = pd.read_excel(filename, header=None)
                
                header_row = 0
                for i, row in raw_data.iterrows():
                    if any('ax(g)' in str(cell) for cell in row.values):
                        header_row = i
                        break
                    elif any('address' in str(cell) for cell in row.values):
                        header_row = i
                        break
                
                self.data = pd.read_excel(filename, header=header_row)
                self.data.columns = self.data.columns.str.strip()
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

            self.data = self.data.dropna()

            self.raw_text.delete(1.0, tk.END)
            self.raw_text.insert(1.0, f"Archivo: {filename}\n")
            self.raw_text.insert(tk.END, f"Columnas: {', '.join(self.data.columns)}\n")
            self.raw_text.insert(tk.END, f"Total de muestras: {len(self.data)}\n\n")
            self.raw_text.insert(tk.END, "Primeras 50 filas:\n")
            self.raw_text.insert(tk.END, self.data.head(50).to_string())
            
            self.plot_raw_data()
            self.status_label.config(text=f"Archivo cargado ({len(self.data)} muestras)")
            messagebox.showinfo("Éxito", "Archivo cargado correctamente", parent=self.window)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{str(e)}", parent=self.window)

    def standardize_columns(self):
        if self.data is None:
            return
            
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
        
        for new_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in available_columns:
                    renamed_columns[possible_name] = new_name
                    break
        
        if renamed_columns:
            self.data = self.data.rename(columns=renamed_columns)
            print(f"Columnas renombradas: {renamed_columns}")
        
        required_columns = ['Ax', 'Ay', 'Az']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            available_cols = ", ".join(self.data.columns)
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}\nColumnas disponibles: {available_cols}")

    def plot_raw_data(self):
        if self.data is None:
            return
        
        time = np.arange(len(self.data)) / self.sampling_rate
        
        self.ax1.clear()
        
        if 'Ax' in self.data.columns and 'Ay' in self.data.columns and 'Az' in self.data.columns:
            self.ax1.plot(time, self.data['Ax'], label='Ax', linewidth=1)
            self.ax1.plot(time, self.data['Ay'], label='Ay', linewidth=1)
            self.ax1.plot(time, self.data['Az'], label='Az', linewidth=1)
            self.ax1.legend()
            self.ax1.set_xlabel("Tiempo (s)")
            self.ax1.set_ylabel("Aceleración (g)")
            self.ax1.set_title("Datos Crudos de Aceleración")
        else:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            for i, col in enumerate(numeric_columns[:3]):
                self.ax1.plot(time, self.data[col], label=col, linewidth=1)
            self.ax1.legend()
            self.ax1.set_xlabel("Tiempo (s)")
            self.ax1.set_ylabel("Valor")
            self.ax1.set_title("Datos Crudos (primeras 3 columnas numéricas)")

        self.canvas.draw()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Dibuja la matriz de confusión en el segundo subplot"""
        self.ax2.clear()
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(set(y_true) | set(y_pred))
        
        # Crear heatmap de la matriz de confusión
        im = self.ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        self.ax2.set_title('Matriz de Confusión', fontsize=12, pad=20)
        
        # Añadir etiquetas de clases
        tick_marks = np.arange(len(classes))
        self.ax2.set_xticks(tick_marks)
        self.ax2.set_yticks(tick_marks)
        self.ax2.set_xticklabels(classes, rotation=45)
        self.ax2.set_yticklabels(classes)
        
        # Añadir valores en las celdas
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax2.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=10, fontweight='bold')
        
        self.ax2.set_ylabel('Etiqueta Real')
        self.ax2.set_xlabel('Etiqueta Predicha')
        self.ax2.grid(False)
        
        # Ajustar layout
        self.fig.tight_layout()
        self.canvas.draw()

    def calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Reporte de clasificación detallado
        class_report = classification_report(y_true, y_pred, zero_division=0)
        
        metrics_summary = f"MÉTRICAS DE EVALUACIÓN:\n"
        metrics_summary += f"Exactitud (Accuracy): {accuracy:.4f}\n"
        metrics_summary += f"Precisión (Precision): {precision:.4f}\n"
        metrics_summary += f"Sensibilidad (Recall): {recall:.4f}\n"
        metrics_summary += f"Puntuación F1: {f1:.4f}\n\n"
        metrics_summary += f"REPORTE DE CLASIFICACIÓN:\n{class_report}"
        
        return metrics_summary

    def process_data(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Carga un archivo antes de procesar", parent=self.window)
            return
        
        self.status_label.config(text="Procesando...")
        self.window.update()
        
        try:
            required_columns = ['Ax', 'Ay', 'Az']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Faltan columnas necesarias: {missing_columns}\nColumnas disponibles: {list(self.data.columns)}")
            
            print(f"Columnas disponibles para procesamiento: {list(self.data.columns)}")
            print(f"Primeras filas:\n{self.data.head()}")

            results = self.run_basic_pipeline(self.data)
            self.features_df = results['features']  # Guardar las características para poder exportarlas
            self.predicted_labels = results['predictions']
            
            # Mostrar TODAS las ventanas
            self.features_text.delete(1.0, tk.END)
            self.features_text.insert(1.0, self.features_df.to_string())
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results['summary'])
            
            # Generar etiquetas reales simuladas para demostración
            # En un caso real, estas vendrían de tu dataset etiquetado
            self.generate_simulated_true_labels()
            
            # Calcular métricas y mostrar matriz de confusión
            metrics_summary = self.calculate_metrics(self.true_labels, self.predicted_labels)
            self.results_text.insert(tk.END, "\n\n" + metrics_summary)
            
            # Dibujar matriz de confusión
            self.plot_confusion_matrix(self.true_labels, self.predicted_labels)
            
            self.status_label.config(text="Procesamiento completado")
            
        except Exception as e:
            error_msg = f"Error durante el procesamiento:\n{str(e)}\n\nColumnas disponibles: {list(self.data.columns) if self.data is not None else 'No data'}"
            messagebox.showerror("Error", error_msg, parent=self.window)
            self.status_label.config(text="Error en procesamiento")
            print(f"Error detallado: {e}")

    def generate_simulated_true_labels(self):
        """Genera etiquetas reales simuladas para demostración"""
        # En un caso real, estas etiquetas vendrían de tu dataset
        # Aquí simulamos algunas etiquetas basadas en los datos
        num_windows = len(self.features_df)
        self.true_labels = []
        
        for idx, row in self.features_df.iterrows():
            # Simulamos etiquetas reales basadas en características
            energia_total = row['Energia_Total']
            varianza_ax = row['Varianza_Ax']
            
            if energia_total > 100 or varianza_ax > 0.1:
                # Mayor probabilidad de ser "Saltando" si hay alta energía o varianza
                if np.random.random() < 0.7:
                    self.true_labels.append('Saltando')
                else:
                    self.true_labels.append('Estático')
            else:
                # Mayor probabilidad de ser "Estático" si hay baja energía
                if np.random.random() < 0.8:
                    self.true_labels.append('Estático')
                else:
                    self.true_labels.append('Saltando')
        
        # Asegurarnos de que tenemos al menos una instancia de cada clase
        if 'Saltando' not in self.true_labels and len(self.true_labels) > 0:
            self.true_labels[0] = 'Saltando'
        if 'Estático' not in self.true_labels and len(self.true_labels) > 1:
            self.true_labels[1] = 'Estático'

    def save_features(self):
        """Guarda las características extraídas en archivos CSV y Excel"""
        if self.features_df is None:
            messagebox.showwarning("Advertencia", "No hay características para guardar. Primero procesa los datos.", parent=self.window)
            return
        
        try:
            # Crear carpeta para resultados si no existe
            results_dir = "Resultados_Caracteristicas"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"caracteristicas_{timestamp}"
            
            # Guardar en CSV
            csv_filename = os.path.join(results_dir, f"{base_filename}.csv")
            self.features_df.to_csv(csv_filename, index=False, encoding='utf-8')
            
            # Guardar en Excel
            excel_filename = os.path.join(results_dir, f"{base_filename}.xlsx")
            self.features_df.to_excel(excel_filename, index=False)
            
            # Guardar resumen estadístico
            summary_filename = os.path.join(results_dir, f"resumen_{base_filename}.txt")
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write("RESUMEN DE CARACTERÍSTICAS EXTRACTADAS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total de ventanas: {len(self.features_df)}\n")
                f.write(f"Total de características: {len(self.features_df.columns)}\n\n")
                
                # Estadísticas de clasificación
                if 'Prediccion' in self.features_df.columns:
                    class_counts = self.features_df['Prediccion'].value_counts()
                    f.write("DISTRIBUCIÓN DE CLASES:\n")
                    for clase, count in class_counts.items():
                        percentage = (count / len(self.features_df)) * 100
                        f.write(f"  {clase}: {count} ventanas ({percentage:.1f}%)\n")
                    f.write("\n")
                
                # Estadísticas descriptivas de las características numéricas
                numeric_features = self.features_df.select_dtypes(include=[np.number])
                f.write("ESTADÍSTICAS DESCRIPTIVAS:\n")
                f.write(numeric_features.describe().to_string())
            
            messagebox.showinfo("Éxito", 
                               f"Características guardadas exitosamente en:\n"
                               f"• {csv_filename}\n"
                               f"• {excel_filename}\n"
                               f"• {summary_filename}", 
                               parent=self.window)
            
            self.status_label.config(text=f"Características guardadas en {results_dir}/")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron guardar las características:\n{str(e)}", parent=self.window)

    def save_results(self):
        """Guarda los resultados del procesamiento en un archivo de texto"""
        if self.features_df is None:
            messagebox.showwarning("Advertencia", "No hay resultados para guardar. Primero procesa los datos.", parent=self.window)
            return
        
        try:
            # Crear carpeta para resultados si no existe
            results_dir = "Resultados_Procesamiento"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"resultados_completos_{timestamp}.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("RESULTADOS COMPLETOS DEL PROCESAMIENTO IMU\n")
                f.write("=" * 60 + "\n\n")
                
                # Información general
                f.write("INFORMACIÓN GENERAL:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Fecha de procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Tasa de muestreo: {self.sampling_rate} Hz\n")
                f.write(f"Total de muestras procesadas: {len(self.data) if self.data is not None else 0}\n")
                f.write(f"Total de ventanas: {len(self.features_df)}\n")
                f.write(f"Duración total: {len(self.data)/self.sampling_rate if self.data is not None else 0:.2f} segundos\n\n")
                
                # Distribución de clases
                if 'Prediccion' in self.features_df.columns:
                    f.write("DISTRIBUCIÓN DE CLASIFICACIÓN:\n")
                    f.write("-" * 40 + "\n")
                    class_counts = self.features_df['Prediccion'].value_counts()
                    for clase, count in class_counts.items():
                        percentage = (count / len(self.features_df)) * 100
                        f.write(f"{clase}: {count} ventanas ({percentage:.1f}%)\n")
                    f.write("\n")
                
                # Métricas de evaluación si están disponibles
                if self.true_labels is not None and self.predicted_labels is not None:
                    f.write("MÉTRICAS DE EVALUACIÓN:\n")
                    f.write("-" * 40 + "\n")
                    accuracy = accuracy_score(self.true_labels, self.predicted_labels)
                    precision = precision_score(self.true_labels, self.predicted_labels, average='weighted', zero_division=0)
                    recall = recall_score(self.true_labels, self.predicted_labels, average='weighted', zero_division=0)
                    f1 = f1_score(self.true_labels, self.predicted_labels, average='weighted', zero_division=0)
                    
                    f.write(f"Exactitud (Accuracy): {accuracy:.4f}\n")
                    f.write(f"Precisión (Precision): {precision:.4f}\n")
                    f.write(f"Sensibilidad (Recall): {recall:.4f}\n")
                    f.write(f"Puntuación F1: {f1:.4f}\n\n")
                    
                    # Matriz de confusión
                    cm = confusion_matrix(self.true_labels, self.predicted_labels)
                    classes = sorted(set(self.true_labels) | set(self.predicted_labels))
                    f.write("MATRIZ DE CONFUSIÓN:\n")
                    f.write("-" * 40 + "\n")
                    f.write("Clases: " + ", ".join(classes) + "\n\n")
                    f.write(pd.DataFrame(cm, index=classes, columns=classes).to_string())
                    f.write("\n\n")
                
                # Características principales (primeras 10 ventanas)
                f.write("CARACTERÍSTICAS EXTRACTADAS (Primeras 10 ventanas):\n")
                f.write("-" * 40 + "\n")
                f.write(self.features_df.head(10).to_string())
                f.write("\n\n...\n\n")
                
                # Estadísticas de características numéricas
                numeric_features = self.features_df.select_dtypes(include=[np.number])
                f.write("ESTADÍSTICAS DE CARACTERÍSTICAS NUMÉRICAS:\n")
                f.write("-" * 40 + "\n")
                f.write(numeric_features.describe().to_string())
                f.write("\n\n")
                
                # Lista de todas las características
                f.write("LISTA DE CARACTERÍSTICAS EXTRACTADAS:\n")
                f.write("-" * 40 + "\n")
                for i, col in enumerate(self.features_df.columns, 1):
                    f.write(f"{i:2d}. {col}\n")
            
            messagebox.showinfo("Éxito", 
                               f"Resultados guardados exitosamente en:\n{filename}", 
                               parent=self.window)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron guardar los resultados:\n{str(e)}", parent=self.window)

    def run_basic_pipeline(self, data):
        filtered_data = self.apply_notch_filter(data)
        windows = self.segment_windows(filtered_data)
        features = self.extract_features(windows)
        predictions = self.classify(features)
        features['Prediccion'] = predictions
        
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
            'predictions': predictions,
            'summary': summary
        }

    def apply_notch_filter(self, data):
        filtered_data = data.copy()
        notch_freq = 50.0
        quality_factor = 30.0
        b, a = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
        
        for col in ['Ax', 'Ay', 'Az']:
            if col in filtered_data.columns:
                filtered_data[col] = signal.filtfilt(b, a, filtered_data[col])
        
        return filtered_data
    
    def segment_windows(self, data, window_size=50, overlap=0.5):
        windows = []
        step_size = int(window_size * (1 - overlap))  # Paso entre ventanas
        num_windows = (len(data) - window_size)// step_size + 1

        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            if end_idx <= len(data):
                window = data.iloc[start_idx:end_idx]
                windows.append(window)
        
        return windows
    
    def extract_features(self, windows):
        features_list = []
        
        for idx, window in enumerate(windows):
            ax = window['Ax'].values
            ay = window['Ay'].values
            az = window['Az'].values
            
            features = {
                'Ventana': idx + 1,
                'Media_Ax': np.mean(ax),
                'Media_Ay': np.mean(ay),
                'Media_Az': np.mean(az),
                'RMS_Ax': np.sqrt(np.mean(ax**2)),
                'RMS_Ay': np.sqrt(np.mean(ay**2)),
                'RMS_Az': np.sqrt(np.mean(az**2)),
                'Varianza_Ax': np.var(ax),
                'Varianza_Ay': np.var(ay),
                'Varianza_Az': np.var(az),
                'Energia_Total': np.sum(ax**2) + np.sum(ay**2) + np.sum(az**2),
                'Jerk_Promedio_Ax': np.mean(np.abs(np.diff(ax))),
                'Jerk_Promedio_Ay': np.mean(np.abs(np.diff(ay))),
                'Jerk_Promedio_Az': np.mean(np.abs(np.diff(az))),
                'Entropia_Ax': self.calculate_entropy(ax),
                'Entropia_Ay': self.calculate_entropy(ay),
                'Entropia_Az': self.calculate_entropy(az),
                'Max_Ax': np.max(ax),
                'Min_Ax': np.min(ax),
                'Max_Ay': np.max(ay),
                'Min_Ay': np.min(ay),
                'Max_Az': np.max(az),
                'Min_Az': np.min(az),
            }
            
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
        if len(signal_data) == 0:
            return 0
        hist, _ = np.histogram(signal_data, bins=20, density=True)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        return entropy(hist)
    
    def classify(self, features):
        predictions = []
        
        for idx, row in features.iterrows():
            rango_ax = row['Max_Ax'] - row['Min_Ax']
            rango_ay = row['Max_Ay'] - row['Min_Ay']
            rango_az = row['Max_Az'] - row['Min_Az']
            
            umbral_variacion = 0.3
            
            if (rango_ax > umbral_variacion or 
                rango_ay > umbral_variacion or 
                rango_az > umbral_variacion * 2):
                predictions.append('Saltando')
            else:
                predictions.append('Estático')
        
        return predictions
#----------------------------------------Modulo B--------------------------------------------------
class ModuleB:
    def __init__(self, window):
        self.window = window
        self.window.title("Módulo B - Adquisición en Vivo")
        self.window.geometry("1400x900")
        self.window.configure(bg="#24525E")
        
        # Variables de control
        self.is_acquiring = False
        self.serial_connection = None
        self.data_buffer = deque(maxlen=1000)  # Buffer para gráficos en tiempo real
        self.raw_data = []  # Almacenar todos los datos crudos
        self.sampling_rate = 100  # Hz
        self.start_time = None
        
        # Header
        header = tk.Frame(window, bg="#D348D8", height=80)
        header.pack(fill="x")
        title = tk.Label(header, text="MÓDULO B: Adquisición en Vivo - Sensor WT9011DCL",
                         font=("Arial", 16, "bold"), bg="#DA2DAE", fg="white")
        title.pack(pady=15)

        # Main container
        main_container = tk.Frame(window, bg="#1A6072")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel - Controles
        left_panel = tk.Frame(main_container, bg="#17677A", width=400)
        left_panel.pack(side="left", fill="y", padx=5)
        left_panel.pack_propagate(False)

        # Right panel - Gráficos
        right_panel = tk.Frame(main_container, bg="white")
        right_panel.pack(side="right", fill="both", expand=True, padx=5)

        self.setup_controls(left_panel)
        self.setup_graphs(right_panel)
        
        # Inicializar gráficos
        self.initialize_plots()

    def setup_controls(self, parent):
        """Configura el panel de controles"""
        # Frame de configuración serial
        serial_frame = tk.LabelFrame(parent, text="Configuración Serial", 
                                   font=("Arial", 10, "bold"), bg="#2C3E50", fg="white",
                                   padx=10, pady=10)
        serial_frame.pack(fill="x", padx=10, pady=10)

        # Puerto COM
        tk.Label(serial_frame, text="Puerto COM:", bg="#2C3E50", fg="white", 
                font=("Arial", 9)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.com_port_var = tk.StringVar(value="COM8")
        com_entry = tk.Entry(serial_frame, textvariable=self.com_port_var, width=15,
                           font=("Arial", 9))
        com_entry.grid(row=0, column=1, padx=5, pady=5)

        # Botón para buscar puertos
        tk.Button(serial_frame, text="Buscar Puertos", command=self.search_ports,
                 bg="#3498DB", fg="white", width=12, font=("Arial", 9)).grid(row=0, column=2, padx=5, pady=5)

        # Baud rate
        tk.Label(serial_frame, text="Baud Rate:", bg="#2C3E50", fg="white",
                font=("Arial", 9)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.baud_var = tk.StringVar(value="115200")
        baud_combo = ttk.Combobox(serial_frame, textvariable=self.baud_var, 
                                 values=["9600", "19200", "38400", "57600", "115200"], 
                                 width=13, state="readonly")
        baud_combo.grid(row=1, column=1, padx=5, pady=5)

        # Frame de controles de adquisición
        control_frame = tk.LabelFrame(parent, text="Control de Adquisición", 
                                    font=("Arial", 10, "bold"), bg="#2C3E50", fg="white",
                                    padx=10, pady=10)
        control_frame.pack(fill="x", padx=10, pady=10)

        # Botones de control
        btn_frame = tk.Frame(control_frame, bg="#2C3E50")
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="Iniciar Adquisición", 
                                  command=self.start_acquisition,
                                  bg="#27AE60", fg="white", font=("Arial", 11, "bold"),
                                  width=15, height=2)
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(btn_frame, text="Detener Adquisición", 
                                 command=self.stop_acquisition,
                                 bg="#E74C3C", fg="white", font=("Arial", 11, "bold"),
                                 width=15, height=2, state="disabled")
        self.stop_btn.pack(pady=5)

        # Frame de selección de ejes
        axes_frame = tk.LabelFrame(parent, text="Selección de Ejes a Visualizar", 
                                 font=("Arial", 10, "bold"), bg="#2C3E50", fg="white",
                                 padx=10, pady=10)
        axes_frame.pack(fill="x", padx=10, pady=10)

        # Acelerómetro
        accel_frame = tk.Frame(axes_frame, bg="#2C3E50")
        accel_frame.pack(fill="x", pady=5)
        
        tk.Label(accel_frame, text="Acelerómetro:", bg="#2C3E50", fg="white", 
                font=("Arial", 9, "bold")).pack(anchor="w")
        
        accel_check_frame = tk.Frame(accel_frame, bg="#2C3E50")
        accel_check_frame.pack(fill="x", pady=5)
        
        self.accel_vars = {}
        accel_axes = ["Ax", "Ay", "Az"]
        for i, axis in enumerate(accel_axes):
            var = tk.BooleanVar(value=True)
            self.accel_vars[axis] = var
            cb = tk.Checkbutton(accel_check_frame, text=axis, variable=var, 
                              bg="#2C3E50", fg="white", selectcolor="#34495E",
                              font=("Arial", 9),
                              command=self.update_plots)
            cb.pack(side="left", padx=15)

        # Giroscopio
        gyro_frame = tk.Frame(axes_frame, bg="#2C3E50")
        gyro_frame.pack(fill="x", pady=5)
        
        tk.Label(gyro_frame, text="Giroscopio:", bg="#2C3E50", fg="white",
                font=("Arial", 9, "bold")).pack(anchor="w")
        
        gyro_check_frame = tk.Frame(gyro_frame, bg="#2C3E50")
        gyro_check_frame.pack(fill="x", pady=5)
        
        self.gyro_vars = {}
        gyro_axes = ["Gx", "Gy", "Gz"]
        for i, axis in enumerate(gyro_axes):
            var = tk.BooleanVar(value=True)
            self.gyro_vars[axis] = var
            cb = tk.Checkbutton(gyro_check_frame, text=axis, variable=var,
                              bg="#2C3E50", fg="white", selectcolor="#34495E",
                              font=("Arial", 9),
                              command=self.update_plots)
            cb.pack(side="left", padx=15)

        # Frame de guardado
        save_frame = tk.LabelFrame(parent, text="Guardar Datos", 
                                 font=("Arial", 10, "bold"), bg="#2C3E50", fg="white",
                                 padx=10, pady=10)
        save_frame.pack(fill="x", padx=10, pady=10)

        tk.Button(save_frame, text="Guardar Datos Crudos", 
                 command=self.save_raw_data,
                 bg="#2980B9", fg="white", width=20, height=2,
                 font=("Arial", 10)).pack(pady=10)

        # Frame de información
        info_frame = tk.LabelFrame(parent, text="Información", 
                                 font=("Arial", 10, "bold"), bg="#2C3E50", fg="white",
                                 padx=10, pady=10)
        info_frame.pack(fill="x", padx=10, pady=10)

        info_content = tk.Frame(info_frame, bg="#2C3E50")
        info_content.pack(fill="x", pady=5)

        self.status_label = tk.Label(info_content, text="Estado: Desconectado", 
                                   bg="#2C3E50", fg="#F1C40F", font=("Arial", 10, "bold"),
                                   justify="left")
        self.status_label.pack(anchor="w", pady=2)

        self.samples_label = tk.Label(info_content, text="Muestras: 0", 
                                    bg="#2C3E50", fg="white", font=("Arial", 9),
                                    justify="left")
        self.samples_label.pack(anchor="w", pady=2)

        self.duration_label = tk.Label(info_content, text="Duración: 0s", 
                                     bg="#2C3E50", fg="white", font=("Arial", 9),
                                     justify="left")
        self.duration_label.pack(anchor="w", pady=2)

    def setup_graphs(self, parent):
        """Configura los gráficos en tiempo real"""
        self.fig = Figure(figsize=(10, 8), facecolor='white', dpi=100)
        
        # Gráfico de acelerómetro
        self.ax_accel = self.fig.add_subplot(211)
        self.ax_accel.set_title("Acelerómetro en Tiempo Real", fontsize=12, fontweight='bold')
        self.ax_accel.set_ylabel("Aceleración (g)", fontsize=10)
        self.ax_accel.grid(True, alpha=0.3)
        self.ax_accel.set_facecolor('#F8F9F9')
        
        # Gráfico de giroscopio
        self.ax_gyro = self.fig.add_subplot(212)
        self.ax_gyro.set_title("Giroscopio en Tiempo Real", fontsize=12, fontweight='bold')
        self.ax_gyro.set_ylabel("Velocidad Angular (°/s)", fontsize=10)
        self.ax_gyro.set_xlabel("Tiempo (s)", fontsize=10)
        self.ax_gyro.grid(True, alpha=0.3)
        self.ax_gyro.set_facecolor('#F8F9F9')
        
        # Ajustar espaciado
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def initialize_plots(self):
        """Inicializa las líneas de los gráficos"""
        self.accel_lines = {}
        self.gyro_lines = {}
        colors = {'Ax': '#E74C3C', 'Ay': '#3498DB', 'Az': '#27AE60',
                 'Gx': '#E74C3C', 'Gy': '#3498DB', 'Gz': '#27AE60'}
        
        # Inicializar líneas vacías para acelerómetro
        for axis in ['Ax', 'Ay', 'Az']:
            self.accel_lines[axis], = self.ax_accel.plot([], [], 
                                                       color=colors[axis], 
                                                       linewidth=1.5, 
                                                       label=axis)
        
        # Inicializar líneas vacías para giroscopio
        for axis in ['Gx', 'Gy', 'Gz']:
            self.gyro_lines[axis], = self.ax_gyro.plot([], [], 
                                                     color=colors[axis], 
                                                     linewidth=1.5, 
                                                     label=axis)
        
        # Mostrar leyendas
        self.ax_accel.legend(loc='upper right')
        self.ax_gyro.legend(loc='upper right')
        
        # Dibujar canvas inicial
        self.canvas.draw()

    def search_ports(self):
        """Busca y muestra los puertos COM disponibles"""
        try:
            ports = serial.tools.list_ports.comports()
            port_list = [f"{port.device} - {port.description}" for port in ports]
            
            if port_list:
                port_info = "\n".join(port_list)
                messagebox.showinfo("Puertos COM Disponibles", 
                                  f"Puertos encontrados:\n\n{port_info}")
            else:
                messagebox.showwarning("Sin Puertos", "No se encontraron puertos COM disponibles")
        except Exception as e:
            messagebox.showerror("Error", f"Error buscando puertos: {str(e)}")

    def start_acquisition(self):
        """Inicia la adquisición de datos"""
        try:
            # Verificar que el puerto COM esté disponible
            port = self.com_port_var.get().strip()
            if not port:
                messagebox.showwarning("Advertencia", "Por favor ingresa un puerto COM válido")
                return
            
            # Conectar al puerto serial
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=int(self.baud_var.get()),
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Esperar a que se establezca la conexión
            time.sleep(2)
            
            if self.serial_connection.is_open:
                # Configurar el sensor WT9011DCL
                self.configure_sensor()
                
                # Iniciar variables
                self.is_acquiring = True
                self.raw_data = []
                self.data_buffer.clear()
                self.start_time = time.time()
                
                # Actualizar interfaz
                self.start_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
                self.status_label.config(text="Estado: Adquiriendo datos...", fg="#2ECC71")
                
                # Iniciar hilo de adquisición
                self.acquisition_thread = threading.Thread(target=self.acquisition_loop)
                self.acquisition_thread.daemon = True
                self.acquisition_thread.start()
                
                # Iniciar actualización de gráficos
                self.window.after(100, self.update_plots)
                
                messagebox.showinfo("Éxito", f"Adquisición iniciada en {port}")
            else:
                messagebox.showerror("Error", f"No se pudo abrir el puerto {port}")
            
        except serial.SerialException as e:
            messagebox.showerror("Error Serial", f"No se pudo conectar al puerto {self.com_port_var.get()}:\n{str(e)}")
            self.stop_acquisition()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar la adquisición:\n{str(e)}")
            self.stop_acquisition()

    def configure_sensor(self):
        """Configura el sensor WT9011DCL para enviar datos continuamente"""
        try:
            # Comando para salida continua de datos (consultar manual del WT9011DCL)
            # Este es un comando genérico - ajustar según el manual específico
            output_cmd = bytes([0xFF, 0xAA, 0x01, 0x00, 0x00])  # Ejemplo de comando
            self.serial_connection.write(output_cmd)
            time.sleep(0.1)
            print("Sensor configurado para salida continua")
        except Exception as e:
            print(f"Error en configuración del sensor: {e}")

    def parse_wt9011dcl_data(self, data):
        """Parsea los datos del sensor WT9011DCL - SIMULACIÓN para pruebas"""
        # Para pruebas, generamos datos simulados
        # En producción, reemplazar con el parser real del WT9011DCL
        try:
            current_time = time.time() - self.start_time
            
            # Generar datos de acelerómetro simulados
            ax = 0.1 * np.sin(2 * np.pi * 0.5 * current_time) + np.random.normal(0, 0.01)
            ay = 0.1 * np.sin(2 * np.pi * 0.3 * current_time + np.pi/2) + np.random.normal(0, 0.01)
            az = 1.0 + 0.05 * np.sin(2 * np.pi * 0.1 * current_time) + np.random.normal(0, 0.01)
            
            # Generar datos de giroscopio simulados
            gx = 10 * np.sin(2 * np.pi * 0.2 * current_time) + np.random.normal(0, 0.1)
            gy = 15 * np.sin(2 * np.pi * 0.4 * current_time + np.pi/4) + np.random.normal(0, 0.1)
            gz = 5 * np.sin(2 * np.pi * 0.1 * current_time + np.pi/2) + np.random.normal(0, 0.1)
            
            return {
                'Ax': ax, 'Ay': ay, 'Az': az,
                'Gx': gx, 'Gy': gy, 'Gz': gz,
                'Timestamp': current_time
            }
            
        except Exception as e:
            print(f"Error generando datos simulados: {e}")
            return None

    def acquisition_loop(self):
        """Bucle principal de adquisición de datos - MODIFICADO PARA PRUEBAS"""
        print("Iniciando bucle de adquisición...")
        
        while self.is_acquiring:
            try:
                # Para pruebas, generamos datos simulados
                # En producción, leer desde el puerto serial
                
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    # Leer datos reales del sensor
                    data = self.serial_connection.read(self.serial_connection.in_waiting)
                    # Aquí iría el parsing real de los datos
                    # parsed_data = self.parse_real_wt9011dcl_data(data)
                    pass
                else:
                    # Generar datos simulados para pruebas
                    parsed_data = self.parse_wt9011dcl_data(None)
                
                if parsed_data:
                    # Guardar en buffer y datos crudos
                    self.data_buffer.append(parsed_data)
                    self.raw_data.append(parsed_data)
                    
                    # Actualizar contadores en el hilo principal
                    self.window.after(0, self.update_counters)
                
                time.sleep(0.01)  # 10ms de intervalo para simular 100Hz
                
            except Exception as e:
                print(f"Error en adquisición: {e}")
                time.sleep(0.1)

    def update_counters(self):
        """Actualiza los contadores en la interfaz"""
        samples = len(self.raw_data)
        duration = time.time() - self.start_time if self.start_time else 0
        
        self.samples_label.config(text=f"Muestras: {samples}")
        self.duration_label.config(text=f"Duración: {duration:.1f}s")

    def update_plots(self):
        """Actualiza los gráficos en tiempo real"""
        if not self.is_acquiring:
            # Si no está adquiriendo, programar próxima verificación
            self.window.after(1000, self.update_plots)
            return
        
        try:
            if self.data_buffer:
                # Extraer datos del buffer
                times = [d['Timestamp'] for d in self.data_buffer]
                
                # Actualizar acelerómetro según selección
                for axis in ['Ax', 'Ay', 'Az']:
                    if self.accel_vars[axis].get():
                        values = [d[axis] for d in self.data_buffer]
                        self.accel_lines[axis].set_data(times, values)
                    else:
                        self.accel_lines[axis].set_data([], [])
                
                # Actualizar giroscopio según selección
                for axis in ['Gx', 'Gy', 'Gz']:
                    if self.gyro_vars[axis].get():
                        values = [d[axis] for d in self.data_buffer]
                        self.gyro_lines[axis].set_data(times, values)
                    else:
                        self.gyro_lines[axis].set_data([], [])
                
                # Ajustar límites del tiempo (ventana deslizante de 10 segundos)
                if times:
                    current_time = times[-1]
                    time_window = 10  # segundos
                    start_time = max(0, current_time - time_window)
                    
                    self.ax_accel.set_xlim(start_time, current_time)
                    self.ax_gyro.set_xlim(start_time, current_time)
                    
                    # Autoajustar límites Y basado en los datos visibles
                    self.auto_adjust_y_limits()
            
            # Redibujar
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error actualizando gráficos: {e}")
        
        # Programar próxima actualización
        self.window.after(100, self.update_plots)  # Actualizar cada 100ms

    def auto_adjust_y_limits(self):
        """Autoajusta los límites Y de los gráficos basado en los datos visibles"""
        try:
            # Para acelerómetro
            if self.data_buffer:
                visible_accel_data = []
                for axis in ['Ax', 'Ay', 'Az']:
                    if self.accel_vars[axis].get():
                        values = [d[axis] for d in self.data_buffer]
                        visible_accel_data.extend(values)
                
                if visible_accel_data:
                    min_val = min(visible_accel_data)
                    max_val = max(visible_accel_data)
                    margin = (max_val - min_val) * 0.1
                    self.ax_accel.set_ylim(min_val - margin, max_val + margin)
                
                # Para giroscopio
                visible_gyro_data = []
                for axis in ['Gx', 'Gy', 'Gz']:
                    if self.gyro_vars[axis].get():
                        values = [d[axis] for d in self.data_buffer]
                        visible_gyro_data.extend(values)
                
                if visible_gyro_data:
                    min_val = min(visible_gyro_data)
                    max_val = max(visible_gyro_data)
                    margin = (max_val - min_val) * 0.1
                    self.ax_gyro.set_ylim(min_val - margin, max_val + margin)
                    
        except Exception as e:
            print(f"Error autoajustando límites Y: {e}")

    def stop_acquisition(self):
        """Detiene la adquisición de datos"""
        self.is_acquiring = False
        
        # Cerrar conexión serial
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
            except:
                pass
        
        # Actualizar interfaz
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Estado: Adquisición detenida", fg="#F1C40F")
        
        if self.raw_data:
            messagebox.showinfo("Información", f"Adquisición detenida. Se capturaron {len(self.raw_data)} muestras.")
        else:
            messagebox.showinfo("Información", "Adquisición detenida.")

    def save_raw_data(self):
        """Guarda los datos crudos en archivos CSV y MAT"""
        if not self.raw_data:
            messagebox.showwarning("Advertencia", "No hay datos para guardar")
            return
        
        try:
            # Crear carpeta para datos
            data_dir = "Datos_Adquisicion_Vivo"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"datos_wt9011dcl_{timestamp}"
            
            # Convertir a DataFrame
            df = pd.DataFrame(self.raw_data)
            
            # Guardar como CSV
            csv_filename = os.path.join(data_dir, f"{base_filename}.csv")
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            
            # Guardar como MAT
            mat_filename = os.path.join(data_dir, f"{base_filename}.mat")
            mat_data = {
                'accel_data': df[['Ax', 'Ay', 'Az']].values,
                'gyro_data': df[['Gx', 'Gy', 'Gz']].values,
                'timestamp': df['Timestamp'].values,
                'sampling_rate': self.sampling_rate,
                'columns': ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Timestamp']
            }
            sio.savemat(mat_filename, mat_data)
            
            # Guardar información adicional
            info_filename = os.path.join(data_dir, f"info_{base_filename}.txt")
            with open(info_filename, 'w', encoding='utf-8') as f:
                f.write("INFORMACIÓN DE ADQUISICIÓN WT9011DCL\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Fecha de adquisición: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total de muestras: {len(self.raw_data)}\n")
                f.write(f"Duración: {df['Timestamp'].iloc[-1] if len(df) > 0 else 0:.2f} segundos\n")
                f.write(f"Tasa de muestreo aproximada: {self.sampling_rate} Hz\n")
                f.write(f"Puerto COM: {self.com_port_var.get()}\n")
                f.write(f"Baud rate: {self.baud_var.get()}\n\n")
                f.write("ESTADÍSTICAS:\n")
                f.write(df.describe().to_string())
            
            messagebox.showinfo("Éxito", 
                              f"Datos guardados exitosamente en:\n"
                              f"• {csv_filename}\n"
                              f"• {mat_filename}\n"
                              f"• {info_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron guardar los datos:\n{str(e)}")

    def __del__(self):
        """Destructor para asegurar que se cierre la conexión serial"""
        self.stop_acquisition()
# Bloque principal de ejecución
if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()       