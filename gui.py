import tkinter as tk
from tkinter import filedialog, ttk
import threading
import perceptron as pm

def elegir_archivo(etiqueta_archivo):
    nombre_archivo = filedialog.askopenfilename(initialdir="/", title="Seleccionar archivo",
                                                filetypes=(("archivos csv", "*.csv"), ("todos los archivos", "*.*")))
    etiqueta_archivo.config(text=nombre_archivo)
    return nombre_archivo

def comenzar_entrenamiento(entrada_eta, entrada_epocas, etiqueta_archivo, barra_progreso):
    tasa = float(entrada_eta.get())
    num_epocas = int(entrada_epocas.get())
    archivo = etiqueta_archivo.cget("text")

    def proceso_entrenamiento():
        pm.iniciar_aprendizaje(tasa, num_epocas, archivo, lambda epoca: actualizar_barra_progreso(barra_progreso, epoca, num_epocas))

    threading.Thread(target=proceso_entrenamiento).start()

def actualizar_barra_progreso(barra_progreso, epoca_actual, epocas_maximas):
    valor_progreso = (epoca_actual / epocas_maximas) * 100
    barra_progreso['value'] = valor_progreso
    barra_progreso.update_idletasks()



def visualizar_graficos():
    pm.visualizar_resultados()

def crear_reporte():
    inicio_pesos, fin_pesos, epocas_realizadas, error_maximo = pm.datos_pesos()
    ventana = tk.Toplevel()
    ventana.title("Resultados")
    marco = tk.LabelFrame(ventana, text=" Datos generales (REPORTE)", padx=10, pady=10)
    marco.pack(padx=10, pady=10, fill="both", expand="yes")
    texto_reporte = tk.Text(marco, height=15, width=50)
    texto_reporte.pack(expand=True)
    texto_reporte.insert(tk.END, f"Número de épocas: {epocas_realizadas}\n")
    texto_reporte.insert(tk.END, f"Error permitido: {error_maximo}\n\n")
    agregar_pesos(texto_reporte, inicio_pesos, "Configuración inicial de pesos:\n")
    agregar_pesos(texto_reporte, fin_pesos, "Configuración final de pesos:\n")
    texto_reporte.config(state=tk.DISABLED)

def agregar_pesos(cuadro_texto, pesos, titulo):
    cuadro_texto.insert(tk.END, titulo)
    
    if pesos.ndim == 1:  # Si 'pesos' es un vector 1D
        linea = " ".join([f"{peso:.4f}" for peso in pesos])
        cuadro_texto.insert(tk.END, linea + "\n")
    else:  # Si 'pesos' es una matriz 2D
        for fila in pesos:
            linea = " ".join([f"{peso:.4f}" for peso in fila])
            cuadro_texto.insert(tk.END, linea + "\n")

    cuadro_texto.insert(tk.END, "\n")


def iniciar_interfaz():
    raiz = tk.Tk()
    raiz.title("Entrenamiento del Perceptrón")
    estilo = ttk.Style()
    estilo.theme_use('clam')
    color_fondo = '#f3f2f5'
    fuente = ('Helvetica', 10, 'bold') 
    estilo.configure('TFrame', background=color_fondo)
    estilo.configure('TButton', background=color_fondo, foreground='black', font=fuente)
    estilo.configure('TLabel', background=color_fondo, foreground='black', font=fuente)
    estilo.configure('TLabelframe', background=color_fondo, foreground='black', font=fuente)
    estilo.configure('TLabelframe.Label', background=color_fondo, foreground='black', font=fuente)
    estilo.configure('TEntry', fieldbackground='white', font=fuente)
    estilo.configure('TProgressbar', troughcolor=color_fondo, background='green')
    raiz.configure(bg=color_fondo)
    contenedor = ttk.LabelFrame(raiz, text=' Entrenamiento del Perceptrón ', padding="10 10 10 10", style='TLabelframe')
    contenedor.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    frame_superior = ttk.Frame(contenedor, style='TFrame')
    frame_superior.pack(fill=tk.X, padx=5, pady=5)
    ttk.Label(frame_superior, text="Tasa de aprendizaje (eta):", background=color_fondo).pack(side=tk.LEFT, padx=5, pady=5)
    entrada_eta = ttk.Entry(frame_superior, width=20)
    entrada_eta.pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Label(frame_superior, text="Número de épocas (Iteraciones):", background=color_fondo).pack(side=tk.LEFT, padx=5, pady=5)
    entrada_epocas = ttk.Entry(frame_superior, width=20)
    entrada_epocas.pack(side=tk.LEFT, padx=5, pady=5)
    frame_medio = ttk.Frame(contenedor, style='TFrame')
    frame_medio.pack(fill=tk.X, padx=5, pady=5)
    ttk.Button(frame_medio, text="Seleccionar archivo CSV", command=lambda: elegir_archivo(etiqueta_archivo)).pack(side=tk.LEFT, padx=5, pady=5)
    etiqueta_archivo = ttk.Label(frame_medio, text="", width=60, relief=tk.SUNKEN, background='white')
    etiqueta_archivo.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
    frame_inferior = ttk.Frame(contenedor, style='TFrame')
    frame_inferior.pack(fill=tk.X, pady=5)
    ttk.Button(frame_inferior, text="INICIAR", command=lambda: comenzar_entrenamiento(entrada_eta, entrada_epocas, etiqueta_archivo, barra_progreso)).pack(side=tk.LEFT, expand=True)
    ttk.Button(frame_inferior, text="VISUALIZAR GRÁFICAS", command=visualizar_graficos).pack(side=tk.LEFT, expand=True)
    ttk.Button(frame_inferior, text="GENERAR REPORTE", command=crear_reporte).pack(side=tk.LEFT, expand=True)
    frame_barra_progreso = ttk.Frame(contenedor, style='TFrame')
    frame_barra_progreso.pack(fill=tk.X, pady=5)
    barra_progreso = ttk.Progressbar(frame_barra_progreso, style='TProgressbar', orient="horizontal", length=200, mode='determinate')
    barra_progreso.pack(expand=True)
    raiz.mainloop()
