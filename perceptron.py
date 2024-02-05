import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pesos_inicio = None
pesos_termino = None
error_por_epoca = []
historial_pesos = []
limite_error = 0
total_epocas = 0

def iniciar_aprendizaje(tasa_aprendizaje, epocas_maximas, archivo_datos, callback_actualizar_progreso=None):
    global pesos_inicio, total_epocas, pesos_termino, error_por_epoca, historial_pesos

    error_por_epoca.clear()
    historial_pesos.clear()

    datos, matriz_x, vector_y = cargar_datos(archivo_datos)
    pesos_actuales = inicializar_pesos(len(datos.columns) - 1)

    pesos_inicio = pesos_actuales.copy()
    total_epocas = epocas_maximas

    print(f"Estos son los pesos iniciales: {pesos_actuales}")

    for _ in range(len(pesos_actuales)):
        historial_pesos.append([])

    for epoca in range(epocas_maximas):
        y_predicha, error = realizar_prediccion(matriz_x, pesos_actuales, vector_y)
        actualizar_historiales(pesos_actuales, error, historial_pesos)
        
        ajustar_pesos(matriz_x, error, tasa_aprendizaje, pesos_actuales)

        if callback_actualizar_progreso:
            callback_actualizar_progreso(epoca)

    pesos_termino = pesos_actuales

def cargar_datos(archivo_datos):
    datos = pd.read_csv(archivo_datos, delimiter=';', header=None)
    matriz_x = np.hstack([datos.iloc[:, :-1].values, np.ones((datos.shape[0], 1))])
    vector_y = datos.iloc[:, -1].values.reshape(-1, 1)
    return datos, matriz_x, vector_y

def inicializar_pesos(num_columnas):
    return np.random.uniform(low=0, high=1, size=(num_columnas + 1, 1)).round(4)

def realizar_prediccion(matriz_x, pesos, vector_y):
    producto_punto = np.dot(matriz_x, pesos)
    y_predicha = np.where(producto_punto >= 0, 1, 0)
    error = vector_y - y_predicha
    return y_predicha, error

def actualizar_historiales(pesos, error, historial):
    norma_error = np.linalg.norm(error)
    error_por_epoca.append(norma_error)
    for i in range(len(pesos)):
        historial[i].append(pesos[i, 0])

def ajustar_pesos(matriz_x, error, tasa_aprendizaje, pesos):
    ajuste_pesos = np.dot(matriz_x.T, error)
    delta_pesos = tasa_aprendizaje * ajuste_pesos
    pesos += np.round(delta_pesos, 4)

def visualizar_resultados():
    plt.style.use('dark_background')   
 
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(error_por_epoca) + 1), error_por_epoca)
    plt.title('Evolución del Error Absoluto')
    plt.xlabel('Época')
    plt.ylabel('Error Absoluto')
 
    plt.figure(figsize=(6, 4))
    for i, peso_por_epoca in enumerate(historial_pesos):
        plt.plot(range(1, len(peso_por_epoca) + 1), peso_por_epoca, label=f'Peso {i + 1}')
    plt.title('Evolución de Pesos')
    plt.xlabel('Época')
    plt.ylabel('Valor de Pesos')
    plt.legend()

    plt.tight_layout()
    plt.show()  

def datos_pesos():
    return pesos_inicio, pesos_termino, total_epocas, limite_error
