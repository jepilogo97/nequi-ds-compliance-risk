import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_optimization(grid_path, best_path, output_path):
    if not os.path.exists(grid_path) or not os.path.exists(best_path):
        print(f"Archivos no encontrados: {grid_path} o {best_path}")
        return

    df = pd.read_csv(grid_path)
    with open(best_path, 'r') as f:
        best = json.load(f)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Separar puntos factibles vs no factibles
    feasible = df[(df['prop_ok'] == True) & (df['fp_ok'] == True)]
    infeasible = df[(df['prop_ok'] == False) | (df['fp_ok'] == False)]

    # Graficar puntos no factibles (fondo, gris, transparente)
    ax.scatter(
        infeasible['t1'], 
        infeasible['t2'], 
        infeasible['recall_high'], 
        c='lightgray', 
        marker='o', 
        s=20, 
        alpha=0.3, 
        label='Infactible (Viola Restricciones)'
    )

    # Graficar puntos factibles (primer plano, colormap basado en Recall)
    # Se usa recall como color para resaltar los picos
    sc = ax.scatter(
        feasible['t1'], 
        feasible['t2'], 
        feasible['recall_high'], 
        c=feasible['recall_high'], 
        cmap='viridis', 
        marker='o', 
        s=60, 
        alpha=1.0, 
        label='Factible'
    )

    # Graficar el mejor punto
    ax.scatter(
        best['t1'], 
        best['t2'], 
        best['recall_high'], 
        c='red', 
        marker='*', 
        s=300, 
        edgecolors='black',
        label='Óptimo Seleccionado'
    )

    # Etiquetas
    ax.set_xlabel('Umbral T1 (Bajo / Medio)')
    ax.set_ylabel('Umbral T2 (Medio / Alto)')
    ax.set_zlabel('Recall (Grupo Alto)')
    ax.set_title('Espacio de Búsqueda 3D: Cortes vs Recall', fontsize=14)

    # Ángulo de vista
    ax.view_init(elev=20, azim=135)

    # Barra de color
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Recall')

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)  # DPI moderado para gráficos 3D
    print(f"Gráfica 3D guardada en {output_path}")
    plt.close()

if __name__ == "__main__":
    grid_csv = 'files/output/threshold_grid_results.csv'
    best_json = 'files/output/threshold_best.json'
    output_png = 'files/output/real_optimization_3d_es.png'
    
    plot_3d_optimization(grid_csv, best_json, output_png)
