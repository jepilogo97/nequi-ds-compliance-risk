import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_optimization_summary(best_path, output_path):
    if not os.path.exists(best_path):
        print(f"File not found: {best_path}")
        return

    with open(best_path, 'r') as f:
        best = json.load(f)

    # REINICIAR FIGURA
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    
    # Título
    fig.text(
        0.5, 0.93,
        'Resumen de Optimización: Configuración Seleccionada', 
        ha='center', va='center',
        fontsize=18, fontweight='bold', color='#2c3e50'
    )
    
    # Caja izquierda: Umbrales
    fig.text(
        0.25, 0.82,
        'Umbrales de Corte (Scores)',
        ha='center',
        fontsize=14,
        fontweight='bold',
        color='#34495e'
    )
    fig.text(
        0.25, 0.77,
        f"t1 (Bajo/Medio): {best['t1']:.3f}",
        ha='center',
        fontsize=16,
        color='#e67e22',
        fontweight='bold'
    )
    fig.text(
        0.25, 0.72,
        f"t2 (Medio/Alto): {best['t2']:.3f}",
        ha='center',
        fontsize=16,
        color='#c0392b',
        fontweight='bold'
    )
    
    # Caja derecha: Desempeño
    fig.text(
        0.75, 0.82,
        'Desempeño Segmento Alto',
        ha='center',
        fontsize=14,
        fontweight='bold',
        color='#34495e'
    )
    fig.text(
        0.75, 0.77,
        f"Recall: {best['recall_high']*100:.1f}%",
        ha='center',
        fontsize=16,
        color='#27ae60',
        fontweight='bold'
    )
    
    fp_color = '#c0392b' if best['fp_rate_high'] > 0.15 else '#27ae60'
    fig.text(
        0.75, 0.72,
        f"FP Rate: {best['fp_rate_high']*100:.1f}%",
        ha='center',
        fontsize=16,
        color=fp_color,
        fontweight='bold'
    )
    fig.text(
        0.75, 0.68,
        "(Max permitido: 15%)",
        ha='center',
        fontsize=10,
        color='gray'
    )

    # Línea divisoria
    fig.add_artist(
        plt.Line2D([0.1, 0.9], [0.63, 0.63], color='lightgray')
    )

    # Gráfico de barras para proporciones
    ax = fig.add_axes([0.15, 0.1, 0.7, 0.45])
    
    categories = ['Bajo', 'Medio', 'Alto']
    targets = [0.70, 0.20, 0.10]
    reals = [best['prop_low'], best['prop_med'], best['prop_high']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(
        x - width/2,
        targets,
        width,
        label='Target',
        color='#bdc3c7'
    )
    rects2 = ax.bar(
        x + width/2,
        reals,
        width,
        label='Real (Optimizado)',
        color='#3498db'
    )
    
    ax.set_ylabel('Proporción de Población')
    ax.set_title('Distribución Poblacional: Planned vs Real', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 0.85)
    ax.legend()
    
    # Etiquetas de las barras
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f'{height*100:.1f}%',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    best_json = 'files/output/threshold_best.json'
    output_png = 'files/output/real_optimization_summary_es.png'
    
    plot_optimization_summary(best_json, output_png)
