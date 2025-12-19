import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def plot_optimization_tradeoff(grid_path, best_path, output_path):
    if not os.path.exists(grid_path) or not os.path.exists(best_path):
        print(f"Files not found: {grid_path} or {best_path}")
        return

    df = pd.read_csv(grid_path)
    with open(best_path, 'r') as f:
        best = json.load(f)

    # Gráfica
    plt.figure(figsize=(10, 7))
    
    # Definir categorías para la gráfica
    # Factible si prop_ok Y fp_ok
    df['Status'] = 'Infeasible'
    df.loc[(df['prop_ok'] == True) & (df['fp_ok'] == True), 'Status'] = 'Feasible'
    
    # Gráfico de dispersión
    sns.scatterplot(
        data=df, 
        x='fp_rate_high', 
        y='recall_high', 
        hue='Status',
        palette={'Infeasible': 'lightgray', 'Feasible': '#3498db'},
        alpha=0.6,
        s=40
    )
    
    # Resaltar el mejor punto
    plt.scatter(
        best['fp_rate_high'], 
        best['recall_high'], 
        color='#e74c3c', 
        s=200, 
        marker='*', 
        label=f"Selected (Recall={best['recall_high']:.2f})"
    )
    
    # Dibujar línea de restricción (FP Máx = 0.15)
    plt.axvline(0.15, color='gray', linestyle='--', label='FP Restriction (0.15)')
    
    # Etiquetas
    plt.title(
        'Optimización de Umbrales: Trade-off Recall vs FP Rate (Segmento Alto)',
        fontsize=14
    )
    plt.xlabel('Tasa de Falsos Positivos en Alto (FP Rate)', fontsize=12)
    plt.ylabel('Recall de Críticos en Alto', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Anotación para el mejor punto
    plt.annotate(
        f"t1={best['t1']:.2f}, t2={best['t2']:.2f}",
        xy=(best['fp_rate_high'], best['recall_high']),
        xytext=(best['fp_rate_high'] + 0.05, best['recall_high'] - 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Optimization plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    grid_csv = 'files/output/threshold_grid_results.csv'
    best_json = 'files/output/threshold_best.json'
    output_png = 'files/output/real_optimization_tradeoff_es.png'
    
    plot_optimization_tradeoff(grid_csv, best_json, output_png)
