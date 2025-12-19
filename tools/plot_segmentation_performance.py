import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_segmentation_confusion(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Mapear etiquetas crudas a encabezados legibles
    df['Label'] = df['y_true'].map({0: 'No Crítico', 1: 'Crítico'})
    
    # Orden de las categorías
    cat_order = ['bajo', 'medio', 'alto']
    label_order = ['No Crítico', 'Crítico']
    
    # Crear tabla cruzada (Matriz de confusión)
    ct = pd.crosstab(df['Label'], df['category'])
    
    # Reindexar para asegurar que existan todas las filas/columnas
    ct = ct.reindex(index=label_order, columns=cat_order, fill_value=0)
    
    # Gráfica
    plt.figure(figsize=(8, 6))
    
    # Heatmap
    sns.heatmap(
        ct,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        annot_kws={"size": 14}
    )
    
    plt.title('Matriz de Desempeño: Segmentación vs Realidad', fontsize=14, pad=15)
    plt.xlabel('Segmento Asignado (Predicción)', fontsize=12)
    plt.ylabel('Criticidad Real (Etiqueta)', fontsize=12)
    
    # Capitalizar etiquetas para la visualización
    ax = plt.gca()
    ax.set_xticklabels([c.title() for c in cat_order])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Confusion Matrix saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    csv_input = 'files/output/annotated_scores.csv'
    png_output = 'files/output/real_segmentation_confusion_es.png'
    
    plot_segmentation_confusion(csv_input, png_output)
