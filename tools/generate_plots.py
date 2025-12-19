import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_risk_matrix(csv_path, output_path):
    # Leer los datos
    # Formato: probability_1_5, 1, 2, 3, 4, 5
    # La primera columna es el índice (Probabilidad 1-5).
    df = pd.read_csv(csv_path, index_col=0)
    
    # Ordenar el índice de forma descendente para que 5 (Alta probabilidad) quede arriba
    df = df.sort_index(ascending=False)
    
    # Configurar la gráfica
    plt.figure(figsize=(8, 6))
    
    ax = sns.heatmap(
        df,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        linewidths=.5,
        cbar_kws={'label': 'Cantidad de Riesgos'}
    )
    
    ax.set_title('Matriz de Riesgo (Conteo de Eventos Reales)', fontsize=14, pad=15)
    ax.set_ylabel('Probabilidad (Frecuencia)', fontsize=12)
    ax.set_xlabel('Impacto (Severidad)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Matriz guardada en {output_path}")
    plt.close()

def plot_ml_metrics(json_path, output_path):
    with open(json_path, 'r') as f:
        metrics = json.load(f)
        
    cm = np.array(metrics['confusion_matrix'])
    # cm = [[TN, FP], [FN, TP]]
    
    # Métricas para el título
    acc = metrics['accuracy']
    roc = metrics['roc_auc']
    f1 = metrics['f1']
    model_name = metrics.get('model_type', 'Modelo ML').upper()
    
    # Graficar la matriz de confusión
    plt.figure(figsize=(6, 5))
    
    labels = ['No Crítico (0)', 'Crítico (1)']
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.title(
        f'Resultados {model_name}\nAcc: {acc:.2f} | AUC: {roc:.2f} | F1: {f1:.2f}',
        fontsize=14
    )
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Gráfica ML guardada en {output_path}")
    plt.close()

def plot_aggregations(csv_path, output_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    
    # Usamos una paleta distinta para separar los tipos de riesgo
    sns.barplot(
        data=df,
        x='process',
        y='risks',
        hue='risk_type',
        palette='viridis'
    )
    
    plt.title('Distribución de Riesgos por Proceso y Tipo', fontsize=14)
    plt.xlabel('Proceso')
    plt.ylabel('Cantidad de Riesgos')
    plt.legend(title='Tipo de Riesgo', loc='upper right')
    
    # Opcional: agregar etiquetas de datos
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%d', padding=3, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Gráfica de agregaciones guardada en {output_path}")
    plt.close()

def plot_feature_importance(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Archivo de features no encontrado: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 8))
    
    # Verificación simple de los nombres de columnas
    if 'coef' in df.columns:
        # Regresión Logística
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in df['coef']]
        sns.barplot(data=df, x='coef', y='feature', palette=colors)
        plt.title('Importancia de Features (Regresión Logística)', fontsize=14)
        plt.xlabel('Coeficiente (Peso)')
    elif 'importance_mean' in df.columns:
        # Permutation Importance (XGBoost / Random Forest)
        sns.barplot(data=df, x='importance_mean', y='feature', palette='viridis')
        plt.title('Importancia de Features (Permutation Importance - XGB)', fontsize=14)
        plt.xlabel('Importancia Media (Impacto en F1-Score)')
    else:
        print("Formato de importancia de features desconocido")
        return

    plt.ylabel('Feature')
    plt.axvline(0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Importancia de features guardada en {output_path}")
    plt.close()

def plot_pr_curve(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Archivo de predicciones no encontrado: {csv_path}")
        return

    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    df = pd.read_csv(csv_path)
    y_true = df['y_true']
    y_score = df['y_score']
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'Curva PR (AP = {ap:.2f})')
    plt.fill_between(recall, precision, alpha=0.2)
    
    plt.xlabel('Recall (Sensibilidad)', fontsize=12)
    plt.ylabel('Precision (Precisión)', fontsize=12)
    plt.title(f'Curva Precision-Recall\nAverage Precision = {ap:.2f}', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Curva PR guardada en {output_path}")
    plt.close()

if __name__ == "__main__":
    import os
    # Rutas basadas en los pasos previos
    base_dir = "files"
    
    matrix_input = "files/output/risk_matrix_counts.csv"
    matrix_output = "files/output/real_risk_matrix_es.png"
    
    ml_input = "files/ml_output_xgb/metrics.json"
    ml_output = "files/output/real_ml_metrics_xgb_es.png"

    agg_input = "files/output/risk_aggregation_process_type.csv"
    agg_output = "files/output/real_risk_aggregation_es.png"
    
    # La salida de XGB usualmente tiene permutation importance,
    # verificar si existe o usar top_coefficients si fuera un modelo lineal
    # El CLI guarda top_permutation_importance.csv para modelos genéricos si está disponible
    # Revisamos lo que haya guardado el CLI. El CLI usa _explain_model.
    # Para XGB calcula permutation importance si feature_names están disponibles.
    feat_input = "files/ml_output_xgb/top_permutation_importance.csv"
    feat_output = "files/output/real_feature_importance_xgb_es.png"
    
    pred_input = "files/ml_output_xgb/predictions.csv"
    pr_output = "files/output/real_pr_curve_xgb_es.png"
    
    # Generar gráficas
    try:
        plot_risk_matrix(matrix_input, matrix_output)
    except Exception as e:
        print(f"Error graficando la matriz: {e}")
        
    try:
        plot_ml_metrics(ml_input, ml_output)
    except Exception as e:
        print(f"Error graficando métricas ML: {e}")

    try:
        plot_aggregations(agg_input, agg_output)
    except Exception as e:
        print(f"Error graficando agregaciones: {e}")
        
    try:
        plot_feature_importance(feat_input, feat_output)
    except Exception as e:
        print(f"Error graficando features: {e}")
        
    try:
        plot_pr_curve(pred_input, pr_output)
    except Exception as e:
        print(f"Error graficando Curva PR: {e}")
