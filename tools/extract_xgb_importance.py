import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def extract_and_plot_xgb_importance(model_path, output_csv, output_img):
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado: {model_path}")
        return

    try:
        pipe = joblib.load(model_path)
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    # Extraer nombres de las features desde el preprocesador
    feature_names = []
    try:
        if 'preprocessor' in pipe.named_steps:
            pre = pipe.named_steps['preprocessor']
            if hasattr(pre, 'get_feature_names_out'):
                feature_names = pre.get_feature_names_out()
            else:
                # Intentar inspeccionar los transformers
                from sklearn.compose import ColumnTransformer
                if isinstance(pre, ColumnTransformer):
                     pass
    except Exception as e:
        print(f"Error obteniendo nombres de features: {e}")

    # Extraer el modelo
    model = pipe.named_steps.get('model')
    if model is None:
        print("El paso 'model' no se encontró en el pipeline")
        print(pipe.named_steps.keys())
        return

    # Verificar si es XGBoost
    is_xgb = 'XGB' in str(type(model))
    
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    
    if importances is None:
        print("El modelo no tiene feature_importances_")
        return

    # Si no hay nombres de features, mapear por índice
    if len(feature_names) != len(importances):
        print(f"Desajuste: {len(feature_names)} nombres vs {len(importances)} importancias")
        # Generar nombres dummy
        if len(feature_names) == 0:
            feature_names = [f"f{i}" for i in range(len(importances))]
        else:
            # ¿Truncar o extender?
            print("No se pueden mapear las features de forma segura.")
            return
            
    # Crear DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importances  # Se llama importance_mean para coincidir con lo que espera generate_plots
    })
    
    df = df.sort_values('importance_mean', ascending=False).head(20)
    
    # Guardar CSV
    df.to_csv(output_csv, index=False)
    print(f"Importancias guardadas en {output_csv}")
    
    # Gráfica
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x='importance_mean', y='feature', palette='viridis')
    plt.title('Importancia de Features XGBoost (Gain/Weight)', fontsize=14)
    plt.xlabel('Importancia')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Imagen guardada en {output_img}")

if __name__ == '__main__':
    model_path = 'files/ml_output_xgb/model_pipeline.joblib'
    output_csv = 'files/ml_output_xgb/top_permutation_importance.csv'  # Se reutiliza este nombre para que generate_plots también pueda usarlo
    output_img = 'files/output/real_feature_importance_xgb_es.png'
    
    extract_and_plot_xgb_importance(model_path, output_csv, output_img)
