from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.ml.train import train_supervised_model
from src.ml.features import FeatureSpec
from src.ml.model import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un modelo supervisado para `critical_flag`")

    parser.add_argument("--input", required=True, help="CSV de entrada con datos de riesgos")
    parser.add_argument("--out", required=True, help="Directorio de salida para artefactos")
    parser.add_argument("--model-type", choices=["logreg", "rf", "xgb"], default="logreg", help="Tipo de modelo")
    parser.add_argument("--use-smote", action="store_true", help="Aplicar SMOTE durante el entrenamiento (requiere imbalanced-learn)")
    parser.add_argument("--use-risk-metrics", action="store_true", help="Ejecutar internamente el pipeline de risk_matrix para enriquecer features (freq_annual_adj, severity_avg, el_annual_adj)")
    parser.add_argument("--test-size", type=float, default=0.25, help="Tamaño del test (fracción)")
    parser.add_argument("--random-state", type=int, default=42, help="Seed para reproducibilidad")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Cargar dataset
    df = pd.read_csv(input_path)

    # Si se solicita, ejecutar pipeline de risk_matrix para enriquecer features
    if args.use_risk_metrics:
        import sys
        repo_root = Path(__file__).parents[2]
        # añadir el paquete risk_matrix 
        risk_pkg = repo_root / "risk_matrix"
        if str(risk_pkg) not in sys.path:
            sys.path.insert(0, str(risk_pkg))

        try:
            # intentar importar el pipeline como `src.pipeline`
            from src.pipeline import run_pipeline
        except Exception as e:
            # fallback: intentar importar como paquete completo risk_matrix.src.pipeline
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            try:
                from risk_matrix.src.pipeline import run_pipeline
            except Exception:
                raise ImportError(
                    "No se pudo importar run_pipeline de risk_matrix; asegúrate de que 'risk_matrix' esté en el repo y que PYTHONPATH sea correcto"
                ) from e

        print("🔁 Ejecutando pipeline de risk_matrix para enriquecer features...")
        rp = run_pipeline(df, None)
        # rp['metrics'] es un DataFrame con columnas calculadas; usarlo como df de entrenamiento
        df = rp["metrics"]

    # 2) Configurar feature spec y modelo
    feature_spec = FeatureSpec()
    model_cfg = ModelConfig(model_type=args.model_type)

    # 3) Ejecutar entrenamiento
    res = train_supervised_model(
        df,
        feature_spec=feature_spec,
        model_cfg=model_cfg,
        test_size=args.test_size,
        random_state=args.random_state,
        use_smote=args.use_smote,
    )

    # 4) Guardar artefactos
    # Pipeline (modelo + preprocesador)
    model_path = out_dir / "model_pipeline.joblib"
    joblib.dump(res["pipeline"], model_path)

    # Metrics
    metrics_path = out_dir / "metrics.json"
    metrics_data = res["metrics"]
    metrics_data["model_type"] = args.model_type
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_data, fh, indent=2, ensure_ascii=False)

    # Used features
    features_path = out_dir / "used_features.json"
    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump(res.get("used_features", {}), fh, indent=2, ensure_ascii=False)

    # Explainability: salvar dataframes si existen
    explain = res.get("explainability", {})
    if explain.get("top_coefficients") is not None:
        explain["top_coefficients"].to_csv(out_dir / "top_coefficients.csv", index=False)
    if explain.get("top_permutation_importance") is not None:
        explain["top_permutation_importance"].to_csv(out_dir / "top_permutation_importance.csv", index=False)

    # Guardar predicciones para ploteo externo
    y_test = res.get("y_test")
    y_score = res.get("y_score")
    if y_test is not None and y_score is not None:
        pd.DataFrame({"y_true": y_test, "y_score": y_score}).to_csv(out_dir / "predictions.csv", index=False)

    # Guardar matriz de confusión y curvas PR/ROC como PNG
    y_test = res.get("y_test")
    y_pred = res.get("y_pred")
    y_score = res.get("y_score")
    try:
        if y_test is not None and y_pred is not None:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            fig.suptitle("Matriz de confusión")
            fig.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
            plt.close(fig)

        if y_score is not None:
            # ROC
            try:
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = res.get("metrics", {}).get("roc_auc", None)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}" if roc_auc is not None else "ROC")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC curve")
                ax.legend(loc="lower right")
                fig.savefig(out_dir / "roc_curve.png", bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

            # Precision-Recall
            try:
                prec, rec, _ = precision_recall_curve(y_test, y_score)
                pr_auc = res.get("metrics", {}).get("pr_auc", None)
                fig, ax = plt.subplots()
                ax.plot(rec, prec, label=f"AP = {pr_auc:.3f}" if pr_auc is not None else "PR")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall curve")
                ax.legend(loc="lower left")
                fig.savefig(out_dir / "pr_curve.png", bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass
    except Exception:
        # No bloquear el CLI por errores de plotting
        pass

    print("✅ Entrenamiento completado")
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
