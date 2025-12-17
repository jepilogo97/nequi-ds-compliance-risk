from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline import run_pipeline
from src.config import RiskMatrixConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Risk Matrix Pipeline - Compliance / ERM"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path al CSV de entrada (dataset de riesgos)",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Directorio de salida para los resultados",
    )

    parser.add_argument(
        "--prob-method",
        choices=["quantile", "fixed"],
        default="quantile",
        help="Método para asignar probabilidad (1–5)",
    )

    parser.add_argument(
        "--impact-method",
        choices=["quantile", "fixed"],
        default="quantile",
        help="Método para asignar impacto (1–5)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar dataset
    df = pd.read_csv(input_path)

    # 2. Configurar pipeline
    cfg = RiskMatrixConfig(
        probability_method=args.prob_method,
        impact_method=args.impact_method,
    )

    # 3. Ejecutar pipeline
    results = run_pipeline(df, cfg)

    # 4. Guardar outputs
    results["metrics"].to_csv(out_dir / "risk_metrics.csv", index=False)
    results["scored"].to_csv(out_dir / "risk_scored.csv", index=False)
    results["matrix_counts"].to_csv(out_dir / "risk_matrix_counts.csv")
    results["matrix_el"].to_csv(out_dir / "risk_matrix_expected_loss.csv")
    results["agg_process_type"].to_csv(
        out_dir / "risk_aggregation_process_type.csv", index=False
    )

    print("✅ Pipeline ejecutado correctamente")
    print(f"📂 Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
