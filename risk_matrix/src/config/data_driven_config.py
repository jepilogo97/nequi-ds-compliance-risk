from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# Variables globales que almacenarán los valores por defecto
# (se intentan calcular desde un CSV y, si falla, usan fallback)
DEFAULT_FREQ_BINS: Tuple[float, float, float, float]
DEFAULT_SEV_BINS: Tuple[int, int, int, int]
DEFAULT_CONTROLS_FREQ_MULTIPLIER: Dict[int, float]
DEFAULT_CONTROLS_LEVEL_MAPPING: Dict[str, int]


def _linear_fill(mapping: Dict[int, float], keys: range) -> Dict[int, float]:
    """
    Rellena claves enteras faltantes (por ejemplo 1..5) usando interpolación lineal.

    Ejemplo:
    mapping = {1: 0.7, 3: 1.0, 5: 1.3}
    => completa automáticamente 2 y 4 de forma interpolada
    """
    out = dict(mapping)            # Copia para no mutar el input
    known = sorted(out.items())    # Pares (clave, valor) conocidos ordenados

    for k in keys:
        # Si la clave ya existe, no se toca
        if k in out:
            continue

        # Busca el vecino izquierdo y derecho más cercanos
        left = max((i for i, _ in known if i < k), default=None)
        right = min((i for i, _ in known if i > k), default=None)

        # Si no hay vecinos conocidos, usar valor neutro
        if left is None and right is None:
            out[k] = 1.0
        # Si solo existe el vecino derecho
        elif left is None:
            out[k] = out[right]
        # Si solo existe el vecino izquierdo
        elif right is None:
            out[k] = out[left]
        else:
            # Interpolación lineal entre left y right
            vleft = out[left]
            vright = out[right]
            out[k] = vleft + (vright - vleft) * ((k - left) / (right - left))

    return out


def compute_from_csv(
    csv_path: str | Path
) -> Tuple[
    Tuple[float, float, float, float],
    Tuple[int, int, int, int],
    Dict[int, float],
    Dict[str, int]
]:
    """
    Calcula automáticamente:
    - freq_bins
    - sev_bins
    - multiplicadores por nivel de control
    - mapping de nivel de control

    a partir de un dataset histórico en CSV.
    """
    p = Path(csv_path)
    df = pd.read_csv(p)

    # --------------------------------------------------
    # 1. FRECUENCIA (eventos por año)
    # --------------------------------------------------
    # events_last_3y está en 3 años → se anualiza
    freq = df["events_last_3y"] / 3.0

    # Se usan cuantiles para definir los cortes
    fq = freq.quantile([0.2, 0.5, 0.8, 0.95]).to_list()

    # Se convierten a una tupla explícita de floats
    freq_bins = (
        float(fq[0]),
        float(fq[1]),
        float(fq[2]),
        float(fq[3]),
    )

    # --------------------------------------------------
    # 2. SEVERIDAD (pérdida promedio por evento)
    # --------------------------------------------------
    # Evita división por cero reemplazando eventos = 0 por NaN
    sev = (
        df["total_loss_last_3y"]
        / df["events_last_3y"].replace(0, np.nan)
    ).fillna(0)

    # Cuantiles de severidad
    svq = sev.quantile([0.2, 0.5, 0.8, 0.95]).to_list()

    # Se convierten a enteros (dinero)
    sev_bins = tuple(int(float(x)) for x in svq)

    # --------------------------------------------------
    # 3. MULTIPLICADORES POR NIVEL DE CONTROL
    # --------------------------------------------------
    # Promedio de frecuencia anual por nivel de control
    ctrl_mean = (
        df.groupby("controls_level")["events_last_3y"].mean() / 3.0
    )

    # Mapping estándar de texto → escala numérica
    controls_level_mapping = {
        "Alto": 1,
        "Medio": 3,
        "Bajo": 5,
    }

    # Diccionario con niveles observados en el dataset
    observed: Dict[int, float] = {}
    medio_val = None

    for label, num in controls_level_mapping.items():
        if label in ctrl_mean.index:
            observed[num] = float(ctrl_mean.loc[label])

            # Se usa "Medio" como baseline
            if label == "Medio":
                medio_val = float(ctrl_mean.loc[label])

    # Si no hay nivel "Medio" en los datos, se usa el promedio global
    if medio_val is None:
        medio_val = float(freq.mean())
        observed.setdefault(3, medio_val)

    # Normalización: Medio == 1.0
    norm = {k: (v / medio_val) for k, v in observed.items()}

    # Rellenar niveles faltantes (1..5) usando interpolación
    controls_freq_multiplier = _linear_fill(norm, range(1, 6))

    return (
        freq_bins,
        sev_bins,
        controls_freq_multiplier,
        controls_level_mapping,
    )


# --------------------------------------------------
# CÁLCULO AUTOMÁTICO DE VALORES POR DEFECTO
# --------------------------------------------------

# Se asume una estructura de proyecto conocida
_project_root = Path(__file__).parents[3]
_sample_csv = _project_root / "data" / "raw" / "dataset_dummy_compliance.csv"

try:
    # Si existe un dataset de ejemplo, se calculan los defaults desde datos reales
    if _sample_csv.exists():
        (
            DEFAULT_FREQ_BINS,
            DEFAULT_SEV_BINS,
            DEFAULT_CONTROLS_FREQ_MULTIPLIER,
            DEFAULT_CONTROLS_LEVEL_MAPPING,
        ) = compute_from_csv(_sample_csv)
    else:
        raise FileNotFoundError()

except Exception:
    # --------------------------------------------------
    # FALLBACKS (valores razonables por defecto)
    # --------------------------------------------------
    DEFAULT_FREQ_BINS = (0.33, 1.0, 1.67, 2.0)
    DEFAULT_SEV_BINS = (750_000, 2_300_000, 3_910_000, 4_720_000)
    DEFAULT_CONTROLS_FREQ_MULTIPLIER = {
        1: 1.125,
        2: 1.063,
        3: 1.0,
        4: 1.029,
        5: 1.059,
    }
    DEFAULT_CONTROLS_LEVEL_MAPPING = {
        "Alto": 1,
        "Medio": 3,
        "Bajo": 5,
    }
