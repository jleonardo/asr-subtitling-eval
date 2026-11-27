"""Rule-based terminology normalization used by the transcription application.

Maps frequent ASR misrecognitions of domain terms (e.g., CESyA, CEACOG, RPD) to their canonical form.
In the paper this corresponds to the "normalization" variant of the ablation study.
The dictionary is kept exactly as used in the experiments.
"""
import re
from typing import Dict

# Mapa personalizable de variantes -> forma canónica
# Puedes añadir aquí las variantes que quieras que se conviertan a 'CESyA'.
CUSTOM_PHRASE_MAP: Dict[str, str] = {
    # Ejemplos solicitados:
    "cesya": "CESyA",
    "cesia": "CESyA",
    "cecia": "CESyA",
    "sesia":"CESyA",
    "secia":"CESyA",
    "csia":"CESyA",
    "secya":"CESyA",
    "CESIA":"CESyA",
    "CECIA":"CESyA",
    "cesía":"CESyA",
    "CESÍA":"CESyA",
    "CECÍA":"CESyA",
    "Ecesia":"CESyA",
    "ECESIA":"CESyA",
    "ECESÍA":"CESyA",
    "CSIA":"CESyA",
    "csia":"CESyA",
    "CSÍA":"CESyA",
    "csía":"CESyA",
    "CCIA":"CESyA",
    "ccsia":"CESyA",
    "CCSÍA":"CESyA",
    "ccsía":"CESyA",
    "fesía":"CESyA",
    "Ecesí":"CESyA",
    "sesilla":"CESyA",
    "SESSIA":"CESyA",
    "sessia":"CESyA",
    "SESSÍA":"CESyA",
    "sessía":"CESyA",
    "cesyá":"CESyA",
    "CESYÁ":"CESyA",
    "cesyia":"CESyA",
    "CESYIA":"CESyA",
    "cesyía":"CESyA",
    "CESYÍA":"CESyA",
    "se cía":"CESyA",
    "cesilla":"CESyA",
    "se silla": "CESyA",
    "se cía":"CESyA",
    "se cia":"CESyA",
    "se sía":"CESyA",
    "seya":"CESyA",
    "CESIC":"CESyA",
    "rpd":"RPD",
    "CEACO":"CEACOG",
    "CEACOJ":"CEACOG",
    "ceaco":"CEACOG",
    "ceacoj":"CEACOG",
    "caco":"CEACOG",
    "CACO":"CEACOG",
    "":"",
    # Agrega más variantes si lo necesitas:
    # "cesyaa": "CESyA",
}

# Compilar patrones con límites de palabra, insensibles a mayúsculas/minúsculas
_compiled_patterns = [
    (re.compile(r"\b" + re.escape(variant) + r"\b", re.IGNORECASE), canonical)
    for variant, canonical in CUSTOM_PHRASE_MAP.items()
]

# Patrones adicionales robustos (incluye tildes, confusiones c/s y espacio opcional)
_robust_patterns = [
    # cesia/cecia/sesia/secia con o sin tilde: cesía/cecÍa/sesía/secía
    (re.compile(r"\b(?:c|s)e(?:c|s)(?:i|í)a\b", re.IGNORECASE), "CESyA"),
    # ces ya (con espacio opcional)
    (re.compile(r"\bces\s*ya\b", re.IGNORECASE), "CESyA"),
]


def normalize_text(text: str) -> str:
    """Reemplaza variantes por sus formas canónicas según CUSTOM_PHRASE_MAP.
    - Respeta límites de palabra, no afecta subcadenas dentro de otras palabras.
    - Es insensible a mayúsculas/minúsculas.
    """
    if not text:
        return text
    normalized = text
    # 1) Reglas explícitas del usuario
    for pattern, canonical in _compiled_patterns:
        normalized = pattern.sub(canonical, normalized)
    # 2) Patrones robustos
    for pattern, canonical in _robust_patterns:
        normalized = pattern.sub(canonical, normalized)
    return normalized
