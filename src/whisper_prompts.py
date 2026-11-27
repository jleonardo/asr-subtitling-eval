"""
Initial prompt used for the "prompt" variant of the ablation study (Section on post-processing).
DEFAULT_PROMPT is the exact prompt used in the experiments.

Configuración de prompts iniciales para mejorar el reconocimiento de Whisper.
El initial_prompt ayuda a Whisper a reconocer mejor palabras específicas, nombres propios,
acrónimos y terminología técnica sin necesidad de fine-tuning.

Uso:
- Agrega palabras que Whisper confunde frecuentemente
- Incluye nombres propios, marcas, acrónimos
- Usa la forma correcta que quieres que aparezca en las transcripciones
- El prompt influye en el estilo de transcripción del modelo
"""

# Lista de palabras clave y términos específicos que queremos que Whisper reconozca correctamente
# Basado en las normalizaciones actuales y términos comunes
PALABRAS_CLAVE = [
    # Acrónimos y nombres propios principales
    "CESyA",
    "UC3M",
    "RPD",
    "CEACOG",
    
    # Términos técnicos y contexto
    "accesibilidad",
    "subtitulado",
    "transcripción",
    "audiodescripción",
    "tecnología",
    
    # Agrega más palabras según tus necesidades:
    # "nombre_empresa",
    # "producto_específico",
    # "términos_técnicos",
]

# Contexto adicional para mejorar la transcripción
CONTEXTO_GENERAL = "Transcripción de contenido sobre accesibilidad y tecnología."

def get_initial_prompt(contexto_adicional: str = "") -> str:
    """
    Genera el initial_prompt combinando palabras clave y contexto.
    
    Args:
        contexto_adicional: Contexto específico adicional del usuario
    
    Returns:
        String con el prompt completo para Whisper
    """
    # Combinar palabras clave en una frase natural
    palabras = ", ".join(PALABRAS_CLAVE)
    
    # Construir prompt
    partes = []
    
    if CONTEXTO_GENERAL:
        partes.append(CONTEXTO_GENERAL)
    
    if palabras:
        partes.append(f"Palabras clave: {palabras}.")
    
    if contexto_adicional:
        partes.append(contexto_adicional)
    
    return " ".join(partes)


def agregar_palabras(nuevas_palabras: list):
    """
    Agrega nuevas palabras a la lista de palabras clave.
    
    Args:
        nuevas_palabras: Lista de palabras a agregar
    """
    for palabra in nuevas_palabras:
        if palabra not in PALABRAS_CLAVE:
            PALABRAS_CLAVE.append(palabra)


# Prompt por defecto (se puede sobrescribir con contexto específico)
DEFAULT_PROMPT = get_initial_prompt()
