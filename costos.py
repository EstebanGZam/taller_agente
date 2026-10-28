"""Costo en dólares de las llamadas al modelo (complementa lo que muestra LangSmith).

LangSmith calcula el costo solo si conoce la tarifa del modelo. Para no depender de eso,
este módulo cuenta los tokens de cada llamada (usage_metadata de LangChain) y los multiplica
por la tarifa pública de Anthropic. Se usa así:

    contador = ContadorCostos()
    agente.invoke(entrada, config={"callbacks": [contador]})
    print(contador.resumen())
"""
import threading

from langchain_core.callbacks import BaseCallbackHandler

# USD por millón de tokens: (entrada, salida, escritura de caché, lectura de caché).
# Tarifas de la API de Anthropic (octubre de 2026). Haiku 5.5: prompts de hasta 100K tokens.
TARIFAS = {
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
}


def _nombre(modelo: str) -> str:
    """'anthropic:claude-haiku-5-5' -> 'claude-haiku-5-5'."""
    return str(modelo or "").split(":")[-1]


def costo_usd(modelo: str, entrada: int, salida: int, cache_escritura: int = 0, cache_lectura: int = 0):
    """Costo de una llamada. `entrada` incluye los tokens de caché, como en usage_metadata.
    Devuelve None si el modelo no está en TARIFAS."""
    tarifa = TARIFAS.get(_nombre(modelo))
    if tarifa is None:
        return None
    p_in, p_out, p_write, p_read = tarifa
    base = max(0, entrada - cache_escritura - cache_lectura)
    return (base * p_in + salida * p_out + cache_escritura * p_write + cache_lectura * p_read) / 1_000_000


def uso_de_mensaje(mensaje) -> dict:
    """Tokens de un AIMessage de LangChain (ceros si no trae usage_metadata)."""
    uso = getattr(mensaje, "usage_metadata", None) or {}
    detalle = uso.get("input_token_details") or {}
    return {"entrada": int(uso.get("input_tokens") or 0), "salida": int(uso.get("output_tokens") or 0),
            "cache_escritura": int(detalle.get("cache_creation") or 0),
            "cache_lectura": int(detalle.get("cache_read") or 0)}


class ContadorCostos(BaseCallbackHandler):
    """Callback que acumula tokens y costo de TODAS las llamadas al modelo de una ejecución,
    incluidas las que hacen las herramientas por dentro (p. ej. la extracción estructurada)."""

    def __init__(self, modelo_por_defecto: str = ""):
        self.modelo_por_defecto = modelo_por_defecto
        self.llamadas: list[dict] = []
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs):
        for generaciones in response.generations:
            for g in generaciones:
                mensaje = getattr(g, "message", None)
                if mensaje is None:
                    continue
                uso = uso_de_mensaje(mensaje)
                meta = getattr(mensaje, "response_metadata", None) or {}
                modelo = meta.get("model_name") or meta.get("model") or self.modelo_por_defecto
                with self._lock:
                    self.llamadas.append({**uso, "modelo": _nombre(modelo),
                                          "costo_usd": costo_usd(modelo, **uso)})

    def resumen(self) -> dict:
        with self._lock:
            ll = list(self.llamadas)
        costos = [c["costo_usd"] for c in ll]
        return {
            "llamadas_modelo": len(ll),
            "tokens_entrada": sum(c["entrada"] for c in ll),
            "tokens_salida": sum(c["salida"] for c in ll),
            "tokens_cache_lectura": sum(c["cache_lectura"] for c in ll),
            "costo_usd": round(sum(x for x in costos if x is not None), 6) if ll else 0.0,
            "costo_completo": all(x is not None for x in costos),  # False si algún modelo no tiene tarifa
        }
