"""Tabla comparativa de configuraciones a partir de los experimentos de evaluar.py.

Uso:  python resumen_experimentos.py A-base B-mejorado [...]
Lee resultados/experimento_<prefijo>.json y escribe resultados/comparacion_configuraciones.md.
Si hay LANGSMITH_API_KEY, completa con LangSmith el costo y la latencia que falten, y si no hay
JSON local para un prefijo, toma las estadísticas del experimento más reciente con ese prefijo.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).parent
RESULTADOS = RAIZ / "resultados"
META = 85.0  # % de precisión exigido por el taller


def _media(stats: dict, clave: str) -> float | None:
    """Promedio de una clave de feedback_stats de LangSmith."""
    valor = (stats or {}).get(clave)
    if isinstance(valor, dict):
        valor = valor.get("avg", valor.get("mean"))
    return float(valor) if valor is not None else None


def _pct(valor: float | None) -> float | None:
    return round(100 * valor, 1) if valor is not None else None


def desde_langsmith(prefijo: str, local: dict = None) -> dict | None:
    """Estadísticas del experimento de LangSmith; completa `local` o crea un resumen mínimo."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return local
    for v in ("LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT"):
        os.environ.pop(v, None)
    try:
        from langsmith import Client

        client = Client()
        nombre = (local or {}).get("experimento")
        if nombre:
            proyecto = client.read_project(project_name=nombre, include_stats=True)
        else:
            candidatos = [p for p in client.list_projects(name_contains=prefijo, include_stats=True)
                          if (p.name or "").startswith(prefijo)]
            if not candidatos:
                return local
            proyecto = max(candidatos, key=lambda p: p.start_time)
    except Exception as e:  # noqa: BLE001  (LangSmith es opcional aquí)
        print(f"AVISO: no se pudo consultar LangSmith para {prefijo}: {e}")
        return local

    costo = float(proyecto.total_cost) if proyecto.total_cost is not None else None
    p50 = proyecto.latency_p50.total_seconds() if proyecto.latency_p50 else None
    if local:
        if local.get("costo_langsmith_usd") is None:
            local["costo_langsmith_usd"] = costo
        return local
    meta = (proyecto.extra or {}).get("metadata", {})
    fb = proyecto.feedback_stats or {}
    n = proyecto.run_count or 0
    costo_pp = _media(fb, "costo_usd")  # evaluador costo_usd de evaluar.py
    return {
        "prefijo": prefijo, "experimento": proyecto.name, "config": meta.get("config", "?"),
        "modelo": meta.get("modelo", "?"), "cambio": proyecto.description or "", "preguntas": n,
        "precision_uso_herramientas": _pct(_media(fb, "uso_herramientas")),
        "precision_respuesta_correcta": _pct(_media(fb, "respuesta_correcta")),
        "precision_combinada": None,
        "tokens_entrada": proyecto.prompt_tokens, "tokens_salida": proyecto.completion_tokens,
        "costo_total_usd": round(costo_pp * n, 6) if costo_pp is not None and n else None,
        "costo_por_pregunta_usd": costo_pp, "costo_por_respuesta_correcta_usd": None,
        "costo_langsmith_usd": costo, "latencia_p50_s": round(p50, 2) if p50 is not None else None,
        "latencia_p95_s": None,
    }


def cargar(prefijo: str) -> dict | None:
    archivo = RESULTADOS / f"experimento_{prefijo}.json"
    local = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else None
    return desde_langsmith(prefijo, local)


def _fmt(valor, formato: str = "{}", sufijo: str = "") -> str:
    return "—" if valor is None else formato.format(valor) + sufijo


def veredicto(exps: list[dict]) -> str:
    """Una línea: quién cumple el 85 % y cuál es más barata por respuesta correcta."""
    cumplen = [e for e in exps if (e.get("precision_respuesta_correcta") or 0) >= META]
    texto = (f"Cumplen el criterio de {META:.0f} % de respuestas correctas: "
             + (", ".join(f"{e['prefijo']} ({e['precision_respuesta_correcta']} %)" for e in cumplen) or "ninguna")
             + ".")
    por_correcta = [(e["costo_por_respuesta_correcta_usd"], e) for e in exps
                    if e.get("costo_por_respuesta_correcta_usd") is not None]
    if por_correcta:
        costo, mejor = min(por_correcta, key=lambda x: x[0])
        texto += f" La más barata por respuesta correcta es {mejor['prefijo']} (USD {costo:.6f})."
    if cumplen:
        elegida = min(cumplen, key=lambda e: e.get("costo_por_pregunta_usd") or float("inf"))
        texto += f" Recomendada: {elegida['prefijo']} (≥ {META:.0f} % al menor costo)."
    return texto


def tabla(exps: list[dict]) -> str:
    filas = [
        ("Experimento en LangSmith", lambda e: f"`{e.get('experimento') or e['prefijo']}`"),
        ("Configuración", lambda e: e.get("config", "?")),
        ("Modelo", lambda e: f"`{e.get('modelo', '?')}`"),
        ("Cambio", lambda e: e.get("cambio") or "—"),
        ("Preguntas", lambda e: _fmt(e.get("preguntas"))),
        ("Uso de herramientas", lambda e: _fmt(e.get("precision_uso_herramientas"), sufijo=" %")),
        ("Respuesta correcta", lambda e: _fmt(e.get("precision_respuesta_correcta"), sufijo=" %")),
        ("Ambas correctas", lambda e: _fmt(e.get("precision_combinada"), sufijo=" %")),
        ("Tokens entrada / salida", lambda e: f"{_fmt(e.get('tokens_entrada'), '{:,}')} / {_fmt(e.get('tokens_salida'), '{:,}')}"),
        ("Costo total (costos.py)", lambda e: _fmt(e.get("costo_total_usd"), "USD {:.6f}")),
        ("Costo por pregunta", lambda e: _fmt(e.get("costo_por_pregunta_usd"), "USD {:.6f}")),
        ("Costo por respuesta correcta", lambda e: _fmt(e.get("costo_por_respuesta_correcta_usd"), "USD {:.6f}")),
        ("Costo total según LangSmith", lambda e: _fmt(e.get("costo_langsmith_usd"), "USD {:.6f}")),
        ("Latencia p50", lambda e: _fmt(e.get("latencia_p50_s"), "{:.2f}", " s")),
        ("Latencia p95", lambda e: _fmt(e.get("latencia_p95_s"), "{:.2f}", " s")),
    ]
    lineas = ["| | " + " | ".join(e["prefijo"] for e in exps) + " |", "|---|" + "---|" * len(exps)]
    for nombre, f in filas:
        lineas.append(f"| {nombre} | " + " | ".join(str(f(e)).replace("|", "/") for e in exps) + " |")
    return "\n".join(lineas)


def generar(prefijos: list[str]) -> str:
    exps = []
    for p in prefijos:
        e = cargar(p)
        if e is None:
            print(f"AVISO: no hay resultados/experimento_{p}.json (corran: python evaluar.py \"{p}\").")
            continue
        exps.append(e)
    if not exps:
        sys.exit("No hay experimentos para comparar.")
    return ("# Comparación de configuraciones · Taller 1\n\n"
            "Conjunto de referencia (18 preguntas, `MODO_DATOS=muestra`). Costo calculado con `costos.py` "
            "(tarifas públicas de Anthropic); LangSmith puede no conocer la tarifa del modelo.\n\n"
            + tabla(exps) + "\n\n**Veredicto:** " + veredicto(exps) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python resumen_experimentos.py A-base B-mejorado [...]")
    md = generar(sys.argv[1:])
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "comparacion_configuraciones.md").write_text(md, encoding="utf-8")
    print(md)
    print("Guardado en resultados/comparacion_configuraciones.md")
