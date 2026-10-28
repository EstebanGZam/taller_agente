"""Evalúa el agente en LangSmith con el conjunto de referencia del curso.

Uso:  CONFIG=A python evaluar.py "A-base"
      CONFIG=B python evaluar.py "B-mejorado"
Cada ejemplo registra en sus salidas los tokens, el costo (costos.py) y la latencia. Al final se
imprime y se guarda resultados/experimento_<prefijo>.json, que lee resumen_experimentos.py.
"""
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).parent
DATASET = "secop-taller1-referencia"
JUEZ_MODELO = os.getenv("JUEZ_MODELO", "anthropic:claude-haiku-5-5")


def limpiar_entorno_langsmith() -> list[str]:
    """LANGSMITH_AGENT_ID/ENVIRONMENT junto con LANGSMITH_PROJECT hacen que LangSmith no sepa a
    dónde enviar las trazas (y no registre tokens ni costo). Se quitan antes de crear el cliente."""
    quitadas = [v for v in ("LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT") if os.environ.pop(v, None)]
    if quitadas:
        print(f"AVISO: se ignoran {', '.join(quitadas)}; las trazas van a LANGSMITH_PROJECT.")
        try:
            from langsmith import utils as ls_utils
            ls_utils.get_env_var.cache_clear()  # la librería guarda las variables en caché
        except (ImportError, AttributeError):
            pass
    return quitadas


limpiar_entorno_langsmith()

from costos import ContadorCostos  # noqa: E402
import agente  # noqa: E402  (después de cargar .env)
from agente import preguntar  # noqa: E402

CASOS = json.loads((RAIZ / "datos" / "conjunto_referencia.json").read_text(encoding="utf-8"))
ID_POR_PREGUNTA = {c["pregunta"]: c["id"] for c in CASOS}


# 1. Lo que se evalúa: el agente, midiendo tokens, costo y latencia de cada pregunta.
def objetivo(inputs: dict) -> dict:
    contador = ContadorCostos(agente.MODELO)
    inicio = time.perf_counter()
    r = preguntar(inputs["pregunta"], callbacks=[contador])
    latencia = time.perf_counter() - inicio
    m = contador.resumen()
    return {**r, "metricas": {"llamadas_modelo": m["llamadas_modelo"], "tokens_entrada": m["tokens_entrada"],
                              "tokens_salida": m["tokens_salida"], "costo_usd": m["costo_usd"],
                              "costo_completo": m["costo_completo"], "latencia_s": round(latencia, 3)}}


# 2. Evaluador exacto: ¿usó las herramientas esperadas (y ninguna cuando no hacía falta)?
def uso_herramientas(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    esperadas, usadas = set(reference_outputs["herramientas"]), set(outputs["herramientas"])
    ok = usadas == set() if not esperadas else esperadas.issubset(usadas)
    return {"key": "uso_herramientas", "score": int(ok)}


# 3. Modelo como juez: ¿la respuesta coincide con la esperada?
_juez = None


def leer_veredicto(texto) -> int | None:
    """1 si el juez dice CORRECTA, 0 si dice INCORRECTA, None si no dice ninguna.
    Se busca la palabra completa: 'INCORRECTA' contiene 'CORRECTA' como subcadena."""
    m = re.search(r"\b(INCORRECTA|CORRECTA)\b", agente.texto_de(texto).upper())
    return None if m is None else int(m.group(1) == "CORRECTA")


def respuesta_correcta(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    global _juez
    if _juez is None:
        from langchain.chat_models import init_chat_model
        _juez = init_chat_model(JUEZ_MODELO)  # sin temperature: Haiku 5.5 la rechaza (error 400)
    veredicto = _juez.invoke(
        "Eres un evaluador estricto. Compara la RESPUESTA con la RESPUESTA ESPERADA a la PREGUNTA. "
        "Las cifras deben coincidir (se aceptan diferencias de formato y redondeo). No penalices "
        "información adicional si es correcta y no contradice la esperada. "
        "Responde solo con una palabra: CORRECTA o INCORRECTA.\n\n"
        f"PREGUNTA: {inputs['pregunta']}\nRESPUESTA ESPERADA: {reference_outputs['respuesta']}\n"
        f"RESPUESTA: {outputs['respuesta']}"
    ).content
    puntaje = leer_veredicto(veredicto)
    return {"key": "respuesta_correcta", "score": puntaje or 0,
            "comment": None if puntaje is not None else f"Veredicto ilegible: {agente.texto_de(veredicto)[:200]}"}


# 4. Costo de cada ejemplo como columna de retroalimentación (visible en la vista de comparación).
def costo_usd(outputs: dict) -> dict:
    return {"key": "costo_usd", "score": (outputs.get("metricas") or {}).get("costo_usd")}


def percentil(valores: list[float], p: float) -> float | None:
    """Percentil con interpolación lineal (p entre 0 y 100)."""
    if not valores:
        return None
    v = sorted(valores)
    k = (len(v) - 1) * p / 100
    i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


def resumir(filas: list[dict], prefijo: str, config: str, modelo: str, experimento: str = None,
            costo_langsmith: float = None) -> dict:
    """Agrega las filas por pregunta en las métricas del experimento."""
    n = len(filas)

    def pct(clave):
        return round(100 * sum(f[clave] or 0 for f in filas) / n, 1) if n else 0.0

    latencias = [f["latencia_s"] for f in filas if f.get("latencia_s") is not None]
    costo = sum(f["costo_usd"] or 0 for f in filas)
    ambas = sum(1 for f in filas if f["uso_herramientas"] == 1 and f["respuesta_correcta"] == 1)
    correctas = sum(f["respuesta_correcta"] or 0 for f in filas)
    p50, p95 = percentil(latencias, 50), percentil(latencias, 95)
    return {
        "prefijo": prefijo, "experimento": experimento, "config": config, "modelo": modelo,
        "cambio": agente.CAMBIOS.get(config, ""), "preguntas": n,
        "precision_uso_herramientas": pct("uso_herramientas"),
        "precision_respuesta_correcta": pct("respuesta_correcta"),
        "precision_combinada": round(100 * ambas / n, 1) if n else 0.0,  # ambas métricas en 1
        "tokens_entrada": sum(f["tokens_entrada"] or 0 for f in filas),
        "tokens_salida": sum(f["tokens_salida"] or 0 for f in filas),
        "costo_total_usd": round(costo, 6),
        "costo_por_pregunta_usd": round(costo / n, 6) if n else 0.0,
        "costo_por_respuesta_correcta_usd": round(costo / correctas, 6) if correctas else None,
        "costo_langsmith_usd": costo_langsmith,
        "latencia_p50_s": round(p50, 2) if p50 is not None else None,
        "latencia_p95_s": round(p95, 2) if p95 is not None else None,
        "errores": sum(1 for f in filas if f.get("error")),
        "por_pregunta": sorted(filas, key=lambda f: f.get("id") or ""),
    }


def fila_de(resultado) -> dict:
    """Una fila por ejemplo a partir de un ExperimentResultRow de LangSmith."""
    run, ejemplo = resultado["run"], resultado["example"]
    salidas = run.outputs or {}
    m = salidas.get("metricas") or {}
    puntajes = {r.key: r.score for r in resultado["evaluation_results"]["results"]}
    pregunta = (ejemplo.inputs or {}).get("pregunta", "")
    latencia = m.get("latencia_s")
    if latencia is None and run.end_time and run.start_time:
        latencia = (run.end_time - run.start_time).total_seconds()
    return {"id": ID_POR_PREGUNTA.get(pregunta), "pregunta": pregunta,
            "herramientas": salidas.get("herramientas"), "respuesta": salidas.get("respuesta"),
            "uso_herramientas": puntajes.get("uso_herramientas"),
            "respuesta_correcta": puntajes.get("respuesta_correcta"),
            "tokens_entrada": m.get("tokens_entrada"), "tokens_salida": m.get("tokens_salida"),
            "costo_usd": m.get("costo_usd"), "latencia_s": latencia, "error": run.error}


def imprimir(r: dict) -> None:
    print(f"\nExperimento {r['experimento'] or r['prefijo']} · CONFIG={r['config']} · {r['modelo']}")
    print(f"  Uso de herramientas:  {r['precision_uso_herramientas']} %")
    print(f"  Respuesta correcta:   {r['precision_respuesta_correcta']} %")
    print(f"  Ambas correctas:      {r['precision_combinada']} %")
    print(f"  Tokens entrada/salida: {r['tokens_entrada']:,} / {r['tokens_salida']:,}")
    print(f"  Costo total (costos.py): USD {r['costo_total_usd']:.6f} · por pregunta USD {r['costo_por_pregunta_usd']:.6f}")
    print(f"  Costo según LangSmith:   {r['costo_langsmith_usd'] if r['costo_langsmith_usd'] is not None else 'no disponible'}")
    print(f"  Latencia p50 / p95:   {r['latencia_p50_s']} s / {r['latencia_p95_s']} s")
    for f in r["por_pregunta"]:
        if f["uso_herramientas"] != 1 or f["respuesta_correcta"] != 1 or f["error"]:
            print(f"  ✗ {f['id']}: herramientas={f['herramientas']} uso={f['uso_herramientas']} "
                  f"correcta={f['respuesta_correcta']}{' error=' + str(f['error'])[:80] if f['error'] else ''}")


def guardar(r: dict) -> Path:
    destino = RAIZ / "resultados" / f"experimento_{r['prefijo']}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return destino


def crear_dataset(client) -> None:
    """Sube el conjunto de referencia a LangSmith (solo la primera vez)."""
    if client.has_dataset(dataset_name=DATASET):
        return
    ds = client.create_dataset(DATASET, description="Preguntas de referencia del Taller 1 (muestra SECOP)")
    client.create_examples(
        dataset_id=ds.id,
        examples=[{"inputs": {"pregunta": c["pregunta"]},
                   "outputs": {"respuesta": c["respuesta_esperada"], "herramientas": c["herramientas_esperadas"]}}
                  for c in CASOS],
    )


def costo_segun_langsmith(client, experimento: str) -> float | None:
    """total_cost del proyecto del experimento; None si LangSmith no conoce la tarifa del modelo."""
    try:
        client.flush()
        proyecto = client.read_project(project_name=experimento, include_stats=True)
        return float(proyecto.total_cost) if proyecto.total_cost is not None else None
    except Exception as e:  # noqa: BLE001  (las estadísticas son opcionales)
        print(f"AVISO: no se pudo leer el costo de LangSmith: {e}")
        return None


def main(prefijo: str) -> dict:
    faltan = [v for v in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY") if not os.getenv(v)]
    if faltan:
        sys.exit(f"Faltan variables de entorno: {', '.join(faltan)} (ver .env.example).")
    from langsmith import Client

    client = Client()
    crear_dataset(client)
    config, modelo = agente.CONFIG, agente.MODELO
    resultados = client.evaluate(
        objetivo, data=DATASET, evaluators=[uso_herramientas, respuesta_correcta, costo_usd],
        experiment_prefix=prefijo, max_concurrency=2,
        description=f"CONFIG={config} · {agente.CAMBIOS.get(config, '')}",
        metadata={"config": config, "modelo": modelo, "juez": JUEZ_MODELO, "modo_datos": os.getenv("MODO_DATOS", "muestra")},
    )
    filas = [fila_de(x) for x in resultados]
    r = resumir(filas, prefijo, config, modelo, resultados.experiment_name,
                costo_segun_langsmith(client, resultados.experiment_name))
    imprimir(r)
    print(f"\nGuardado en {guardar(r).relative_to(RAIZ)}")
    print("Revisen el experimento en LangSmith > Datasets & Experiments.")
    return r


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"configuracion-{agente.CONFIG}")
