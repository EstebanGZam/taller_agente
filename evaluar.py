"""Evalúa el agente en LangSmith con el conjunto de referencia del curso.

Uso:  python evaluar.py "A-haiku-prompt-base"
Cambien MODELO o el prompt en .env/agente.py, ejecuten de nuevo con otro nombre y comparen
los dos experimentos en LangSmith (precisión, costo y latencia).
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import Client

load_dotenv()
from agente import preguntar  # noqa: E402  (después de cargar .env)

DATASET = "secop-taller1-referencia"
client = Client()

# 1. Crear el dataset en LangSmith (solo la primera vez)
if not client.has_dataset(dataset_name=DATASET):
    casos = json.loads((Path(__file__).parent / "datos" / "conjunto_referencia.json").read_text(encoding="utf-8"))
    ds = client.create_dataset(DATASET, description="Preguntas de referencia del Taller 1 (muestra SECOP)")
    client.create_examples(
        dataset_id=ds.id,
        examples=[{"inputs": {"pregunta": c["pregunta"]},
                   "outputs": {"respuesta": c["respuesta_esperada"], "herramientas": c["herramientas_esperadas"]}}
                  for c in casos],
    )


# 2. Lo que se evalúa
def objetivo(inputs: dict) -> dict:
    return preguntar(inputs["pregunta"])


# 3. Evaluador exacto: ¿usó las herramientas esperadas (y ninguna cuando no hacía falta)?
def uso_herramientas(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    esperadas, usadas = set(reference_outputs["herramientas"]), set(outputs["herramientas"])
    ok = usadas == set() if not esperadas else esperadas.issubset(usadas)
    return {"key": "uso_herramientas", "score": int(ok)}


# 4. Modelo como juez: ¿la respuesta coincide con la esperada?
juez = init_chat_model("anthropic:claude-haiku-5-5", temperature=0)


def respuesta_correcta(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    veredicto = juez.invoke(
        "Eres un evaluador estricto. Compara la RESPUESTA con la RESPUESTA ESPERADA a la PREGUNTA. "
        "Las cifras deben coincidir (se aceptan diferencias de formato y redondeo). "
        "Responde solo CORRECTA o INCORRECTA.\n\n"
        f"PREGUNTA: {inputs['pregunta']}\nRESPUESTA ESPERADA: {reference_outputs['respuesta']}\n"
        f"RESPUESTA: {outputs['respuesta']}"
    ).content
    return {"key": "respuesta_correcta", "score": int("INCORRECTA" not in str(veredicto).upper())}


if __name__ == "__main__":
    nombre = sys.argv[1] if len(sys.argv) > 1 else "configuracion-A"
    client.evaluate(objetivo, data=DATASET, evaluators=[uso_herramientas, respuesta_correcta],
                    experiment_prefix=nombre, max_concurrency=2)
    print("Listo. Revisen el experimento en LangSmith > Datasets & Experiments.")
