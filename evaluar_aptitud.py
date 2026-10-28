"""Dataset y evaluación de aptitud en LangSmith, sin juez de lenguaje."""
from __future__ import annotations

import argparse
import json
import math
import os

from langsmith import Client

from agente_aptitud import AgenteAptitud
from analizar_procesos import BASE, cargar_empresa
from pliegos import nombre_modelo

DATASET = "secop-taller1-aptitud"


def asegurar_dataset(cliente):
    """Crea los veinte ejemplos solo si el dataset aún no existe."""
    if cliente.has_dataset(dataset_name=DATASET):
        return cliente.read_dataset(dataset_name=DATASET)
    clave = json.loads((BASE / "datos/clave_aptitud.json").read_text(encoding="utf-8"))["procesos"]
    dataset = cliente.create_dataset(DATASET, description="20 procesos SECOP frente a la empresa multipropósito; pesos nominales.")
    cliente.create_examples(dataset_id=dataset.id,
        examples=[{"inputs": {"archivo": p["archivo"]}, "outputs": {
            "veredicto": p["veredicto_esperado"], "requisitos": p["requisitos_esperados"]}} for p in clave])
    return dataset


def veredicto_correcto(run, example):
    return {"key": "veredicto_correcto", "score":
            (run.outputs or {}).get("veredicto") == (example.outputs or {}).get("veredicto")}


def _coincide(actual, esperado):
    if isinstance(esperado, list):
        return isinstance(actual, list) and set(actual) == set(esperado)
    if isinstance(esperado, bool) or esperado is None:
        return type(actual) is type(esperado) and actual == esperado
    if isinstance(esperado, (int, float)):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(actual, esperado, rel_tol=0, abs_tol=1e-6))
    return actual == esperado


def extraccion_requisitos(run, example):
    esperados = (example.outputs or {}).get("requisitos", {})
    obtenidos = (run.outputs or {}).get("requisitos") or {}
    aciertos = sum(k in obtenidos and _coincide(obtenidos[k], v) for k, v in esperados.items())
    return {"key": "extraccion_requisitos", "score": aciertos / len(esperados) if esperados else 0}


def uso_herramientas(run, example):
    usadas = set((run.outputs or {}).get("herramientas", []))
    return {"key": "uso_herramientas", "score":
            {"extraer_requisitos", "verificar_requisitos"}.issubset(usadas)}


def objetivo(inputs):
    """Ejecuta el modelo elegido con MODELO sobre el PDF de un ejemplo."""
    ruta = (BASE / "datos" / inputs["archivo"]).resolve()
    if ruta.parent != (BASE / "datos").resolve() or ruta.suffix.lower() != ".pdf":
        raise ValueError("El ejemplo debe señalar un PDF de datos/.")
    return AgenteAptitud(ruta, cargar_empresa(), modelo=nombre_modelo()).analizar()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefijo", nargs="?", default="apt-haiku")
    args = parser.parse_args(argv)
    if not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("Falta ANTHROPIC_API_KEY.")
    if not os.getenv("LANGSMITH_API_KEY"):
        parser.error("Falta LANGSMITH_API_KEY.")
    cliente = Client()
    asegurar_dataset(cliente)
    return cliente.evaluate(objetivo, data=DATASET,
        evaluators=[veredicto_correcto, extraccion_requisitos, uso_herramientas],
        experiment_prefix=args.prefijo, max_concurrency=3,
        metadata={"modelo": nombre_modelo(), "empresa": "empresa_multiproposito.json", "smmlv_por_anio": False})


if __name__ == "__main__":
    main()
