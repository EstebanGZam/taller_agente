"""Agente analista del SECOP - punto de partida del Taller 1.

Uso:  python agente.py "¿Qué licitaciones hay en la muestra?"
"""
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

import herramientas as h

load_dotenv()


# TODO del equipo: mejoren estas descripciones. El modelo decide qué herramienta usar
# leyendo el docstring, así que una buena descripción mejora la precisión y baja el costo.
@tool
def buscar_procesos(entidad: str = None, departamento: str = None, ciudad: str = None,
                    modalidad: str = None, texto: str = None,
                    precio_min: float = None, precio_max: float = None,
                    fecha_desde: str = None, fecha_hasta: str = None,
                    limite: int = 20) -> dict:
    """Busca procesos de contratación pública en el SECOP II. Filtros opcionales: entidad,
    departamento, ciudad, modalidad (p. ej. 'licitaci', 'directa', 'abreviada'), texto en la
    descripción, rango de presupuesto en pesos y rango de fechas de publicación (AAAA-MM-DD).
    Devuelve los procesos y un resumen con cantidad, presupuesto total y promedio."""
    return h.buscar_procesos(entidad, departamento, ciudad, modalidad, texto,
                             precio_min, precio_max, fecha_desde, fecha_hasta, limite)


@tool
def consultar_empresa() -> dict:
    """Devuelve el perfil de NUESTRA empresa: RUP, códigos UNSPSC, estados financieros y
    contratos de experiencia."""
    return h.consultar_empresa()


@tool
def calcular_indicadores(presupuesto_proceso: float = None) -> dict:
    """Calcula los indicadores financieros de NUESTRA empresa: liquidez, endeudamiento,
    cobertura de intereses, capital de trabajo y rentabilidades. Si se da el presupuesto de
    un proceso, calcula qué porcentaje de él cubre nuestro capital de trabajo."""
    return h.calcular_indicadores(presupuesto_proceso)


PROMPT = """Eres un analista de contratación pública colombiana que asesora a una empresa.
Responde en español, de forma breve y precisa.
- Usa las herramientas solo cuando necesites datos; las preguntas generales respóndelas directamente.
- Nunca inventes cifras: si un dato no está o parece inconsistente (por ejemplo, plazo de 0 días
  o presupuesto de $0), dilo explícitamente.
- Cita la referencia del proceso cuando hables de uno."""

MODELO = os.getenv("MODELO", "anthropic:claude-haiku-5-5")
agente = create_agent(MODELO, tools=[buscar_procesos, consultar_empresa, calcular_indicadores],
                      system_prompt=PROMPT)


def preguntar(pregunta: str) -> dict:
    """Ejecuta el agente y devuelve la respuesta final y las herramientas que usó."""
    resultado = agente.invoke({"messages": [{"role": "user", "content": pregunta}]})
    usadas = [tc["name"] for m in resultado["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
    return {"respuesta": resultado["messages"][-1].content, "herramientas": usadas}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "¿Cuál es nuestro índice de liquidez?"
    r = preguntar(q)
    print(r["respuesta"])
    print("\nHerramientas usadas:", r["herramientas"] or "ninguna")
