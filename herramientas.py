"""Herramientas del agente analista del SECOP (Taller 1).

Funciones de Python puras: se pueden probar sin modelo de lenguaje.
agente.py las envuelve como herramientas de LangChain.
"""
import json
import os
import unicodedata
from pathlib import Path

import requests

DATOS = Path(__file__).parent / "datos"
URL_API = "https://www.datos.gov.co/resource/p6dx-8zbt.json"  # SECOP II - Procesos de Contratación
CAMPOS = [
    "entidad", "departamento_entidad", "ciudad_entidad", "nombre_de_la_unidad_de",
    "ciudad_de_la_unidad_de", "id_del_proceso", "referencia_del_proceso",
    "descripci_n_del_procedimiento", "modalidad_de_contratacion", "tipo_de_contrato",
    "precio_base", "duracion", "unidad_de_duracion", "fecha_de_publicacion_del",
    "estado_resumen", "nombre_del_proveedor", "valor_total_adjudicacion",
    "respuestas_al_procedimiento", "urlproceso",
]


def _norm(texto) -> str:
    """Minúsculas y sin tildes, para comparar textos."""
    texto = unicodedata.normalize("NFKD", str(texto or ""))
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()


def _num(valor) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return 0.0


def _resumen(procesos: list[dict]) -> dict:
    precios = [_num(p.get("precio_base")) for p in procesos]
    con_precio = [x for x in precios if x > 0]
    return {
        "cantidad": len(procesos),
        "presupuesto_total": sum(precios),
        "presupuesto_promedio": round(sum(precios) / len(precios), 2) if precios else 0,
        "procesos_con_presupuesto_cero": len(precios) - len(con_precio),
    }


def _filtrar_muestra(entidad, departamento, ciudad, modalidad, texto,
                     precio_min, precio_max, fecha_desde, fecha_hasta):
    procesos = json.loads((DATOS / "secop_muestra.json").read_text(encoding="utf-8"))
    res = []
    for p in procesos:
        if entidad and _norm(entidad) not in _norm(p.get("entidad")):
            continue
        if departamento and _norm(departamento) not in _norm(p.get("departamento_entidad")):
            continue
        if ciudad and _norm(ciudad) not in (_norm(p.get("ciudad_entidad")) + " " + _norm(p.get("ciudad_de_la_unidad_de"))):
            continue
        if modalidad and _norm(modalidad) not in _norm(p.get("modalidad_de_contratacion")):
            continue
        if texto and _norm(texto) not in _norm(p.get("descripci_n_del_procedimiento")):
            continue
        precio = _num(p.get("precio_base"))
        if precio_min is not None and precio < precio_min:
            continue
        if precio_max is not None and precio > precio_max:
            continue
        fecha = (p.get("fecha_de_publicacion_del") or "")[:10]
        if fecha_desde and fecha < fecha_desde:
            continue
        if fecha_hasta and fecha > fecha_hasta:
            continue
        res.append(p)
    return res


def _consultar_api(entidad, departamento, ciudad, modalidad, texto,
                   precio_min, precio_max, fecha_desde, fecha_hasta, limite):
    def q(v):  # escapa comillas simples para SoQL
        return str(v).replace("'", "''").upper()

    cond = []
    if entidad:
        cond.append(f"upper(entidad) like '%{q(entidad)}%'")
    if departamento:
        cond.append(f"upper(departamento_entidad) like '%{q(departamento)}%'")
    if ciudad:
        cond.append(f"(upper(ciudad_entidad) like '%{q(ciudad)}%' OR upper(ciudad_de_la_unidad_de) like '%{q(ciudad)}%')")
    if modalidad:
        cond.append(f"upper(modalidad_de_contratacion) like '%{q(modalidad)}%'")
    if texto:
        cond.append(f"upper(descripci_n_del_procedimiento) like '%{q(texto)}%'")
    if precio_min is not None:
        cond.append(f"precio_base >= {float(precio_min)}")
    if precio_max is not None:
        cond.append(f"precio_base <= {float(precio_max)}")
    if fecha_desde:
        cond.append(f"fecha_de_publicacion_del >= '{fecha_desde}T00:00:00'")
    if fecha_hasta:
        cond.append(f"fecha_de_publicacion_del <= '{fecha_hasta}T23:59:59'")
    params = {"$select": ",".join(CAMPOS), "$limit": limite, "$order": "fecha_de_publicacion_del DESC"}
    if cond:
        params["$where"] = " AND ".join(cond)
    headers = {}
    if os.getenv("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    r = requests.get(URL_API, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def buscar_procesos(entidad: str = None, departamento: str = None, ciudad: str = None,
                    modalidad: str = None, texto: str = None,
                    precio_min: float = None, precio_max: float = None,
                    fecha_desde: str = None, fecha_hasta: str = None,
                    limite: int = 20) -> dict:
    """Busca procesos de contratación del SECOP II.

    Usa la muestra local (MODO_DATOS=muestra, por defecto) o la API en vivo (MODO_DATOS=api).
    Las fechas van en formato AAAA-MM-DD. Devuelve los procesos y un resumen con cantidad,
    presupuesto total y promedio.
    """
    limite = max(1, min(int(limite or 20), 50))
    args = (entidad, departamento, ciudad, modalidad, texto, precio_min, precio_max, fecha_desde, fecha_hasta)
    if os.getenv("MODO_DATOS", "muestra") == "api":
        try:
            procesos = _consultar_api(*args, limite)
        except requests.RequestException as e:
            return {"error": f"No se pudo consultar la API del SECOP: {e}"}
    else:
        procesos = _filtrar_muestra(*args)[:limite]
    return {"resumen": _resumen(procesos), "procesos": procesos}


def consultar_empresa() -> dict:
    """Devuelve el perfil de la empresa: RUP, códigos UNSPSC, estados financieros y experiencia."""
    return json.loads((DATOS / "empresa.json").read_text(encoding="utf-8"))


def calcular_indicadores(presupuesto_proceso: float = None) -> dict:
    """Calcula los indicadores financieros y organizacionales de la empresa (como en el RUP).

    Si se indica el presupuesto de un proceso, también calcula qué porcentaje de ese
    presupuesto representa el capital de trabajo de la empresa.
    """
    ef = consultar_empresa()["estados_financieros"]
    ind = {
        "indice_liquidez": round(ef["activo_corriente"] / ef["pasivo_corriente"], 2),
        "indice_endeudamiento_pct": round(100 * ef["pasivo_total"] / ef["activo_total"], 1),
        "razon_cobertura_intereses": round(ef["utilidad_operacional"] / ef["gastos_de_intereses"], 2) if ef["gastos_de_intereses"] else "indeterminado",
        "capital_de_trabajo": ef["activo_corriente"] - ef["pasivo_corriente"],
        "rentabilidad_patrimonio": round(ef["utilidad_operacional"] / ef["patrimonio"], 3),
        "rentabilidad_activo": round(ef["utilidad_operacional"] / ef["activo_total"], 3),
        "corte_estados_financieros": ef["corte"],
    }
    if presupuesto_proceso:
        ind["capital_trabajo_pct_del_presupuesto"] = round(100 * ind["capital_de_trabajo"] / float(presupuesto_proceso), 1)
    return ind


if __name__ == "__main__":
    # Prueba rápida sin modelo de lenguaje
    print(calcular_indicadores())
    print(buscar_procesos(modalidad="licitaci")["resumen"])
