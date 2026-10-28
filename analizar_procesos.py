"""Análisis por lotes y registro de precisión, costo y latencia."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

from agente_aptitud import AgenteAptitud
from pliegos import nombre_modelo
from validacion import validar_empresa, verificar

BASE = Path(__file__).resolve().parent
EMPRESA_PREDETERMINADA = BASE / "empresa_multiproposito.json"
COLUMNAS = ["n", "archivo", "referencia", "entidad", "modalidad", "presupuesto_oficial",
            "veredicto", "veredicto_esperado", "acierto", "motivo", "herramientas",
            "llamadas_modelo", "tokens_entrada", "tokens_salida", "costo_usd", "latencia_s", "modelo"]


def cargar_empresa(ruta=EMPRESA_PREDETERMINADA):
    """Carga y valida el perfil antes de consumir llamadas al modelo."""
    empresa = json.loads(Path(ruta).read_text(encoding="utf-8"))
    errores = validar_empresa(empresa)
    if errores:
        raise ValueError("Empresa inválida: " + " ".join(errores))
    return empresa


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def deduplicar(rutas):
    """Conserva una ruta por contenido, prefiriendo el nombre sin sufijo de copia."""
    unicos = {}
    for ruta in sorted(map(Path, rutas), key=lambda p: ("(" in p.stem, str(p))):
        unicos.setdefault(huella(ruta), ruta)
    return sorted(unicos.values(), key=lambda p: (p.name, str(p)))


def clave_por_contenido(empresa):
    """La clave solo mide la empresa de referencia y PDF con contenido conocido."""
    if empresa != cargar_empresa():
        return {}
    clave = json.loads((BASE / "datos/clave_aptitud.json").read_text(encoding="utf-8"))
    return {huella(BASE / "datos" / p["archivo"]): p for p in clave["procesos"]}


def _repositorio():
    try:
        return subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=BASE,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "No disponible"


def resumir(resultados, filas, modelo):
    """Calcula métricas observadas sin imputar precisión a otra empresa."""
    evaluadas = [f for f in filas if f["acierto"] is not None]
    aciertos = sum(f["acierto"] for f in evaluadas)
    costos = [r["metricas"].get("costo_usd") for r in resultados]
    costo = round(sum(c or 0 for c in costos), 6)
    return {
        "nombre": os.getenv("EQUIPO") or os.getenv("LANGSMITH_PROJECT", "Equipo sin nombre"),
        "repositorio": _repositorio(), "procesos": len(resultados),
        "conteos": dict(Counter(r["veredicto"] for r in resultados)),
        "aciertos": aciertos if evaluadas else None, "evaluados": len(evaluadas),
        "precision_pct": round(100 * aciertos / len(evaluadas), 2) if evaluadas else None,
        "costo_total_usd": costo, "costo_promedio_por_proceso": costo / len(resultados) if resultados else 0,
        "costo_completo": all(r["metricas"].get("costo_completo", False) for r in resultados),
        "latencia_p50_s": statistics.median(r["metricas"]["latencia_s"] for r in resultados) if resultados else 0,
        "tokens_entrada": sum(r["metricas"]["tokens_entrada"] for r in resultados),
        "tokens_salida": sum(r["metricas"]["tokens_salida"] for r in resultados),
        "modelo": nombre_modelo(modelo),
    }


def analizar_lote(rutas, empresa, modelo=None, trabajadores=3, estricto=False):
    """Analiza archivos únicos y conserva los agentes para preguntas posteriores."""
    errores = validar_empresa(empresa)
    if errores:
        raise ValueError("Empresa inválida: " + " ".join(errores))
    if trabajadores < 1:
        raise ValueError("El número de trabajadores debe ser mayor que cero.")
    rutas = deduplicar(rutas)
    if not rutas:
        raise ValueError("No se encontraron PDF para analizar.")
    clave = clave_por_contenido(empresa)
    modelo = nombre_modelo(modelo)

    def uno(ruta):
        agente = None
        try:
            agente = AgenteAptitud(ruta, empresa, modelo=modelo, smmlv_por_anio=estricto)
            return agente, agente.analizar()
        except Exception as exc:
            # El error de un documento no impide analizar los demás ni se cuenta como acierto.
            return agente, {**verificar(None, empresa), "archivo": ruta.name,
                "avisos": [f"Error de análisis ({type(exc).__name__}); revise el archivo y la configuración."],
                "explicacion": "", "herramientas": [], "error": True, "run_id": None,
                "metricas": {"llamadas_modelo": 0, "tokens_entrada": 0, "tokens_salida": 0,
                             "costo_usd": 0, "latencia_s": 0, "costo_completo": False}}

    with ThreadPoolExecutor(max_workers=trabajadores) as pool:
        pares = list(pool.map(uno, rutas))
    resultados, agentes, filas = [], {}, []
    for n, (ruta, (agente, res)) in enumerate(zip(rutas, pares), 1):
        esperado = clave.get(huella(ruta))
        res["n"] = n
        res["veredicto_esperado"] = esperado["veredicto_esperado"] if esperado else None
        res["acierto"] = (not res.get("error", False) and res["veredicto"] == esperado["veredicto_esperado"]
                          if esperado else None)
        res["modelo"] = modelo
        resultados.append(res)
        if agente is not None and not res.get("error"):
            agentes[str(n)] = agente
        fila = {k: res.get(k) for k in COLUMNAS}
        fila.update({k: res["metricas"].get(k) for k in (
            "llamadas_modelo", "tokens_entrada", "tokens_salida", "costo_usd", "latencia_s")})
        fila["motivo"] = res["resumen"]
        fila["herramientas"] = ", ".join(res["herramientas"])
        filas.append(fila)
    return {"resultados": resultados, "agentes": agentes, "filas": filas,
            "resumen": resumir(resultados, filas, modelo)}


def texto_resumen(resumen):
    """Registro del equipo y métricas del lote en texto plano."""
    precision = (f"{resumen['aciertos']}/{resumen['evaluados']} · precisión {resumen['precision_pct']:.2f}%"
                 if resumen["evaluados"] else "Precisión: sin clave aplicable")
    return (f"Nombre: {resumen['nombre']}\nRepositorio: {resumen['repositorio']}\n"
            f"Procesos: {resumen['procesos']} · {precision}\n"
            f"Costo total USD: {resumen['costo_total_usd']:.6f}\n"
            f"Costo promedio por proceso USD: {resumen['costo_promedio_por_proceso']:.6f}\n"
            f"Costo completo: {'sí' if resumen['costo_completo'] else 'no (hay costos sin determinar)'}\n"
            f"Latencia p50: {resumen['latencia_p50_s']:.3f} s\n"
            f"Tokens: entrada {resumen['tokens_entrada']}, salida {resumen['tokens_salida']}\n"
            f"Modelo: {resumen['modelo']}")


def escribir_resultados(lote, salida):
    """Escribe las columnas del contrato y un resumen en Markdown y JSON."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    csv_path = salida / "aptitud_procesos.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(lote["filas"])
    contenido = {k: lote[k] for k in ("resumen", "resultados", "filas")}
    (salida / "aptitud_procesos.json").write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def celda(valor):
        return escape(str(valor if valor is not None else "")).replace("|", "\\|").replace("\n", " ")

    tabla = ["| " + " | ".join(COLUMNAS) + " |", "| " + " | ".join(["---"] * len(COLUMNAS)) + " |"]
    tabla.extend("| " + " | ".join(celda(f[c]) for c in COLUMNAS) + " |" for f in lote["filas"])
    (salida / "aptitud_procesos.md").write_text(
        "# Aptitud de los procesos\n\n" + texto_resumen(lote["resumen"]) + "\n\n" + "\n".join(tabla) + "\n",
        encoding="utf-8")
    return csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empresa", type=Path, default=EMPRESA_PREDETERMINADA)
    parser.add_argument("--modelo", default=None)
    parser.add_argument("--pdfs", type=Path, default=BASE / "datos")
    parser.add_argument("--trabajadores", type=int, default=3)
    parser.add_argument("--estricto", action="store_true")
    parser.add_argument("--salida", type=Path, default=BASE / "resultados")
    args = parser.parse_args(argv)
    if not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("Falta ANTHROPIC_API_KEY; configúrala en el entorno o en .env.")
    try:
        lote = analizar_lote(args.pdfs.glob("*.pdf"), cargar_empresa(args.empresa), args.modelo,
                             args.trabajadores, args.estricto)
        escribir_resultados(lote, args.salida)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(texto_resumen(lote["resumen"]))
    return lote


if __name__ == "__main__":
    main()
