"""Interfaz Gradio para analizar los procesos y consultar cada resultado."""
from __future__ import annotations

import os
import tempfile
from html import escape
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

from analizar_procesos import (BASE, EMPRESA_PREDETERMINADA, analizar_lote,
                               cargar_empresa, escribir_resultados)

COLORES = {"APTA": "#167042", "NO APTA": "#b02f36", "NO APLICA": "#516579",
           "REVISAR MANUALMENTE": "#926300"}
EMOJIS = {"APTA": "✅", "NO APTA": "❌", "NO APLICA": "➖", "REVISAR MANUALMENTE": "⚠️"}
ICONOS = {True: "✅", False: "❌", None: "⚠️"}

# Mesa de revisión: azul institucional, papel blanco y resultados con semáforo.
CSS = """
.gradio-container {max-width: 1440px !important; font-family: Arial, sans-serif;}
#portada {border-left: 6px solid #20547a; padding: 8px 20px; background: #edf4f8;}
#portada h1 {font-size: 30px; color: #173b57;}
#portada p {max-width: 76ch; color: #344d61;}
.lista-requisitos {width: 100%; border-collapse: collapse; margin: 16px 0;}
.lista-requisitos th, .lista-requisitos td {padding: 10px; text-align: left;
 border-bottom: 1px solid #cbd7df; overflow-wrap: anywhere;}
.lista-requisitos th {background: #edf4f8; color: #173b57;}
.detalle-proceso {padding: 12px 18px; border-left: 5px solid #20547a;}
.detalle-proceso pre {white-space: pre-wrap; font-family: inherit;}
"""


def _ruta(archivo):
    return Path(archivo if isinstance(archivo, (str, Path)) else archivo.name)


def _html_veredicto(resultado):
    """Escapa todos los valores externos antes de construir el detalle."""
    def e(valor):
        return escape(str(valor if valor is not None else "—"), quote=True)

    veredicto = resultado["veredicto"]
    color = COLORES.get(veredicto, "#516579")
    filas = "".join(
        f"<tr><td>{e(c['requisito'])}</td><td>{e(c['exigido'])}</td>"
        f"<td>{e(c['empresa'])}</td><td>{ICONOS.get(c['cumple'], '⚠️')} "
        f"{e(c.get('detalle', ''))}</td></tr>" for c in resultado["checks"])
    tabla = ("<table class='lista-requisitos'><thead><tr><th>Requisito</th><th>Exigido</th>"
             "<th>Empresa</th><th>Cumple / detalle</th></tr></thead><tbody>" + filas + "</tbody></table>") if filas else ""
    avisos = "".join("<li>" + e(a) + "</li>" for a in resultado.get("avisos", []))
    enlace = ""
    # Solo se admiten URLs de LangSmith construidas por su cliente.
    url = resultado.get("trace_url")
    from urllib.parse import urlparse
    if url and urlparse(url).scheme == "https" and urlparse(url).hostname == "smith.langchain.com":
        enlace = f"<p><a href='{e(url)}' target='_blank' rel='noopener noreferrer'>Ver traza en LangSmith</a></p>"
    elif resultado.get("run_id"):
        enlace = "<p>Identificador de traza: <code>" + e(resultado["run_id"]) + "</code></p>"
    return (f"<section class='detalle-proceso' style='border-color:{color}'>"
            f"<h2 style='color:{color}'>{EMOJIS.get(veredicto, '⚠️')} {e(veredicto)}</h2>"
            f"<p><strong>{e(resultado.get('referencia'))}</strong> | {e(resultado.get('entidad'))}</p>"
            f"<p>{e(resultado['resumen'])}</p>{tabla}"
            f"{'<ul>' + avisos + '</ul>' if avisos else ''}"
            f"<pre>{e(resultado.get('explicacion', ''))}</pre>{enlace}</section>")


def analizar(pdfs=None, json_empresa=None, modelo="claude-haiku-5-5", estricto=False,
             progreso=gr.Progress()):
    """Analiza los archivos cargados o, sin cargas, los veinte del repositorio."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise gr.Error("Falta ANTHROPIC_API_KEY. Configúrala en el entorno o en .env antes de analizar.")
    try:
        empresa = cargar_empresa(_ruta(json_empresa) if json_empresa else EMPRESA_PREDETERMINADA)
        rutas = [_ruta(p) for p in pdfs] if pdfs else list((BASE / "datos").glob("*.pdf"))
        if progreso is not None:
            progreso(0, desc="Analizando procesos…")
        lote = analizar_lote(rutas, empresa, modelo, trabajadores=3, estricto=bool(estricto))
    except (ValueError, OSError) as exc:
        raise gr.Error("No se pudo iniciar el análisis: " + escape(str(exc))) from exc
    carpeta = Path(tempfile.mkdtemp(prefix="aptitud-"))
    descarga = str(escribir_resultados(lote, carpeta))
    resultados = {str(r["n"]): r for r in lote["resultados"]}
    historiales = {k: [{"role": "assistant", "content": escape(r["explicacion"])}]
                   if r["explicacion"] else [] for k, r in resultados.items()}
    filas = [[r["n"], escape(r["archivo"]), escape(r.get("referencia") or ""),
              escape(r.get("entidad") or ""), EMOJIS[r["veredicto"]] + " " + r["veredicto"],
              escape(r["resumen"]), r["metricas"]["costo_usd"],
              r["metricas"]["tokens_entrada"] + r["metricas"]["tokens_salida"],
              r["metricas"]["latencia_s"]] for r in lote["resultados"]]
    resumen = lote["resumen"]
    conteos = " | ".join(f"{EMOJIS[v]} {v}: {resumen['conteos'].get(v, 0)}" for v in EMOJIS)
    texto = f"{len(filas)} procesos únicos · {conteos}\n\nCosto total del análisis: USD {resumen['costo_total_usd']:.6f}"
    if not resumen["costo_completo"]:
        texto += " (hay costos sin determinar)"
    if resumen["evaluados"]:
        texto += f" · Aciertos: {resumen['aciertos']}/{resumen['evaluados']} ({resumen['precision_pct']:.2f}%)"
        if estricto:
            texto += " · Clave nominal; la conversión estricta puede cambiar los veredictos."
    opciones = [(f"{r['n']}. {escape(r['archivo'])}", str(r["n"])) for r in lote["resultados"]]
    primero = opciones[0][1]
    if progreso is not None:
        progreso(1, desc="Análisis completo")
    return (lote["agentes"], resultados, historiales, texto, filas, descarga,
            gr.update(choices=opciones, value=primero), _html_veredicto(resultados[primero]), historiales[primero])


def analizar_repositorio(json_empresa, modelo, estricto, progreso=gr.Progress()):
    return analizar(None, json_empresa, modelo, estricto, progreso)


def ver_detalle(nombre, resultados, historiales):
    if nombre not in (resultados or {}):
        return "", []
    return _html_veredicto(resultados[nombre]), historiales.get(nombre, [])


def chatear(mensaje, nombre, agentes, historiales):
    if not mensaje or not mensaje.strip():
        return "", historiales.get(nombre, []), historiales
    agente = (agentes or {}).get(nombre)
    if agente is None:
        raise gr.Error("Selecciona un proceso cuyo análisis se haya completado.")
    try:
        respuesta = agente.preguntar(mensaje)
    except Exception as exc:
        raise gr.Error("No se pudo responder. Revisa la conexión y la configuración del modelo.") from exc
    historial = list(historiales.get(nombre, [])) + [
        {"role": "user", "content": escape(mensaje)},
        {"role": "assistant", "content": escape(respuesta)}]
    historiales = {**historiales, nombre: historial}
    return "", historial, historiales


def crear_app():
    """Construye Blocks sin iniciar el servidor ni consultar modelos."""
    with gr.Blocks(title="SECOP | Decisión de oferta", css=CSS,
                   theme=gr.themes.Base(font=["Arial", "sans-serif"])) as demo:
        gr.Markdown("# ¿A qué procesos podemos presentarnos?\n"
                    "Analiza los 20 procesos del repositorio con la empresa multipropósito. "
                    "También puedes cargar otros pliegos y el perfil de tu empresa.", elem_id="portada")
        agentes, resultados, historiales = gr.State({}), gr.State({}), gr.State({})
        with gr.Row():
            empresa = gr.File(label="Empresa JSON (opcional)", file_types=[".json"], type="filepath")
            with gr.Column():
                modelo = gr.Dropdown(["claude-haiku-5-5", "claude-sonnet-5-5"],
                                     value="claude-haiku-5-5", label="Modelo")
                estricto = gr.Checkbox(label="Convertir experiencia a SMMLV de cada año", value=False)
        repositorio = gr.Button("Analizar los 20 procesos del repositorio", variant="primary")
        with gr.Accordion("Analizar otros PDF", open=False):
            pdfs = gr.File(label="Pliegos en PDF", file_types=[".pdf"], file_count="multiple", type="filepath")
            subir = gr.Button("Analizar los PDF cargados")
        resumen = gr.Markdown("Los resultados aparecerán aquí al terminar el análisis.")
        tabla = gr.Dataframe(headers=["#", "Archivo", "Referencia", "Entidad", "Veredicto", "Motivo",
                                      "Costo USD", "Tokens", "Latencia s"],
                             datatype=["number", "str", "str", "str", "str", "str", "number", "number", "number"],
                             interactive=False, wrap=True)
        descarga = gr.File(label="Descargar resultados en CSV")
        gr.Markdown("## Revisión de un proceso")
        elegido = gr.Dropdown(choices=[], label="Proceso", interactive=True)
        detalle = gr.HTML()
        chat = gr.Chatbot(label="Preguntas sobre este proceso", type="messages", sanitize_html=True, height=260)
        entrada = gr.Textbox(label="Pregunta", placeholder="¿Qué requisito nos falta cumplir?")
        salidas = [agentes, resultados, historiales, resumen, tabla, descarga, elegido, detalle, chat]
        repositorio.click(analizar_repositorio, [empresa, modelo, estricto], salidas,
                          concurrency_limit=1, concurrency_id="analisis")
        subir.click(analizar, [pdfs, empresa, modelo, estricto], salidas,
                    concurrency_limit=1, concurrency_id="analisis")
        elegido.change(ver_detalle, [elegido, resultados, historiales], [detalle, chat])
        entrada.submit(chatear, [entrada, elegido, agentes, historiales], [entrada, chat, historiales],
                       concurrency_limit=1)
    return demo


if __name__ == "__main__":
    crear_app().queue().launch(share=bool(os.getenv("COLAB_RELEASE_TAG")))
