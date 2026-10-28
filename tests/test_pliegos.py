"""Pruebas sin red del agente, los reportes, la evaluación y Gradio."""
import copy
import csv
import importlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

import agente_aptitud
import analizar_procesos as lotes
import evaluar_aptitud as evaluar
import pliegos

BASE = Path(__file__).resolve().parents[1]
CLAVE = json.loads((BASE / "datos/clave_aptitud.json").read_text())["procesos"]
EMPRESA = lotes.cargar_empresa()


@pytest.fixture(autouse=True)
def sin_red(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    for nombre in ["ANTHROPIC_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"]:
        monkeypatch.delenv(nombre, raising=False)

    def bloquear(*args, **kwargs):
        raise AssertionError("Las pruebas no deben usar la red")

    monkeypatch.setattr(socket.socket, "connect", bloquear)


class ModeloSimulado(BaseChatModel):
    """Modelo local con mensajes y contabilidad reales de LangChain."""
    respuestas: list
    _posicion: int = PrivateAttr(default=0)

    @property
    def _llm_type(self):
        return "simulado"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        respuesta = self.respuestas[self._posicion]
        self._posicion += 1
        return ChatResult(generations=[ChatGeneration(message=respuesta)])


def mensaje(texto="", herramientas=None):
    return AIMessage(content=texto, tool_calls=herramientas or [],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        response_metadata={"model_name": "claude-haiku-5-5"})


def llamada(nombre, argumentos=None):
    return {"name": nombre, "args": argumentos or {}, "id": "id-" + nombre, "type": "tool_call"}


def req_completo(proceso):
    return {**proceso["requisitos_esperados"], "presupuesto_oficial": proceso["presupuesto_oficial"],
            "referencia": proceso["referencia"], "entidad": proceso["entidad"],
            "modalidad": proceso["modalidad"], "objeto": proceso["referencia"],
            "fecha_publicacion": "2019-04-29" if proceso["n"] == 11 else "2024-01-01"}


@pytest.fixture
def extraccion_simulada(monkeypatch):
    def extraer(texto, modelo=None, config=None):
        assert config and config["callbacks"]
        proceso = next(p for p in CLAVE if p["referencia"] in texto)
        return req_completo(proceso)

    def similares(objeto, experiencia, modelo=None, config=None):
        assert config and config["callbacks"]
        return next(p["contratos_similares_esperados"] for p in CLAVE if p["referencia"] == objeto)

    monkeypatch.setattr(pliegos, "extraer_requisitos", extraer)
    monkeypatch.setattr(pliegos, "contratos_similares", similares)
    # El grafo real responde sin herramientas: debe funcionar el respaldo determinista.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-simulada-sin-red")
    monkeypatch.setattr(agente_aptitud, "init_chat_model", lambda *a, **k: ModeloSimulado(
        respuestas=[mensaje("No consulté herramientas."), mensaje("Seguimiento con memoria.")]))


def test_esquema_y_lectura_completa():
    campos = set(pliegos.Requisitos.model_fields)
    assert set(CLAVE[0]["requisitos_esperados"]).issubset(campos)
    texto = pliegos.leer_pdf(BASE / "datos/11_FDLS_LP_098_2019.pdf")
    assert "FDLS-LP-098-2019" in texto and "Forma de pago" in texto


def test_pdf_corrupto(tmp_path):
    ruta = tmp_path / "roto.pdf"
    ruta.write_bytes(b"no es PDF")
    with pytest.raises(ValueError, match="No se pudo leer"):
        pliegos.leer_pdf(ruta)


def test_dedupe_sha256_y_nombre_canonico():
    rutas = lotes.deduplicar((BASE / "datos").glob("*.pdf"))
    assert len(rutas) == 20
    assert "13_LP_DT_CAU_003_2019.pdf" in [r.name for r in rutas]
    assert not any("(1)" in r.name for r in rutas)


@pytest.mark.parametrize("proceso", CLAVE, ids=lambda p: p["archivo"])
def test_respaldo_del_agente_los_20(proceso, extraccion_simulada):
    agente = agente_aptitud.AgenteAptitud(BASE / "datos" / proceso["archivo"], EMPRESA)
    res = agente.analizar()
    assert res["veredicto"] == proceso["veredicto_esperado"]
    assert {"extraer_requisitos", "verificar_requisitos"}.issubset(res["herramientas"])
    assert res["metricas"]["llamadas_modelo"] == 1
    assert res["explicacion"].startswith(res["veredicto"])
    assert agente.analizar() == res


def test_memoria_y_callback_en_seguimiento(extraccion_simulada):
    agente = agente_aptitud.AgenteAptitud(BASE / "datos/07_135116.pdf", EMPRESA)
    agente.analizar()
    estado = agente.agente.get_state({"configurable": {"thread_id": agente.thread_id}})
    assert any("Resultado verificado" in str(m.content) for m in estado.values["messages"])
    assert agente.preguntar("¿Por qué?") == "Seguimiento con memoria."
    assert agente.metricas_seguimiento[-1]["llamadas_modelo"] == 1


def test_callbacks_incluyen_extraccion_y_similitud(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-simulada-sin-red")
    req = req_completo(CLAVE[6])
    modelo_agente = ModeloSimulado(respuestas=[
        mensaje(herramientas=[llamada("extraer_requisitos")]),
        mensaje(herramientas=[llamada("verificar_requisitos")]), mensaje("APTA")])
    modelo_requisitos = ModeloSimulado(respuestas=[mensaje(herramientas=[llamada("Requisitos", req)])])
    modelo_similitud = ModeloSimulado(respuestas=[mensaje(herramientas=[llamada("Similares", {"ids": ["EXP-9"]})])])
    monkeypatch.setattr(agente_aptitud, "init_chat_model", lambda *a, **k: modelo_agente)
    modelos = iter([modelo_requisitos, modelo_similitud])

    def crear_modelo(nombre, **kwargs):
        assert not {"temperature", "top_p", "top_k"} & kwargs.keys()
        assert 0 < kwargs["max_tokens"] <= 2400
        return next(modelos)

    monkeypatch.setattr(pliegos, "init_chat_model", crear_modelo)
    res = agente_aptitud.AgenteAptitud(BASE / "datos/07_135116.pdf", EMPRESA).analizar()
    assert res["veredicto"] == "APTA"
    assert res["metricas"]["llamadas_modelo"] == 5
    assert res["metricas"]["tokens_entrada"] == 500
    assert res["metricas"]["tokens_salida"] == 50
    from costos import costo_usd
    assert res["metricas"]["costo_usd"] == round(5 * costo_usd("claude-haiku-5-5", 100, 10), 6)


def test_pdf_se_lee_una_vez(monkeypatch, extraccion_simulada):
    original, leidas = pliegos.leer_pdf, []

    def leer(ruta):
        leidas.append(ruta)
        return original(ruta)

    monkeypatch.setattr(pliegos, "leer_pdf", leer)
    agente = agente_aptitud.AgenteAptitud(BASE / "datos/07_135116.pdf", EMPRESA)
    agente.analizar()
    agente.preguntar("¿Qué sigue?")
    assert len(leidas) == 1


def test_traza_raiz_con_metadatos_sin_red(extraccion_simulada):
    from langsmith import tracing_context
    from langsmith.run_helpers import get_current_run_tree
    from langsmith import traceable
    capturadas = []

    @traceable(name="capturar")
    def capturar():
        raiz = get_current_run_tree()
        capturadas.append(raiz)
        return agente_aptitud.AgenteAptitud(BASE / "datos/07_135116.pdf", EMPRESA).analizar()

    with tracing_context(enabled="local"):
        res = capturar()
    analisis = next(r for r in capturadas[0].child_runs if r.name == "aptitud_analizar")
    assert res["run_id"] == str(analisis.id)
    assert analisis.metadata["archivo"] == "07_135116.pdf"
    assert analisis.metadata["modelo"] == "anthropic:claude-haiku-5-5"


def test_fallo_del_agente_con_respaldo_exitoso(monkeypatch, extraccion_simulada):
    class AgenteFallido:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("Ciclo del agente interrumpido")

    agente = agente_aptitud.AgenteAptitud(BASE / "datos/07_135116.pdf", EMPRESA)
    agente.agente = AgenteFallido()
    assert agente.analizar()["veredicto"] == "APTA"


def test_fallo_de_extraccion_devuelve_revision(monkeypatch, extraccion_simulada):
    def fallar(*args, **kwargs):
        raise ValueError("Salida estructurada inválida")

    monkeypatch.setattr(pliegos, "extraer_requisitos", fallar)
    res = agente_aptitud.AgenteAptitud(BASE / "datos/07_135116.pdf", EMPRESA).analizar()
    assert res["veredicto"] == "REVISAR MANUALMENTE"
    assert res["error"] is True
    assert res["metricas"]["llamadas_modelo"] == 1


def test_fallo_de_similitud_no_oculta_incumplimiento_financiero(monkeypatch, extraccion_simulada):
    def extraer(*args, **kwargs):
        return {**req_completo(CLAVE[17]), "liquidez_min": 5}

    def fallar(*args, **kwargs):
        raise RuntimeError("Similitud no disponible")

    monkeypatch.setattr(pliegos, "extraer_requisitos", extraer)
    monkeypatch.setattr(pliegos, "contratos_similares", fallar)
    res = agente_aptitud.AgenteAptitud(BASE / "datos/18_0511_2024.pdf", EMPRESA).analizar()
    assert res["veredicto"] == "NO APTA"
    assert any(c["cumple"] is None for c in res["checks"])
    assert any(c["cumple"] is False for c in res["checks"])


def test_herramientas_en_paralelo_no_duplican_extraccion(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from costos import ContadorCostos
    llamadas = []

    def extraer(*args, **kwargs):
        llamadas.append(1)
        return req_completo(CLAVE[0])

    monkeypatch.setattr(pliegos, "extraer_requisitos", extraer)
    agente = agente_aptitud.AgenteAptitud(BASE / "datos/01_EDP_545_2022.pdf", EMPRESA)
    config = agente._config(ContadorCostos(agente.modelo), "prueba")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futuros = [pool.submit(agente._extraer, config), pool.submit(agente._verificar, config)]
        for futuro in futuros:
            futuro.result()
    assert len(llamadas) == 1
    assert agente.resultado["veredicto"] == "NO APLICA"


def test_lote_cli_y_columnas(tmp_path, extraccion_simulada, monkeypatch):
    monkeypatch.setenv("EQUIPO", "Equipo de prueba")
    lote = lotes.main(["--salida", str(tmp_path), "--trabajadores", "3"])
    assert lote["resumen"]["procesos"] == 20
    assert lote["resumen"]["aciertos"] == 20
    assert lote["resumen"]["precision_pct"] == 100
    assert lote["resumen"]["conteos"] == {"APTA": 6, "NO APTA": 2, "NO APLICA": 12}
    assert lote["resumen"]["nombre"] == "Equipo de prueba"
    with (tmp_path / "aptitud_procesos.csv").open(encoding="utf-8-sig") as archivo:
        lector = csv.DictReader(archivo)
        assert lector.fieldnames == lotes.COLUMNAS
        assert len(list(lector)) == 20
    reporte = json.loads((tmp_path / "aptitud_procesos.json").read_text())
    assert reporte["resumen"]["tokens_entrada"] == 2000
    assert "20/20" in (tmp_path / "aptitud_procesos.md").read_text()


def test_comparacion_no_aplica_a_empresa_o_pdf_ajenos(tmp_path, extraccion_simulada):
    empresa = copy.deepcopy(EMPRESA)
    empresa["razon_social"] = "Otra empresa"
    lote = lotes.analizar_lote([BASE / "datos/07_135116.pdf"], empresa)
    assert lote["resumen"]["precision_pct"] is None
    ruta = tmp_path / "07_135116.pdf"
    ruta.write_bytes(b"PDF distinto")
    lote = lotes.analizar_lote([ruta], EMPRESA)
    assert lote["filas"][0]["acierto"] is None
    assert lote["resultados"][0]["veredicto"] == "REVISAR MANUALMENTE"


def test_app_importa_y_construye_sin_claves():
    app = importlib.import_module("app")
    demo = app.crear_app()
    assert demo.config["components"]
    with pytest.raises(app.gr.Error, match="ANTHROPIC_API_KEY"):
        app.analizar(progreso=None)


def test_app_sin_uploads_y_chat_aislado(extraccion_simulada, monkeypatch, tmp_path):
    import app
    monkeypatch.setattr(app.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    agentes, resultados, historiales, resumen, filas, csv_path, opciones, detalle, chat = app.analizar(progreso=None)
    assert len(filas) == 20 and len(filas[0]) == 9
    assert "20/20" in resumen
    assert Path(csv_path).is_file()
    assert "NO APLICA" in detalle
    _, _, nuevos = app.chatear("¿Qué sigue?", "7", agentes, historiales)
    assert len(nuevos["7"]) == len(historiales["7"]) + 2
    assert nuevos["1"] == historiales["1"]


def test_html_externo_escapado_y_enlaces_seguros():
    import app
    malicioso = '<img src=x onerror="alert(1)">'
    res = {"veredicto": "APTA", "referencia": malicioso, "entidad": malicioso,
           "resumen": malicioso, "explicacion": malicioso, "avisos": [malicioso],
           "trace_url": "javascript:alert(1)", "run_id": malicioso,
           "checks": [{"requisito": malicioso, "exigido": malicioso, "empresa": malicioso,
                       "detalle": malicioso, "cumple": True}]}
    html = app._html_veredicto(res)
    assert "<img" not in html and "&lt;img" in html
    assert "javascript:" not in html


def test_evaluadores_sin_juez():
    esperado = {"a": 1.0, "b": ["2", "1"], "c": None, "d": False}
    ejemplo = SimpleNamespace(outputs={"veredicto": "APTA", "requisitos": esperado})
    run = SimpleNamespace(outputs={"veredicto": "APTA", "requisitos":
        {"a": 1.0000005, "b": ["1", "2"], "c": None, "d": False},
        "herramientas": ["extraer_requisitos", "verificar_requisitos"]})
    assert evaluar.veredicto_correcto(run, ejemplo)["score"]
    assert evaluar.extraccion_requisitos(run, ejemplo)["score"] == 1
    assert evaluar.uso_herramientas(run, ejemplo)["score"]
    run.outputs["requisitos"]["a"] = 1.01
    del run.outputs["requisitos"]["c"]
    assert evaluar.extraccion_requisitos(run, ejemplo)["score"] == 0.5
    run.outputs["herramientas"] = []
    assert not evaluar.uso_herramientas(run, ejemplo)["score"]


def test_dataset_solo_se_crea_si_falta():
    class ClienteSimulado:
        existe = False
        creados = 0
        ejemplos = None

        def has_dataset(self, **kwargs):
            return self.existe

        def read_dataset(self, **kwargs):
            return SimpleNamespace(id="dataset")

        def create_dataset(self, *args, **kwargs):
            self.existe = True
            self.creados += 1
            return SimpleNamespace(id="dataset")

        def create_examples(self, *, dataset_id, examples):
            assert dataset_id == "dataset"
            self.ejemplos = examples

    cliente = ClienteSimulado()
    evaluar.asegurar_dataset(cliente)
    evaluar.asegurar_dataset(cliente)
    assert cliente.creados == 1
    assert len(cliente.ejemplos) == 20
    assert set(cliente.ejemplos[0]["inputs"]) == {"archivo"}
    assert set(cliente.ejemplos[0]["outputs"]) == {"veredicto", "requisitos"}


def test_evaluacion_prefijo_y_modelo(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-simulada-sin-red")
    monkeypatch.setenv("LANGSMITH_API_KEY", "clave-simulada-sin-red")
    monkeypatch.setenv("MODELO", "anthropic:claude-sonnet-5-5")

    class ClienteSimulado:
        def has_dataset(self, **kwargs):
            return True

        def read_dataset(self, **kwargs):
            return SimpleNamespace(id="dataset")

        def evaluate(self, target, **kwargs):
            assert target is evaluar.objetivo
            return kwargs

    monkeypatch.setattr(evaluar, "Client", ClienteSimulado)
    resultado = evaluar.main(["apt-sonnet"])
    assert resultado["experiment_prefix"] == "apt-sonnet"
    assert resultado["data"] == evaluar.DATASET
    assert resultado["metadata"]["modelo"] == "anthropic:claude-sonnet-5-5"
    assert len(resultado["evaluators"]) == 3
