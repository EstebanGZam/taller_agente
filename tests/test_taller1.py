"""Pruebas del Taller 1 sin claves ni red: datos de las herramientas, configuración A y juez.

Uso:  python -m pytest -q
"""
import io
import json
import re
import sys
import tokenize
from pathlib import Path
from types import SimpleNamespace

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import agente  # noqa: E402  (no necesita ANTHROPIC_API_KEY: el agente se crea al usarlo)
import evaluar  # noqa: E402
import resumen_experimentos as rx  # noqa: E402

CASOS = {c["id"]: c for c in json.loads((RAIZ / "datos" / "conjunto_referencia.json").read_text(encoding="utf-8"))}
B = {t.name: t for t in agente.HERRAMIENTAS["B"]}


@pytest.fixture(autouse=True)
def muestra(monkeypatch):
    monkeypatch.setenv("MODO_DATOS", "muestra")


def buscar(**filtros):
    return B["buscar_procesos"].invoke(filtros)


# --- Configuración A: idéntica al agente base del curso -------------------------------------

PROMPT_ORIGINAL = """Eres un analista de contratación pública colombiana que asesora a una empresa.
Responde en español, de forma breve y precisa.
- Usa las herramientas solo cuando necesites datos; las preguntas generales respóndelas directamente.
- Nunca inventes cifras: si un dato no está o parece inconsistente (por ejemplo, plazo de 0 días
  o presupuesto de $0), dilo explícitamente.
- Cita la referencia del proceso cuando hables de uno."""

DESCRIPCIONES_ORIGINALES = {
    "buscar_procesos": "Busca procesos de contratación pública en el SECOP II. Filtros opcionales: entidad,\n"
                       "departamento, ciudad, modalidad (p. ej. 'licitaci', 'directa', 'abreviada'), texto en la\n"
                       "descripción, rango de presupuesto en pesos y rango de fechas de publicación (AAAA-MM-DD).\n"
                       "Devuelve los procesos y un resumen con cantidad, presupuesto total y promedio.",
    "consultar_empresa": "Devuelve el perfil de NUESTRA empresa: RUP, códigos UNSPSC, estados financieros y\n"
                         "contratos de experiencia.",
    "calcular_indicadores": "Calcula los indicadores financieros de NUESTRA empresa: liquidez, endeudamiento,\n"
                            "cobertura de intereses, capital de trabajo y rentabilidades. Si se da el presupuesto de\n"
                            "un proceso, calcula qué porcentaje de él cubre nuestro capital de trabajo.",
}


def test_config_a_reproduce_prompt_y_descripciones_originales():
    assert agente.PROMPTS["A"] == PROMPT_ORIGINAL
    assert {t.name: t.description for t in agente.HERRAMIENTAS["A"]} == DESCRIPCIONES_ORIGINALES
    assert set(B) == set(DESCRIPCIONES_ORIGINALES)  # B conserva los nombres que espera el evaluador


def test_config_a_devuelve_lo_mismo_que_herramientas():
    import herramientas as h
    a = {t.name: t for t in agente.HERRAMIENTAS["A"]}
    assert a["buscar_procesos"].invoke({"modalidad": "licitaci"}) == h.buscar_procesos(modalidad="licitaci")
    assert a["calcular_indicadores"].invoke({}) == h.calcular_indicadores()


def test_sin_temperature_top_p_top_k():
    for archivo in ("agente.py", "evaluar.py"):
        codigo = (RAIZ / archivo).read_text(encoding="utf-8")
        sin_comentarios = " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(codigo).readline)
                                   if t.type not in (tokenize.COMMENT, tokenize.STRING))
        assert not re.search(r"\b(temperature|top_p|top_k)\s*=", sin_comentarios), archivo


def test_config_invalida():
    with pytest.raises(ValueError):
        agente.obtener_agente("C")


# --- Datos que necesita cada pregunta de referencia (configuración B) ------------------------

def test_r01_a_r04_indicadores():
    ind = B["calcular_indicadores"].invoke({})
    assert ind["indice_liquidez"] == 1.7                      # R01
    assert ind["indice_endeudamiento_pct"] == 62.0            # R02
    assert ind["razon_cobertura_intereses"] == 2.5            # R03
    assert ind["capital_de_trabajo"] == 182_000_000           # R04


def test_r05_r06_empresa():
    perfil = B["consultar_empresa"].invoke({"codigo_unspsc": "81101500"})
    assert ", ".join(perfil["rup"]["codigos_unspsc"]) == CASOS["R05"]["respuesta_esperada"]
    assert perfil["experiencia_con_codigo"] == {"codigo": "81101500", "contratos": ["EXP-1", "EXP-2"],
                                                "valor_total": 950_000_000}


@pytest.mark.parametrize("pedida", ["directa", "Contratación directa", "contratacion directa"])
def test_r07_contratacion_directa(pedida):
    r = buscar(modalidad=pedida)
    assert r["resumen"]["cantidad"] == 9
    assert all(p["modalidad"] == "Contratación directa" for p in r["procesos"])


def test_r07_directa_no_cuenta_otras_modalidades():
    assert not agente._modalidad_coincide("directa", "Contratación directa (con ofertas)")
    assert not agente._modalidad_coincide("directa", "Contratación régimen especial")
    assert agente._modalidad_coincide("régimen especial", "Contratación régimen especial (con ofertas)")


@pytest.mark.parametrize("limite", [1, 5, 20])
def test_r08_mayor_presupuesto_de_los_20(limite):
    r = buscar(limite=limite)
    assert r["resumen"]["cantidad"] == 20
    assert r["resumen"]["mayor_presupuesto"] == {"referencia": "LP-DT-CAU-003-2019", "entidad": "INVIAS",
                                                 "presupuesto_oficial": 53_520_784_298}
    assert buscar(ordenar_por="presupuesto", limite=1)["procesos"][0]["referencia"] == "LP-DT-CAU-003-2019"


def test_r09_licitaciones():
    assert buscar(modalidad="licitación pública")["resumen"]["referencias"] == ["FDLS-LP-098-2019", "LP-DT-CAU-003-2019"]


def test_r10_promedio_antioquia():
    r = buscar(departamento="Antioquia")["resumen"]
    assert r["cantidad"] == 3 and round(r["presupuesto_promedio"]) == 120_120_571


def test_r11_santander_mas_reciente():
    r = buscar(departamento="Santander")
    assert r["resumen"]["cantidad"] == 2
    assert r["resumen"]["mas_reciente"] == {"referencia": "FLO-INFRA-SA-MC-007-2025", "fecha_publicacion": "2025-12-30"}
    assert buscar(departamento="Santander", ordenar_por="fecha")["procesos"][0]["referencia"] == "FLO-INFRA-SA-MC-007-2025"


def test_r12_unidad_de_contratacion():
    p = next(p for p in buscar(entidad="INVIAS")["procesos"] if p["unidad_de_contratacion"] == "Dirección Territorial Cauca")
    assert (p["referencia"], p["ciudad_entidad"], p["ciudad_unidad_de_contratacion"]) == ("LP-DT-CAU-003-2019", "Bogotá", "Cali")


def test_r13_ofertas_y_adjudicatario():
    [p] = buscar(referencia="LP-DT-CAU-003-2019")["procesos"]
    assert p["ofertas_recibidas"] == 51
    assert p["proveedor_adjudicado"] == "COOPERATIVA DE TRABAJO ASOCIADO SIGLO XXI"
    assert p["valor_adjudicado"] == 1_217_527_123


def test_r14_plazo_cero_es_inconsistente():
    [p] = buscar(referencia="SA-004-2023")["procesos"]
    assert p["plazo"].startswith("0 ") and any("inconsistente" in a for a in p["alertas"])


def test_r15_presupuesto_cero_no_disponible():
    [p] = buscar(entidad="Agrosavia")["procesos"]
    assert p["presupuesto_oficial"] == 0 and any("no está disponible" in a for a in p["alertas"])


def test_r16_capital_de_trabajo_vs_40_por_ciento():
    [p] = buscar(referencia="FLO-INFRA-SA-MC-007-2025")["procesos"]
    assert p["presupuesto_oficial"] == 305_813_941
    ind = B["calcular_indicadores"].invoke({"presupuesto_proceso": p["presupuesto_oficial"], "porcentaje_exigido": 40})
    assert ind["capital_trabajo_pct_del_presupuesto"] == 59.5
    assert ind["monto_exigido"] == 122_325_576 and ind["cumple_capital_de_trabajo"] is True


def test_r17_r18_no_esperan_herramientas():
    assert CASOS["R17"]["herramientas_esperadas"] == [] and CASOS["R18"]["herramientas_esperadas"] == []
    assert "SIN herramientas" in agente.PROMPTS["B"]


def test_salida_b_es_mas_compacta_que_a():
    a = {t.name: t for t in agente.HERRAMIENTAS["A"]}
    assert len(json.dumps(buscar(limite=20), ensure_ascii=False)) < len(json.dumps(a["buscar_procesos"].invoke({}), ensure_ascii=False))


# --- Extracción de la respuesta y de las herramientas ----------------------------------------

def test_herramientas_usadas_con_dicts_y_objetos():
    mensajes = [SimpleNamespace(tool_calls=[{"name": "buscar_procesos", "args": {}}]),
                SimpleNamespace(tool_calls=[SimpleNamespace(name="calcular_indicadores")]),
                SimpleNamespace(content="sin herramientas")]
    assert agente.herramientas_usadas(mensajes) == ["buscar_procesos", "calcular_indicadores"]


def test_texto_de_bloques():
    assert agente.texto_de("hola") == "hola"
    assert agente.texto_de([{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "1.7"}]) == "1.7"


# --- Juez y resúmenes de evaluar.py / resumen_experimentos.py --------------------------------

@pytest.mark.parametrize("texto, esperado", [
    ("CORRECTA", 1), ("INCORRECTA", 0), ("incorrecta.", 0), ("Correcta", 1),
    ("**INCORRECTA**: la cifra no coincide", 0), ([{"type": "text", "text": "CORRECTA"}], 1), ("no sé", None),
])
def test_juez_lee_veredicto(texto, esperado):
    assert evaluar.leer_veredicto(texto) == esperado


def test_uso_herramientas():
    ref = {"herramientas": ["buscar_procesos", "calcular_indicadores"]}
    assert evaluar.uso_herramientas({}, {"herramientas": ["buscar_procesos", "calcular_indicadores"]}, ref)["score"] == 1
    assert evaluar.uso_herramientas({}, {"herramientas": ["buscar_procesos"]}, ref)["score"] == 0
    assert evaluar.uso_herramientas({}, {"herramientas": []}, {"herramientas": []})["score"] == 1
    assert evaluar.uso_herramientas({}, {"herramientas": ["consultar_empresa"]}, {"herramientas": []})["score"] == 0


def _filas(n, correctas, costo=0.001):
    return [{"id": f"R{i + 1:02d}", "pregunta": "", "herramientas": [], "respuesta": "", "uso_herramientas": 1,
             "respuesta_correcta": int(i < correctas), "tokens_entrada": 1000, "tokens_salida": 100,
             "costo_usd": costo, "latencia_s": float(i + 1), "error": None} for i in range(n)]


def test_resumir_metricas():
    r = evaluar.resumir(_filas(18, 16), "B-mejorado", "B", "anthropic:claude-haiku-5-5")
    assert r["precision_respuesta_correcta"] == 88.9 and r["precision_uso_herramientas"] == 100.0
    assert r["precision_combinada"] == 88.9
    assert r["tokens_entrada"] == 18_000 and r["costo_total_usd"] == 0.018
    assert r["costo_por_pregunta_usd"] == 0.001 and r["costo_por_respuesta_correcta_usd"] == round(0.018 / 16, 6)
    assert r["latencia_p50_s"] == 9.5 and r["latencia_p95_s"] == 17.15


def test_comparacion_configuraciones(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(rx, "RESULTADOS", tmp_path)
    for prefijo, config, correctas, costo in (("A-base", "A", 12, 0.002), ("B-mejorado", "B", 17, 0.001)):
        r = evaluar.resumir(_filas(18, correctas, costo), prefijo, config, "anthropic:claude-haiku-5-5")
        (tmp_path / f"experimento_{prefijo}.json").write_text(json.dumps(r), encoding="utf-8")
    md = rx.generar(["A-base", "B-mejorado"])
    assert "| | A-base | B-mejorado |" in md and "Respuesta correcta | 66.7 % | 94.4 %" in md
    assert "Cumplen el criterio de 85 % de respuestas correctas: B-mejorado (94.4 %)." in md
    assert "La más barata por respuesta correcta es B-mejorado" in md


def modelo_falso(mensajes):
    """Modelo de chat que responde con mensajes fijos (sin red), para probar el circuito completo."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class Falso(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return Falso(messages=iter(mensajes))


def test_objetivo_registra_herramientas_tokens_y_costo(monkeypatch):
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage

    uso, meta = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}, {"model_name": "claude-haiku-5-5"}
    modelo = modelo_falso([
        AIMessage(content=[{"type": "text", "text": "Busco."}], usage_metadata=uso, response_metadata=meta,
                  tool_calls=[{"name": "buscar_procesos", "args": {"modalidad": "directa"}, "id": "t1"}]),
        AIMessage(content=[{"type": "text", "text": "Hay 9."}], usage_metadata=uso, response_metadata=meta),
    ])
    monkeypatch.setattr(agente, "CONFIG", "B")
    monkeypatch.setitem(agente._agentes, ("B", agente.MODELO),
                        create_agent(modelo, tools=agente.HERRAMIENTAS["B"], system_prompt=agente.PROMPT_B))
    out = evaluar.objetivo({"pregunta": CASOS["R07"]["pregunta"]})
    assert out["respuesta"] == "Hay 9." and out["herramientas"] == ["buscar_procesos"]
    m = out["metricas"]
    assert (m["llamadas_modelo"], m["tokens_entrada"], m["tokens_salida"]) == (2, 2000, 100)
    assert m["costo_usd"] == pytest.approx(2000 * 0.10 / 1e6 + 100 * 0.50 / 1e6) and m["latencia_s"] >= 0


def test_fila_de_resultado_de_langsmith():
    from langsmith.evaluation import EvaluationResult

    fila = evaluar.fila_de({
        "run": SimpleNamespace(outputs={"respuesta": "9", "herramientas": ["buscar_procesos"],
                                        "metricas": {"tokens_entrada": 10, "tokens_salida": 2, "costo_usd": 0.1, "latencia_s": 1.5}},
                               error=None, start_time=None, end_time=None),
        "example": SimpleNamespace(inputs={"pregunta": CASOS["R07"]["pregunta"]}),
        "evaluation_results": {"results": [EvaluationResult(key="uso_herramientas", score=1),
                                           EvaluationResult(key="respuesta_correcta", score=0)]},
    })
    assert fila["id"] == "R07" and fila["uso_herramientas"] == 1 and fila["respuesta_correcta"] == 0
    assert fila["costo_usd"] == 0.1 and fila["latencia_s"] == 1.5
