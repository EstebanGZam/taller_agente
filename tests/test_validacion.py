"""Regresión de la clave y casos límite financieros y de experiencia."""
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from validacion import indicadores, validar_empresa, verificar

BASE = Path(__file__).resolve().parents[1]
CLAVE = json.loads((BASE / "datos/clave_aptitud.json").read_text())["procesos"]
EMPRESA = json.loads((BASE / "empresa_multiproposito.json").read_text())


def requisitos(proceso):
    return {**proceso["requisitos_esperados"], "presupuesto_oficial": proceso["presupuesto_oficial"]}


@pytest.mark.parametrize("proceso", CLAVE, ids=lambda p: p["archivo"])
def test_clave_completa(proceso):
    resultado = verificar(requisitos(proceso), EMPRESA, proceso["contratos_similares_esperados"])
    assert resultado["veredicto"] == proceso["veredicto_esperado"]


def test_indicadores_multiproposito():
    ind = indicadores(EMPRESA, 600_000_000)
    assert ind["indice_liquidez"] == 2.1
    assert ind["indice_endeudamiento_pct"] == 55.0
    assert ind["razon_cobertura_intereses"] == 3.2
    assert ind["capital_de_trabajo"] == 381_000_000
    assert ind["capital_trabajo_pct_del_presupuesto"] == 63.5


def test_smmlv_estricto_cambia_proceso_11():
    req = {**requisitos(CLAVE[10]), "fecha_publicacion": "2019-04-29"}
    assert verificar(req, EMPRESA, [])['veredicto'] == "APTA"
    assert verificar(req, EMPRESA, [], smmlv_por_anio=True)['veredicto'] == "NO APTA"


def test_smmlv_desconocido_no_se_inventa():
    req = {**requisitos(CLAVE[10]), "fecha_publicacion": "1990-04-29"}
    assert verificar(req, EMPRESA, [], True)["veredicto"] == "REVISAR MANUALMENTE"


def test_umbral_nulo_no_es_faltante():
    req = {**requisitos(CLAVE[17]), "exp_por_objeto_similar": False,
           "exp_max_contratos": None, "exp_pct_presupuesto": None}
    res = verificar(req, EMPRESA)
    assert res["veredicto"] == "APTA"
    assert len(res["checks"]) == 2


@pytest.mark.parametrize("presupuesto", [0, None])
def test_presupuesto_incompleto_con_similar(presupuesto):
    req = {**requisitos(CLAVE[6]), "presupuesto_oficial": presupuesto}
    res = verificar(req, EMPRESA, ["EXP-9"])
    assert res["veredicto"] == "APTA"
    assert any("Presupuesto incompleto" in a for a in res["avisos"])
    assert verificar(req, EMPRESA, [])["veredicto"] == "NO APTA"
    assert verificar(req, EMPRESA, None)["veredicto"] == "REVISAR MANUALMENTE"


def test_presupuesto_cero_estricto_no_necesita_conversion():
    empresa = copy.deepcopy(EMPRESA)
    empresa["experiencia"][-1]["anio"] = 1990
    assert verificar(requisitos(CLAVE[6]), empresa, ["EXP-9"], True)["veredicto"] == "APTA"


def test_sin_extraccion_no_se_declara_apta():
    assert verificar(None, EMPRESA)["veredicto"] == "REVISAR MANUALMENTE"


def test_datos_necesarios_indeterminables_y_prioridad_del_incumplimiento():
    req = {**requisitos(CLAVE[10]), "presupuesto_oficial": None}
    assert verificar(req, EMPRESA)["veredicto"] == "REVISAR MANUALMENTE"
    req["liquidez_min"] = 5
    assert verificar(req, EMPRESA)["veredicto"] == "NO APTA"
    req["verifica_capacidad_financiera"] = False
    req.update(codigos_unspsc_experiencia=[], exp_max_contratos=None,
               exp_pct_presupuesto=None, exp_contrato_min_pct=None)
    assert verificar(req, EMPRESA)["veredicto"] == "APTA"


def test_experiencia_sin_criterio_es_indeterminable():
    req = {**requisitos(CLAVE[10]), "codigos_unspsc_experiencia": []}
    assert verificar(req, EMPRESA)["veredicto"] == "REVISAR MANUALMENTE"


def test_cobertura_indeterminada_y_divisores_cero():
    empresa = copy.deepcopy(EMPRESA)
    empresa["estados_financieros"]["gastos_de_intereses"] = 0
    assert indicadores(empresa)["razon_cobertura_intereses"] == "indeterminado"
    assert verificar(requisitos(CLAVE[10]), empresa)["veredicto"] == "APTA"
    empresa["estados_financieros"]["pasivo_corriente"] = 0
    assert verificar(requisitos(CLAVE[10]), empresa)["veredicto"] == "REVISAR MANUALMENTE"


def test_no_redondear_antes_de_comparar():
    # 727 / 346 es menor que 2.105 aunque se muestre como 2.1.
    req = {"liquidez_min": 2.105}
    assert verificar(req, EMPRESA)["veredicto"] == "NO APTA"


def test_elegir_mayores_e_interseccion_de_codigos():
    req = {"verifica_capacidad_financiera": False, "presupuesto_oficial": 900_000_000,
           "codigos_unspsc_experiencia": ["V1.81101500", "72141000"],
           "exp_max_contratos": 1, "exp_pct_presupuesto": 100}
    assert verificar(req, EMPRESA)["veredicto"] == "NO APTA"
    req["exp_max_contratos"] = 2
    assert verificar(req, EMPRESA)["veredicto"] == "APTA"
    req["exp_contrato_min_pct"] = 70
    assert verificar(req, EMPRESA)["veredicto"] == "NO APTA"


@pytest.mark.parametrize("empresa", [None, [], {}, {**EMPRESA, "estados_financieros": []},
    {**EMPRESA, "experiencia": [1]}, {**EMPRESA, "rup": "sí"}])
def test_perfiles_invalidos(empresa):
    assert validar_empresa(empresa)


def test_numeros_invalidos_y_contratos_duplicados():
    empresa = copy.deepcopy(EMPRESA)
    empresa["estados_financieros"]["activo_total"] = float("nan")
    empresa["experiencia"].append(empresa["experiencia"][0])
    assert len(validar_empresa(empresa)) == 2


def test_empresa_original_sin_clave_propia():
    empresa = json.loads((BASE / "datos/empresa.json").read_text())
    # Los juicios de similitud de la clave multipropósito no se reutilizan en esta empresa.
    similares = {3: [], 7: [], 8: [], 18: []}
    resultados = [(p["n"], verificar(requisitos(p), empresa, similares.get(p["n"], []))["veredicto"])
                  for p in CLAVE]
    print("Veredictos datos/empresa.json con requisitos de referencia:", resultados)
    print("Conteo sin clave de precisión:", dict(Counter(v for _, v in resultados)))
    assert len(resultados) == 20
    assert all(v in {"APTA", "NO APTA", "NO APLICA", "REVISAR MANUALMENTE"} for _, v in resultados)
