"""Compara requisitos y empresa sin delegar el veredicto al modelo."""
from __future__ import annotations

import math

# Tabla del ejercicio de referencia, en pesos colombianos.
SMMLV = {
    2018: 781_242, 2019: 828_116, 2020: 877_803, 2021: 908_526,
    2022: 1_000_000, 2023: 1_160_000, 2024: 1_300_000,
    2025: 1_423_500, 2026: 1_750_905,
}
CAMPOS_FINANCIEROS = (
    "activo_corriente", "pasivo_corriente", "activo_total", "pasivo_total",
    "patrimonio", "utilidad_operacional", "gastos_de_intereses",
)


def _numero(valor):
    return (isinstance(valor, (int, float)) and not isinstance(valor, bool)
            and math.isfinite(valor))


def validar_empresa(empresa: dict) -> list[str]:
    """Devuelve los errores del perfil; una lista vacía indica que es válido."""
    if not isinstance(empresa, dict):
        return ["El JSON de la empresa debe ser un objeto."]
    errores = []
    if not isinstance(empresa.get("razon_social"), str) or not empresa["razon_social"].strip():
        errores.append("Falta una razón social válida.")
    rup = empresa.get("rup")
    if not isinstance(rup, dict) or not isinstance(rup.get("vigente"), bool):
        errores.append("rup debe indicar vigente como booleano.")
    elif not isinstance(rup.get("codigos_unspsc"), list):
        errores.append("rup.codigos_unspsc debe ser una lista.")
    ef = empresa.get("estados_financieros")
    if not isinstance(ef, dict):
        errores.append("estados_financieros debe ser un objeto.")
    else:
        for clave in CAMPOS_FINANCIEROS:
            if not _numero(ef.get(clave)):
                errores.append(f"estados_financieros.{clave} debe ser un número finito.")
            elif clave not in {"patrimonio", "utilidad_operacional"} and ef[clave] < 0:
                errores.append(f"estados_financieros.{clave} no puede ser negativo.")
    experiencia = empresa.get("experiencia")
    if not isinstance(experiencia, list):
        errores.append("experiencia debe ser una lista.")
    else:
        ids = set()
        for i, contrato in enumerate(experiencia, 1):
            if not isinstance(contrato, dict):
                errores.append(f"El contrato {i} debe ser un objeto.")
                continue
            identificador = contrato.get("id")
            if not isinstance(identificador, str) or not identificador or identificador in ids:
                errores.append(f"El contrato {i} necesita un id único de texto.")
            else:
                ids.add(identificador)
            if not isinstance(contrato.get("objeto"), str) or not contrato["objeto"].strip():
                errores.append(f"Falta el objeto del contrato {i}.")
            if not _numero(contrato.get("valor")) or contrato["valor"] < 0:
                errores.append(f"El valor del contrato {i} debe ser un número no negativo.")
            if not isinstance(contrato.get("codigos_unspsc"), list):
                errores.append(f"Los códigos del contrato {i} deben ser una lista.")
            if not isinstance(contrato.get("anio"), int) or isinstance(contrato.get("anio"), bool):
                errores.append(f"El año del contrato {i} debe ser un entero.")
    return errores


def _indicadores(empresa, presupuesto=None):
    ef = empresa["estados_financieros"]

    def razon(numerador, denominador):
        return ef[numerador] / ef[denominador] if ef[denominador] else None

    ind = {
        "indice_liquidez": razon("activo_corriente", "pasivo_corriente"),
        "indice_endeudamiento_pct": (100 * ef["pasivo_total"] / ef["activo_total"]
                                     if ef["activo_total"] else None),
        "razon_cobertura_intereses": (razon("utilidad_operacional", "gastos_de_intereses")
                                      if ef["gastos_de_intereses"] else "indeterminado"),
        "capital_de_trabajo": ef["activo_corriente"] - ef["pasivo_corriente"],
        "rentabilidad_patrimonio": razon("utilidad_operacional", "patrimonio"),
        "rentabilidad_activo": razon("utilidad_operacional", "activo_total"),
        "corte_estados_financieros": ef.get("corte"),
    }
    if presupuesto:
        ind["capital_trabajo_pct_del_presupuesto"] = 100 * ind["capital_de_trabajo"] / presupuesto
    return ind


def indicadores(empresa: dict, presupuesto: float | None = None) -> dict:
    """Calcula indicadores; los cocientes con divisor cero quedan sin determinar."""
    ind = _indicadores(empresa, presupuesto)
    decimales = {"indice_liquidez": 2, "indice_endeudamiento_pct": 1,
                 "razon_cobertura_intereses": 2, "rentabilidad_patrimonio": 3,
                 "rentabilidad_activo": 3, "capital_trabajo_pct_del_presupuesto": 1}
    return {k: round(v, decimales[k]) if k in decimales and _numero(v) else v
            for k, v in ind.items()}


def verificar(req: dict | None, empresa: dict, similares_ids: list[str] | None = None,
              smmlv_por_anio: bool = False) -> dict:
    """Verifica umbrales explícitos, experiencia y prioridad del incumplimiento."""
    if req is None:
        return {"referencia": None, "entidad": None, "modalidad": None, "presupuesto_oficial": None,
                "veredicto": "REVISAR MANUALMENTE", "resumen": "No se pudieron determinar los requisitos del PDF.",
                "checks": [{"requisito": "Extracción", "exigido": "Requisitos del PDF",
                            "empresa": "Sin determinar", "cumple": None, "detalle": ""}],
                "avisos": ["Revisa el PDF, la conexión y la configuración del modelo."]}
    checks, avisos = [], list(req.get("observaciones") or [])
    base = {k: req.get(k) for k in ("referencia", "entidad", "modalidad", "presupuesto_oficial")}

    def resultado(veredicto, resumen, **extras):
        return {**base, "veredicto": veredicto, "resumen": resumen,
                "checks": checks, "avisos": avisos, **extras}

    if req.get("es_contratacion_directa"):
        return resultado("NO APLICA", "Contratación directa o sin convocatoria competitiva: no hay oferta que presentar.")
    if req.get("es_venta_de_bienes"):
        return resultado("NO APLICA", "La entidad vende bienes; la empresa participaría como compradora.")

    def agregar(nombre, exigido, valor, cumple, detalle=""):
        checks.append({"requisito": nombre, "exigido": exigido, "empresa": valor,
                       "cumple": cumple, "detalle": detalle})

    errores = validar_empresa(empresa)
    if errores:
        agregar("Perfil de empresa", "Datos válidos", "; ".join(errores), None)
        return resultado("REVISAR MANUALMENTE", "No se pudo validar el perfil de empresa.")
    presupuesto = req.get("presupuesto_oficial")
    ind = _indicadores(empresa, presupuesto)
    if req.get("verifica_capacidad_financiera") is False:
        agregar("Capacidad financiera", "No se verifica", "No se exige", True)
    else:
        umbrales = [
            ("Liquidez", "liquidez_min", "indice_liquidez", False),
            ("Endeudamiento (%)", "endeudamiento_max_pct", "indice_endeudamiento_pct", True),
            ("Cobertura de intereses", "cobertura_intereses_min", "razon_cobertura_intereses", False),
            ("Rentabilidad del patrimonio", "rentabilidad_patrimonio_min", "rentabilidad_patrimonio", False),
            ("Rentabilidad del activo", "rentabilidad_activo_min", "rentabilidad_activo", False),
        ]
        for nombre, campo, indicador, maximo in umbrales:
            limite = req.get(campo)
            if limite is None:
                continue
            valor = ind[indicador]
            cumple = (True if valor == "indeterminado" else None if valor is None
                      else valor <= limite if maximo else valor >= limite)
            agregar(nombre, f"{'≤' if maximo else '≥'} {limite}", valor, cumple)
        pct = req.get("capital_trabajo_min_pct_presupuesto")
        if pct is not None:
            minimo = presupuesto * pct / 100 if presupuesto else None
            agregar("Capital de trabajo", f"≥ {pct}% del presupuesto", ind["capital_de_trabajo"],
                    None if minimo is None else ind["capital_de_trabajo"] >= minimo,
                    "Falta un presupuesto válido." if minimo is None else f"Mínimo: {minimo:,.2f} COP")

    codigos = {str(c).removeprefix("V1.") for c in req.get("codigos_unspsc_experiencia", [])}
    por_objeto = bool(req.get("exp_por_objeto_similar")) and not codigos
    maximo, pct, pct_uno = (req.get("exp_max_contratos"), req.get("exp_pct_presupuesto"),
                            req.get("exp_contrato_min_pct"))
    pide_exp = bool(codigos or por_objeto or any(v is not None for v in (maximo, pct, pct_uno)))
    if pide_exp:
        faltantes = []
        if not codigos and not por_objeto:
            faltantes.append("códigos UNSPSC o criterio de objeto similar")
        if por_objeto and similares_ids is None:
            faltantes.append("contratos con objeto similar")
        if maximo is None or maximo <= 0:
            faltantes.append("máximo de contratos")
        if pct is None:
            faltantes.append("porcentaje de experiencia")
        if not presupuesto and not por_objeto:
            faltantes.append("presupuesto")
        if por_objeto and not presupuesto:
            avisos.append("Presupuesto incompleto (cero o ausente): el mínimo de experiencia es $0; se exige al menos un contrato similar.")
        if faltantes:
            agregar("Experiencia", "Datos suficientes", "; ".join(faltantes), None)
        else:
            elegibles = [c for c in empresa["experiencia"] if c["id"] in similares_ids] if por_objeto else [
                c for c in empresa["experiencia"] if codigos & {
                    str(v).removeprefix("V1.") for v in c["codigos_unspsc"]}]
            # Un listado vacío es un incumplimiento probado, incluso con presupuesto cero.
            if not elegibles:
                agregar("Experiencia", "Al menos un contrato elegible", "Ninguno", False)
            else:
                salario_pub = 1
                convertir = smmlv_por_anio and bool(presupuesto)
                if convertir:
                    try:
                        salario_pub = SMMLV[int(str(req.get("fecha_publicacion"))[:4])]
                    except (ValueError, KeyError):
                        salario_pub = None
                valores = [(c, c["valor"] / SMMLV[c["anio"]] if convertir and c["anio"] in SMMLV
                            else c["valor"] if not convertir else None) for c in elegibles]
                if salario_pub is None or any(v is None for _, v in valores):
                    agregar("Experiencia en SMMLV", "Años incluidos en la tabla", "Año ausente o desconocido", None)
                else:
                    valores.sort(key=lambda par: par[1], reverse=True)
                    elegidos = valores[:int(maximo)]
                    suma = sum(v for _, v in elegidos)
                    minimo = (presupuesto or 0) / salario_pub * pct / 100
                    unidad = "SMMLV" if convertir else "COP"
                    agregar("Experiencia acumulada", f"≥ {minimo:,.2f} {unidad}", suma,
                            suma >= minimo, "Contratos: " + ", ".join(c["id"] for c, _ in elegidos))
                    if pct_uno is not None:
                        minimo_uno = (presupuesto or 0) / salario_pub * pct_uno / 100
                        agregar("Experiencia individual", f"≥ {minimo_uno:,.2f} {unidad}",
                                valores[0][1], valores[0][1] >= minimo_uno)
            if not smmlv_por_anio:
                avisos.append("Experiencia comparada en pesos nominales, sin conversión por año a SMMLV.")

    fallas = [c["requisito"] for c in checks if c["cumple"] is False]
    pendientes = [c["requisito"] for c in checks if c["cumple"] is None]
    if fallas:
        veredicto, resumen = "NO APTA", "No cumple: " + ", ".join(fallas) + "."
    elif pendientes:
        veredicto, resumen = "REVISAR MANUALMENTE", "No se pudo determinar: " + ", ".join(pendientes) + "."
    else:
        veredicto, resumen = "APTA", "Cumple todos los requisitos verificados."
    return resultado(veredicto, resumen, indicadores=indicadores(empresa, presupuesto))
