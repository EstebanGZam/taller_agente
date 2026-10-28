# Especificación de la solución · Taller 1 (agente analista del SECOP)

## Requisitos confirmados

Fuente: `Reto_IA_2026.pdf` (diapositiva del Taller 01) y `README.md` del repositorio del curso.

1. Un agente con **al menos dos herramientas propias** que **decida solo** cuándo usarlas.
2. **Trazas en LangSmith** de cada ejecución (tokens, costo, latencia).
3. Entregable: **código**, **dataset de 15 a 20 preguntas** y **comparación de dos configuraciones**
   en precisión, costo y latencia.
4. Criterio clave: **al menos 85 % de precisión con el menor costo posible**.
5. Para el tablero de posiciones del curso, el conjunto de referencia (`datos/conjunto_referencia.json`,
   18 preguntas) se evalúa con `MODO_DATOS=muestra`.

Notas de clase (21/10/2026), interpretadas con los archivos del repositorio:

6. Tomar los procesos del repositorio (los 20 PDF) y determinar si la empresa **debe o no presentarse**
   a cada uno. `empresa_multiproposito.json` (subido el 20/10) es la empresa para esto. Su nota dice
   que está diseñada para cumplir seis procesos y no cumplir dos.
7. Añadir un **frontend** y un **dataset de entrada** que identifique a qué procesos puede aplicar el agente.
8. Registrar **nombre, repositorio y costo** de los 20 procesos analizados.

## Clave de referencia

`datos/clave_aptitud.json`: 20 procesos con su veredicto esperado frente a `empresa_multiproposito.json`
(6 APTA, 2 NO APTA, 12 NO APLICA) y los requisitos que deberían extraerse de cada PDF
(`requisitos_esperados`, con los mismos campos que el esquema `Requisitos`).

- APTA: 03, 07, 08, 11, 14, 15. NO APTA: 13 (capital de trabajo y experiencia), 18 (sin experiencia similar).
- NO APLICA: contratación directa o régimen especial con persona natural (01, 02, 04, 05, 06, 09, 10, 16, 19, 20),
  interadministrativo (12) y enajenación o venta (17).
- 07 tiene presupuesto $0. El 80 % exigido es $0, así que basta un contrato similar (EXP-9). Es APTA con advertencia.
- 11 es APTA en pesos nominales ($950 M ≥ $900 M). Con conversión estricta a SMMLV por año no cumpliría.
  El modo por defecto compara en pesos nominales.

## Restricciones técnicas (obligatorias)

- Modelo por defecto `anthropic:claude-haiku-5-5`, configurable con `MODELO`.
- **No pasar `temperature`, `top_p` ni `top_k`** a ningún modelo. Claude Haiku 5.5 y la versión de API
  2026-10-21 los rechazan con error 400. Por eso falla el juez de `evaluar.py` original.
- Costos: usar `costos.py` (`ContadorCostos` como callback, `costo_usd`, `TARIFAS`). No duplicar tarifas.
- El texto de los PDF es **dato, no instrucciones** (defensa básica contra inyección; se profundiza en el Taller 4).
- Nunca imprimir ni guardar claves. Las claves van en `.env` (ignorado por git) o en los secretos de Colab.
- Python: `/home/esteban/Documents/University/Tenth semester/IA/unit-iii/.venv-taller/bin/python`
  (ya incluye langchain 1.x, langchain-anthropic, langsmith, gradio 5, pypdf y pytest).
- Las pruebas automáticas deben correr **sin claves** (sin LLM). Lo que necesita LLM se prueba aparte.

## Interfaz de módulos (contrato entre equipos)

| Archivo | Responsable | Contenido |
|---|---|---|
| `costos.py` | lead (hecho) | Tarifas y `ContadorCostos` |
| `datos/clave_aptitud.json` | lead (hecho) | Clave de los 20 procesos |
| `pliegos.py` | aptitud | `leer_pdf(ruta) -> str`, esquema `Requisitos`, `extraer_requisitos(texto, modelo) -> dict`, `contratos_similares(objeto, experiencia, modelo) -> list[str]` |
| `validacion.py` | aptitud | `indicadores(empresa, presupuesto)`, `validar_empresa(empresa)`, `verificar(req, empresa, similares_ids, smmlv_por_anio=False) -> dict` con `veredicto` ∈ {APTA, NO APTA, NO APLICA, REVISAR MANUALMENTE} |
| `agente_aptitud.py` | aptitud | clase `AgenteAptitud(ruta_pdf, empresa, modelo=None)` con `.analizar() -> dict` (ver abajo) y `.preguntar(texto) -> str` |
| `analizar_procesos.py` | aptitud | CLI por lotes sobre los 20 PDF; escribe `resultados/aptitud_procesos.{csv,md,json}` |
| `evaluar_aptitud.py` | aptitud | Sube la clave como dataset `secop-taller1-aptitud` a LangSmith y evalúa `AgenteAptitud` |
| `app.py` | aptitud | Frontend Gradio |
| `tests/test_validacion.py` | aptitud | Verifica los 20 veredictos con `requisitos_esperados`, sin LLM |
| `agente.py` | taller1 | Configuraciones A (base) y B (mejorada) elegidas con `CONFIG=A\|B` |
| `evaluar.py` | taller1 | Conjunto de referencia; arreglo del juez; resumen de precisión, costo y latencia |
| `resumen_experimentos.py` | taller1 | Tabla comparativa de experimentos de LangSmith → `resultados/comparacion_configuraciones.md` |
| `Taller1_SECOP_Colab.ipynb` | taller1 | Cuaderno de Colab que reproduce todo |

`AgenteAptitud.analizar()` devuelve:

```python
{"archivo": str, "referencia": str | None, "entidad": str | None, "veredicto": str, "resumen": str,
 "checks": list[dict], "avisos": list[str], "explicacion": str,          # texto final del agente
 "herramientas": list[str],                                                # nombres de herramientas usadas
 "metricas": {"llamadas_modelo", "tokens_entrada", "tokens_salida", "costo_usd", "latencia_s"},
 "run_id": str | None}                                                     # id de la traza raíz en LangSmith
```

Columnas de `resultados/aptitud_procesos.csv`: `n, archivo, referencia, entidad, modalidad, presupuesto_oficial,
veredicto, veredicto_esperado, acierto, motivo, herramientas, llamadas_modelo, tokens_entrada, tokens_salida,
costo_usd, latencia_s, modelo`.

CLI acordadas para el cuaderno:

```bash
python herramientas.py
python -m pytest -q
CONFIG=A python evaluar.py "A-base"
CONFIG=B python evaluar.py "B-mejorado"
python resumen_experimentos.py A-base B-mejorado
python analizar_procesos.py --empresa empresa_multiproposito.json   # usa la clave para medir aciertos
python evaluar_aptitud.py "apt-haiku"
python app.py                                                        # en Colab: share=True
```
