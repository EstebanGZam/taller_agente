# Taller 1 · Agente analista del SECOP: insumos

## Contenido

| Archivo | Para qué sirve |
|---|---|
| `datos/empresa.json` | Perfil de la empresa ficticia: RUP, códigos UNSPSC, estados financieros a 2025 y cuatro contratos de experiencia. |
| `datos/secop_muestra.json` | Los 20 procesos de SECOP II que corresponden a los 20 PDF, con los nombres de campo de la API real. |
| `datos/conjunto_referencia.json` | 18 preguntas comunes con su respuesta esperada y las herramientas que debería usar el agente. Es el conjunto del tablero de posiciones. |
| `herramientas.py` | Las tres herramientas en Python puro: `buscar_procesos`, `consultar_empresa` y `calcular_indicadores`. |
| `agente.py` | Agente base con LangChain, listo para ejecutar y mejorar. |
| `evaluar.py` | Sube el conjunto de referencia a LangSmith y evalúa el agente con dos evaluadores. |
| `.env.example` | Plantilla de variables de entorno (claves, modelo, modo de datos). |

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # y completen sus claves
python herramientas.py      # prueba las herramientas sin modelo de lenguaje
python agente.py "¿Qué procesos de licitación pública hay?"
python evaluar.py "A-modelo-pequeno"
```

Luego cambien el modelo (`MODELO` en `.env`) o el prompt, ejecuten `python evaluar.py "B-..."` y comparen los dos experimentos en LangSmith.

## Modos de datos

- `MODO_DATOS=muestra` (por defecto): usa los 20 procesos locales. Es obligatorio para el conjunto de referencia, porque así todos los equipos se evalúan sobre los mismos datos.
- `MODO_DATOS=api`: consulta datos.gov.co en vivo. Úsenlo para su propio conjunto de preguntas.

## Qué deben mejorar los equipos

El agente base funciona, pero es deliberadamente simple. Hay espacio para mejorar las descripciones de las herramientas, el prompt del sistema, el manejo de datos incompletos y la elección del modelo. La meta es al menos 85 % de precisión con el menor costo posible.
