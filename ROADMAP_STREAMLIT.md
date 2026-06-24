# 🧠 Roadmap — Deploy del Predictor de Estrés Estudiantil en Streamlit

> **Proyecto:** Predicción de Niveles de Estrés Estudiantil  
> **Modelo:** Regresión Polinomial grado 3 + Ridge (alpha=100) — R²=0.77  
> **Repositorio:** https://github.com/cristianndan14/stress-predictor-streamlit.git  
> **Objetivo:** Deployar el modelo entrenado en Streamlit Cloud usando Claude Code para automatizar todo el proceso.

---

## ~~Fase 1 — Exportar el modelo entrenado~~ ✅ COMPLETADA

El pipeline completo y el scaler fueron serializados desde el notebook de Google Colab.

- [x] Serializar el pipeline completo con `joblib.dump()` → `model/modelo_estres.pkl`
- [x] Exportar el scaler por separado con `joblib.dump()` → `model/scaler.pkl`
- [x] Verificar la carga con `joblib.load()` y hacer una predicción de prueba

---

## ~~Fase 2 — Estructura del proyecto~~ ✅ COMPLETADA

Repositorio creado, remoto linkeado y modelos pusheados.

- [x] **Crear el repositorio en GitHub:** https://github.com/cristianndan14/stress-predictor-streamlit.git
- [x] **Linkear remoto** con `git remote add origin`
- [x] **Subir los archivos `.pkl`** a `model/` y pushear al remoto

### Estado actual del repositorio

```
stress-predictor-streamlit/
│
└── model/                  # ✅ Pusheado al remoto
    ├── modelo_estres.pkl   # Pipeline: StandardScaler + PolynomialFeatures(degree=3) + Ridge(alpha=100)
    └── scaler.pkl          # StandardScaler para escalar inputs
```

### Pendiente para Claude Code

```
stress-predictor-streamlit/
│
├── app.py                  # ⬅ A crear
├── requirements.txt        # ⬅ A crear
│
└── model/                  # ✅ Ya existe
    ├── modelo_estres.pkl
    └── scaler.pkl
```

---

## Fase 3 — Construir la app en Streamlit 🤖 PARA CLAUDE CODE

La app replica la funcionalidad del widget interactivo del notebook (`ipywidgets`) usando componentes nativos de Streamlit.

### 3.1 Carga del modelo

Usar `@st.cache_resource` para cargar el modelo y el scaler una sola vez. Manejar errores si los archivos `.pkl` no se encuentran.

```python
import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('model/modelo_estres.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return modelo, scaler
```

### 3.2 Inputs con sliders agrupados por categoría

Usar `st.columns()` para mostrar los sliders en pares (2 por fila). Grupos y rangos exactos:

| Categoría | Variable | Min | Max | Default |
|---|---|---|---|---|
| 🧠 Psicológicos | `anxiety_level` | 0 | 21 | 10 |
| 🧠 Psicológicos | `self_esteem` | 0 | 30 | 15 |
| 🧠 Psicológicos | `mental_health_history` | 0 | 1 | 0 |
| 🧠 Psicológicos | `depression` | 0 | 27 | 10 |
| 🫀 Fisiológicos | `headache` | 0 | 5 | 2 |
| 🫀 Fisiológicos | `blood_pressure` | 1 | 3 | 2 |
| 🫀 Fisiológicos | `sleep_quality` | 0 | 5 | 3 |
| 🫀 Fisiológicos | `breathing_problem` | 0 | 5 | 2 |
| 🌍 Ambientales | `noise_level` | 0 | 5 | 2 |
| 🌍 Ambientales | `living_conditions` | 0 | 5 | 3 |
| 🌍 Ambientales | `safety` | 0 | 5 | 3 |
| 🌍 Ambientales | `basic_needs` | 0 | 5 | 3 |
| 📚 Académicos | `academic_performance` | 0 | 5 | 3 |
| 📚 Académicos | `study_load` | 0 | 5 | 3 |
| 📚 Académicos | `teacher_student_relationship` | 0 | 5 | 3 |
| 📚 Académicos | `future_career_concerns` | 0 | 5 | 3 |
| 👥 Sociales | `social_support` | 0 | 3 | 2 |
| 👥 Sociales | `peer_pressure` | 0 | 5 | 2 |
| 👥 Sociales | `extracurricular_activities` | 0 | 5 | 2 |
| 👥 Sociales | `bullying` | 0 | 5 | 1 |

> ⚠️ El orden de las columnas al armar el array de predicción debe respetar exactamente este orden: `anxiety_level`, `self_esteem`, `mental_health_history`, `depression`, `headache`, `blood_pressure`, `sleep_quality`, `breathing_problem`, `noise_level`, `living_conditions`, `safety`, `basic_needs`, `academic_performance`, `study_load`, `teacher_student_relationship`, `future_career_concerns`, `social_support`, `peer_pressure`, `extracurricular_activities`, `bullying`.

### 3.3 Output visual de la predicción

- Botón **"Predecir nivel de estrés"** que dispara la predicción
- Clampear el valor predicho al rango [0, 2]
- Mostrar resultado con badge de color según umbral:
  - 🟢 **BAJO** — score < 0.5
  - 🟡 **MEDIO** — 0.5 ≤ score < 1.5
  - 🔴 **ALTO** — score ≥ 1.5

### 3.4 Tabs de navegación

```python
tab1, tab2, tab3 = st.tabs(["🔮 Predictor", "📊 Análisis EDA", "📈 Métricas del modelo"])
```

- **Tab 1 — Predictor:** Los sliders agrupados por categoría y el resultado de la predicción
- **Tab 2 — Análisis EDA:** Distribución del nivel de estrés, mapa de correlación, gráfico de importancia de variables
- **Tab 3 — Métricas del modelo:** Tabla comparativa de modelos, gráfico Real vs Predicho, distribución de errores

### 3.5 Gráfico de importancia de variables

Mostrar los coeficientes del modelo lineal base (Ridge sin polinomialización). Colores: rojo para variables que **aumentan** el estrés, verde para las que lo **reducen**. Usar `plotly.express`.

**Top 5 que aumentan el estrés:** `noise_level` (0.091), `bullying` (0.087), `extracurricular_activities` (0.086), `study_load` (0.067), `depression` (0.052)

**Top 5 que reducen el estrés:** `self_esteem` (-0.092), `basic_needs` (-0.083), `safety` (-0.069), `academic_performance` (-0.061), `sleep_quality` (-0.047)

---

## Fase 4 — Deploy en Streamlit Cloud 🤖 PARA CLAUDE CODE

Una vez que `app.py` y `requirements.txt` estén creados y commiteados, Claude Code debe:

1. Hacer commit y push de todos los archivos nuevos al remoto
2. Indicar los pasos para hacer el deploy manual en [share.streamlit.io](https://share.streamlit.io):
   - Repository: `cristianndan14/stress-predictor-streamlit`
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: `3.11`

### `requirements.txt` esperado

```
streamlit>=1.32.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
joblib>=1.3.0
plotly>=5.19.0
```

### Errores comunes a anticipar

- Versión de `scikit-learn` incompatible con el `.pkl` → fijar versión exacta en `requirements.txt`
- Archivo `.pkl` no encontrado → verificar rutas relativas desde la raíz del repo
- Falta de memoria → reducir `degree` o usar `sparse=True` en `PolynomialFeatures`

---

## Fase 5 — Mejoras opcionales

Una vez que la app base funciona:

- [ ] Gráfico interactivo de violín por nivel de estrés para las 6 variables clave (Plotly)
- [ ] Scatter plot Real vs Predicho con hover info (Plotly)
- [ ] CSS custom con `st.markdown()` para replicar la estética oscura del notebook
- [ ] Sidebar con info del modelo: R²=0.77, MSE=0.1548, dataset: 1.100 estudiantes
- [ ] `st.metric()` para R², MSE y número de features

**Comparativa de modelos para mostrar en la app:**

| Modelo | R² |
|---|---|
| Polinomial simple | 0.24 |
| Ridge grado 2 | 0.50 |
| **Ridge grado 3 (final)** | **0.77** |

---

## 🤖 Prompt para Claude Code

Copiar y pegar este prompt al iniciar Claude Code en la carpeta local del proyecto:

```
Tengo un proyecto de Streamlit para deployar un modelo de predicción de estrés estudiantil.

CONTEXTO DEL MODELO:
- Tipo: Regresión Polinomial grado 3 + Ridge (alpha=100), implementado como sklearn Pipeline
- Variable objetivo: stress_level (0=Bajo, 1=Medio, 2=Alto)
- R²=0.77, MSE=0.1548, entrenado con 880 estudiantes (80/20 split)
- Los archivos ya están en el repo: model/modelo_estres.pkl y model/scaler.pkl
- El pipeline incluye internamente el PolynomialFeatures, por lo que al predecir solo hay que escalar con el scaler y luego llamar al pipeline

REPOSITORIO:
- URL: https://github.com/cristianndan14/stress-predictor-streamlit.git
- Ya tiene la carpeta model/ con los .pkl pusheados

TAREAS:
1. Crear requirements.txt con: streamlit>=1.32.0, scikit-learn>=1.4.0, pandas>=2.0.0, numpy>=1.26.0, joblib>=1.3.0, plotly>=5.19.0
2. Construir app.py completo con:
   a. @st.cache_resource para cargar modelo y scaler
   b. 3 tabs: "🔮 Predictor" / "📊 Análisis EDA" / "📈 Métricas del modelo"
   c. Tab Predictor: sliders agrupados en 5 categorías (ver tabla de variables abajo), botón de predicción, resultado con badge de color (🟢 BAJO / 🟡 MEDIO / 🔴 ALTO según score < 0.5 / < 1.5 / >= 1.5)
   d. Tab EDA: gráfico de importancia de variables con Plotly (rojo = aumenta estrés, verde = reduce)
   e. Tab Métricas: tabla comparativa de modelos y métricas R²=0.77, MSE=0.1548
3. Hacer commit y push de todos los archivos al remoto

VARIABLES Y RANGOS (respetar este orden exacto en el array de predicción):
anxiety_level (0-21, default 10), self_esteem (0-30, default 15), mental_health_history (0-1, default 0), depression (0-27, default 10),
headache (0-5, default 2), blood_pressure (1-3, default 2), sleep_quality (0-5, default 3), breathing_problem (0-5, default 2),
noise_level (0-5, default 2), living_conditions (0-5, default 3), safety (0-5, default 3), basic_needs (0-5, default 3),
academic_performance (0-5, default 3), study_load (0-5, default 3), teacher_student_relationship (0-5, default 3), future_career_concerns (0-5, default 3),
social_support (0-3, default 2), peer_pressure (0-5, default 2), extracurricular_activities (0-5, default 2), bullying (0-5, default 1)

IMPORTANCIA DE VARIABLES (para el gráfico del Tab EDA):
Aumentan estrés: noise_level (0.091), bullying (0.087), extracurricular_activities (0.086), study_load (0.067), depression (0.052)
Reducen estrés: self_esteem (-0.092), basic_needs (-0.083), safety (-0.069), academic_performance (-0.061), sleep_quality (-0.047)
```

---

*Generado con Claude — Proyecto: Modelizado de Sistemas, Unidad 5*
