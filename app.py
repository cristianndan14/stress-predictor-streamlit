"""
🧠 Predictor de Estrés Estudiantil
Deploy de un modelo de Regresión Polinomial (grado 3) + Ridge (alpha=100).

Flujo de predicción:
    1. Se construye el vector de features en el ORDEN exacto esperado por el modelo.
    2. Se escala con `scaler.transform()` (StandardScaler entrenado sobre los datos crudos).
    3. Se llama a `pipeline.predict()` — el pipeline aplica internamente
       PolynomialFeatures(degree=3) + Ridge.
    4. El valor predicho se clampea al rango [0, 2].
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------- #
# Configuración general
# ----------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Predictor de Estrés Estudiantil",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, "model", "modelo_estres.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Orden EXACTO de features esperado por el scaler y el pipeline.
# ⚠️ No modificar este orden: el array de predicción debe respetarlo.
FEATURE_ORDER = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
]

# Etiquetas legibles (para sliders y gráficos).
FEATURE_LABELS = {
    "anxiety_level": "Nivel de ansiedad",
    "self_esteem": "Autoestima",
    "mental_health_history": "Historial de salud mental",
    "depression": "Depresión",
    "headache": "Dolores de cabeza",
    "blood_pressure": "Presión arterial",
    "sleep_quality": "Calidad del sueño",
    "breathing_problem": "Problemas respiratorios",
    "noise_level": "Nivel de ruido",
    "living_conditions": "Condiciones de vivienda",
    "safety": "Seguridad",
    "basic_needs": "Necesidades básicas",
    "academic_performance": "Rendimiento académico",
    "study_load": "Carga de estudio",
    "teacher_student_relationship": "Relación docente-alumno",
    "future_career_concerns": "Preocupación por el futuro",
    "social_support": "Apoyo social",
    "peer_pressure": "Presión de pares",
    "extracurricular_activities": "Actividades extracurriculares",
    "bullying": "Acoso (bullying)",
}

# Categorías de inputs: (emoji + título, [(feature, min, max, default), ...])
CATEGORIAS = [
    (
        "🧠 Factores psicológicos",
        [
            ("anxiety_level", 0, 21, 10),
            ("self_esteem", 0, 30, 15),
            ("mental_health_history", 0, 1, 0),
            ("depression", 0, 27, 10),
        ],
    ),
    (
        "🫀 Factores fisiológicos",
        [
            ("headache", 0, 5, 2),
            ("blood_pressure", 1, 3, 2),
            ("sleep_quality", 0, 5, 3),
            ("breathing_problem", 0, 5, 2),
        ],
    ),
    (
        "🌍 Factores ambientales",
        [
            ("noise_level", 0, 5, 2),
            ("living_conditions", 0, 5, 3),
            ("safety", 0, 5, 3),
            ("basic_needs", 0, 5, 3),
        ],
    ),
    (
        "📚 Factores académicos",
        [
            ("academic_performance", 0, 5, 3),
            ("study_load", 0, 5, 3),
            ("teacher_student_relationship", 0, 5, 3),
            ("future_career_concerns", 0, 5, 3),
        ],
    ),
    (
        "👥 Factores sociales",
        [
            ("social_support", 0, 3, 2),
            ("peer_pressure", 0, 5, 2),
            ("extracurricular_activities", 0, 5, 2),
            ("bullying", 0, 5, 1),
        ],
    ),
]

# Importancia de variables (coeficientes del modelo lineal base, Ridge sin poly).
IMPORTANCIA = {
    "noise_level": 0.091,
    "bullying": 0.087,
    "extracurricular_activities": 0.086,
    "study_load": 0.067,
    "depression": 0.052,
    "self_esteem": -0.092,
    "basic_needs": -0.083,
    "safety": -0.069,
    "academic_performance": -0.061,
    "sleep_quality": -0.047,
}


# ----------------------------------------------------------------------------- #
# Carga del modelo (una sola vez)
# ----------------------------------------------------------------------------- #
@st.cache_resource
def cargar_modelo():
    """Carga el pipeline y el scaler. Devuelve (modelo, scaler) o (None, None)."""
    if not (os.path.exists(MODELO_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    modelo = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)
    return modelo, scaler


def predecir(modelo, scaler, valores: dict) -> float:
    """Construye el vector en el orden correcto, escala, predice y clampea a [0, 2]."""
    x = np.array([[valores[f] for f in FEATURE_ORDER]], dtype=float)
    x_scaled = scaler.transform(x)
    pred = float(modelo.predict(x_scaled)[0])
    return float(np.clip(pred, 0.0, 2.0))


def clasificar(score: float):
    """Devuelve (nivel, emoji, color_hex) según el umbral del score."""
    if score < 0.5:
        return "BAJO", "🟢", "#22c55e"
    if score < 1.5:
        return "MEDIO", "🟡", "#eab308"
    return "ALTO", "🔴", "#ef4444"


# ----------------------------------------------------------------------------- #
# App
# ----------------------------------------------------------------------------- #
modelo, scaler = cargar_modelo()

st.title("🧠 Predictor de Nivel de Estrés Estudiantil")
st.caption(
    "Regresión Polinomial (grado 3) + Ridge (α=100) · "
    "R² = 0.77 · MSE = 0.1548 · 880 estudiantes"
)

# Sidebar con info del modelo
with st.sidebar:
    st.header("ℹ️ Sobre el modelo")
    st.metric("R²", "0.77")
    st.metric("MSE", "0.1548")
    st.metric("N.º de features", str(len(FEATURE_ORDER)))
    st.markdown(
        """
        **Tipo:** Regresión Polinomial grado 3 + Ridge (α=100)
        **Objetivo:** `stress_level` → 0 = Bajo · 1 = Medio · 2 = Alto
        **Entrenamiento:** 880 estudiantes (split 80/20)
        """
    )
    if modelo is None:
        st.error("⚠️ No se encontraron los archivos del modelo en `model/`.")
    else:
        st.success("✅ Modelo cargado correctamente.")

tab1, tab2, tab3 = st.tabs(
    ["🔮 Predictor", "📊 Análisis EDA", "📈 Métricas del modelo"]
)

# ----------------------------------------------------------------------------- #
# TAB 1 — Predictor
# ----------------------------------------------------------------------------- #
with tab1:
    st.subheader("Ajustá los factores del estudiante")
    st.caption("Mové los sliders y presioná **Predecir** para estimar el nivel de estrés.")

    valores = {}
    for titulo, features in CATEGORIAS:
        st.markdown(f"#### {titulo}")
        cols = st.columns(2)
        for i, (feature, vmin, vmax, vdef) in enumerate(features):
            with cols[i % 2]:
                valores[feature] = st.slider(
                    FEATURE_LABELS[feature],
                    min_value=vmin,
                    max_value=vmax,
                    value=vdef,
                    key=f"slider_{feature}",
                )
        st.divider()

    if st.button("🔮 Predecir nivel de estrés", type="primary", width="stretch"):
        if modelo is None or scaler is None:
            st.error(
                "No se puede predecir: faltan `model/modelo_estres.pkl` "
                "y/o `model/scaler.pkl`."
            )
        else:
            score = predecir(modelo, scaler, valores)
            nivel, emoji, color = clasificar(score)

            st.markdown(
                f"""
                <div style="
                    background-color:{color};
                    padding:28px;
                    border-radius:14px;
                    text-align:center;
                    color:white;
                    margin-top:10px;">
                    <div style="font-size:54px; line-height:1;">{emoji}</div>
                    <div style="font-size:30px; font-weight:700; margin-top:8px;">
                        ESTRÉS {nivel}
                    </div>
                    <div style="font-size:18px; margin-top:6px; opacity:0.9;">
                        Score predicho: {score:.3f} / 2.0
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(score / 2.0)

# ----------------------------------------------------------------------------- #
# TAB 2 — Análisis EDA
# ----------------------------------------------------------------------------- #
with tab2:
    st.subheader("📊 Importancia de las variables")
    st.caption(
        "Coeficientes del modelo lineal base (Ridge sin polinomialización). "
        "🔴 aumentan el estrés · 🟢 lo reducen."
    )

    df_imp = (
        pd.DataFrame(
            {
                "Variable": [FEATURE_LABELS.get(k, k) for k in IMPORTANCIA],
                "Coeficiente": list(IMPORTANCIA.values()),
            }
        )
        .sort_values("Coeficiente")
        .reset_index(drop=True)
    )
    df_imp["Efecto"] = np.where(
        df_imp["Coeficiente"] >= 0, "Aumenta estrés", "Reduce estrés"
    )

    fig = px.bar(
        df_imp,
        x="Coeficiente",
        y="Variable",
        orientation="h",
        color="Efecto",
        color_discrete_map={
            "Aumenta estrés": "#ef4444",
            "Reduce estrés": "#22c55e",
        },
        text="Coeficiente",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        height=520,
        yaxis_title="",
        xaxis_title="Coeficiente (impacto en el estrés)",
        legend_title="",
        margin=dict(l=10, r=30, t=20, b=10),
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 🔴 Top 5 — aumentan el estrés")
        top_up = df_imp[df_imp["Coeficiente"] >= 0].sort_values(
            "Coeficiente", ascending=False
        )
        st.dataframe(top_up[["Variable", "Coeficiente"]], hide_index=True,
                     width="stretch")
    with col_b:
        st.markdown("##### 🟢 Top 5 — reducen el estrés")
        top_down = df_imp[df_imp["Coeficiente"] < 0].sort_values("Coeficiente")
        st.dataframe(top_down[["Variable", "Coeficiente"]], hide_index=True,
                     width="stretch")

# ----------------------------------------------------------------------------- #
# TAB 3 — Métricas del modelo
# ----------------------------------------------------------------------------- #
with tab3:
    st.subheader("📈 Desempeño del modelo")

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", "0.77")
    c2.metric("MSE", "0.1548")
    c3.metric("Features", str(len(FEATURE_ORDER)))

    st.markdown("#### Comparativa de modelos evaluados")
    df_modelos = pd.DataFrame(
        {
            "Modelo": [
                "Polinomial simple",
                "Ridge grado 2",
                "Ridge grado 3 (final)",
            ],
            "R²": [0.24, 0.50, 0.77],
        }
    )
    st.dataframe(df_modelos, hide_index=True, width="stretch")

    fig_m = px.bar(
        df_modelos,
        x="Modelo",
        y="R²",
        text="R²",
        color="Modelo",
        color_discrete_sequence=["#94a3b8", "#60a5fa", "#22c55e"],
    )
    fig_m.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_m.update_layout(
        height=420,
        showlegend=False,
        yaxis_title="R²",
        xaxis_title="",
        yaxis_range=[0, 1],
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_m, width="stretch")

    st.info(
        "El modelo final (Ridge grado 3, α=100) explica el **77%** de la "
        "varianza del nivel de estrés, superando ampliamente a los modelos "
        "más simples."
    )
