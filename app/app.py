from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TFM Toxicidad", layout="wide")

st.title("App local de detección de toxicidad")
st.caption("Piloto experimental para el TFM")

st.markdown(
    """
    Esta aplicación será el frente local del proyecto.

    Módulos previstos:
    - carga de comentarios
    - predicción individual
    - predicción masiva
    - visualización de métricas
    - exportación de resultados
    """
)

sample = pd.DataFrame(
    {
        "comentario": [
            "Este video está excelente",
            "Eres un inútil, qué asco",
            "No estoy de acuerdo con tu opinión",
        ],
        "prediccion_demo": ["no_toxico", "toxico", "no_toxico"],
        "score_demo": [0.05, 0.92, 0.12],
    }
)

st.subheader("Vista preliminar")
st.dataframe(sample, use_container_width=True)

user_text = st.text_area("Probar comentario manual", placeholder="Escribe aquí un comentario...")
if st.button("Clasificar demo"):
    if user_text.strip():
        st.success("Predicción demo: pendiente de conectar al modelo final")
        st.write({"texto": user_text, "prediccion": "pendiente", "score": None})
    else:
        st.warning("Escribe un comentario antes de clasificar.")
