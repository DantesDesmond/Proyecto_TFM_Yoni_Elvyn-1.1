from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from services.inference import predict_texts

BASE_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="TFM Toxicidad", layout="wide")

st.title("Detección de toxicidad en comentarios en español")
st.caption("Aplicación local basada en BETO para clasificación binaria de toxicidad.")

tab1, tab2 = st.tabs(["Predicción individual", "Predicción por CSV"])

with tab1:
    st.subheader("Clasificación de texto individual")

    text_input = st.text_area(
        "Introduce un comentario",
        height=150,
        placeholder="Escribe aquí un comentario en español..."
    )

    if st.button("Clasificar texto"):
        if not text_input.strip():
            st.warning("Introduce un texto antes de clasificar.")
        else:
            result = predict_texts([text_input])[0]

            st.success("Predicción completada")
            st.write(f"**Clase predicha:** {result['pred_class']}")
            st.write(f"**Confianza:** {result['score_confianza']:.4f}")
            st.write(f"**Score no tóxico:** {result['score_no_toxico']:.4f}")
            st.write(f"**Score tóxico:** {result['score_toxico']:.4f}")

with tab2:
    st.subheader("Clasificación masiva desde CSV")

    uploaded_file = st.file_uploader("Carga un CSV", type=["csv"])

    st.info("El archivo debe contener una columna llamada `text` o `text_clean`.")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Vista previa del archivo:")
        st.dataframe(df.head())

        candidate_cols = [c for c in ["text", "text_clean"] if c in df.columns]

        if not candidate_cols:
            st.error("El CSV no contiene una columna `text` ni `text_clean`.")
        else:
            selected_col = st.selectbox("Selecciona la columna de texto", candidate_cols)

            if st.button("Clasificar CSV"):
                texts = df[selected_col].fillna("").astype(str).tolist()
                results = predict_texts(texts)

                results_df = pd.DataFrame(results)
                merged_df = pd.concat([df.reset_index(drop=True), results_df.drop(columns=["text"])], axis=1)

                st.success("Clasificación completada")
                st.dataframe(merged_df.head(20))

                csv_bytes = merged_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="Descargar resultados",
                    data=csv_bytes,
                    file_name="predicciones_toxicidad.csv",
                    mime="text/csv"
                )