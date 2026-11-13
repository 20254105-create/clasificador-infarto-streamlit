import pickle
import numpy as np
import pandas as pd
import streamlit as st
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

st.set_page_config(
    page_title="Clasificador de Riesgo de Infarto",
    page_icon="🫀",
    layout="wide"
)

# Rangos observados en Medicaldataset.xlsx
RANGES = {
    "Age": {"min": 19.0, "max": 103.0, "label": "Edad (años)"},
    "Gender": {"min": 0, "max": 1, "label": "Sexo (0 = Mujer, 1 = Hombre)"},
    "Heart rate": {"min": 36.0, "max": 135.0, "label": "Frecuencia cardíaca (lpm)"},
    "Systolic blood pressure": {"min": 65.0, "max": 225.0, "label": "Presión arterial sistólica (mmHg)"},
    "Diastolic blood pressure": {"min": 38.0, "max": 128.0, "label": "Presión arterial diastólica (mmHg)"},
    "Blood sugar": {"min": 35.0, "max": 541.0, "label": "Glucosa en sangre (mg/dL)"},
    "CK-MB": {"min": 0.321, "max": 300.0, "label": "CK-MB (ng/mL)"},
    "Troponin": {"min": 0.001, "max": 10.3, "label": "Troponina (ng/mL)"},
}

REQUIRED_COLS = [
    "Age",
    "Gender",
    "Heart rate",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Blood sugar",
    "CK-MB",
    "Troponin",
]


@st.cache_resource
def load_orange_model(model_path: str = "Modelo_Infarto.pkcls"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def validate_dataframe(df: pd.DataFrame):
    errors = []
    warnings = []

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Faltan las siguientes columnas obligatorias: {', '.join(missing_cols)}.")
        return errors, warnings

    for idx, row in df.iterrows():
        fila = idx + 1

        # Verificar valores faltantes
        for col in REQUIRED_COLS:
            if pd.isna(row[col]) or str(row[col]).strip() == "":
                errors.append(f"Fila {fila}: La variable '{col}' tiene un valor faltante.")

        # Validar tipos y valores no permitidos
        for col in REQUIRED_COLS:
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                errors.append(f"Fila {fila}: La variable '{col}' ('{val}') no es numérica.")
                continue

            if col == "Gender":
                if num_val not in (0.0, 1.0):
                    errors.append(f"Fila {fila}: 'Gender' debe ser 0 (Mujer) o 1 (Hombre), se recibió {val}.")
            else:
                if num_val <= 0:
                    errors.append(f"Fila {fila}: '{col}' no puede ser cero ni negativo ({num_val}).")
                else:
                    r_min = RANGES[col]["min"]
                    r_max = RANGES[col]["max"]
                    if num_val < r_min or num_val > r_max:
                        warnings.append(
                            f"Fila {fila}: '{col}' ({num_val}) está fuera del rango observado en Medicaldataset [{r_min} - {r_max}]."
                        )

        # Consistencia clínica básica de presión arterial
        sbp = pd.to_numeric(row.get("Systolic blood pressure"), errors="coerce")
        dbp = pd.to_numeric(row.get("Diastolic blood pressure"), errors="coerce")
        if pd.notna(sbp) and pd.notna(dbp) and sbp <= dbp:
            errors.append(
                f"Fila {fila}: Inconsistencia en presión arterial: Sistólica ({sbp}) debe ser mayor que Diastólica ({dbp})."
            )

    return errors, warnings


def build_orange_table(df: pd.DataFrame) -> Table:
    # Orange convierte automáticamente el dominio de entrada al original_domain del modelo
    # usando los nombres exactos de las variables de Medicaldataset.
    attributes = [
        ContinuousVariable("Age"),
        DiscreteVariable("Gender", values=["0", "1"]),
        ContinuousVariable("Heart rate"),
        ContinuousVariable("Systolic blood pressure"),
        ContinuousVariable("Diastolic blood pressure"),
        ContinuousVariable("Blood sugar"),
        ContinuousVariable("CK-MB"),
        ContinuousVariable("Troponin"),
    ]
    domain = Domain(attributes)

    n = len(df)
    X = np.zeros((n, len(attributes)), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        X[i, 0] = float(row["Age"])
        X[i, 1] = 0.0 if int(float(row["Gender"])) == 0 else 1.0
        X[i, 2] = float(row["Heart rate"])
        X[i, 3] = float(row["Systolic blood pressure"])
        X[i, 4] = float(row["Diastolic blood pressure"])
        X[i, 5] = float(row["Blood sugar"])
        X[i, 6] = float(row["CK-MB"])
        X[i, 7] = float(row["Troponin"])

    return Table.from_numpy(domain, X)


def predict_records(model, df: pd.DataFrame) -> pd.DataFrame:
    orange_table = build_orange_table(df)
    pred_indices, probs = model(orange_table, ret=model.ValueProbs)
    class_names = list(model.domain.class_var.values)

    results_df = df.copy()
    results_df["Predicción"] = [class_names[int(idx)] for idx in pred_indices]
    for c_idx, c_name in enumerate(class_names):
        results_df[f"Prob_{c_name}"] = np.round(probs[:, c_idx], 4)

    return results_df


# --- INTERFAZ PRINCIPAL ---
st.title("🫀 Clasificador de Riesgo de Infarto")

st.warning(
    "⚠️ **AVISO ACADÉMICO:** Esta aplicación ha sido desarrollada con fines exclusivamente educativos y académicos. "
    "Los resultados mostrados son estimaciones basadas en un modelo de aprendizaje automático entrenado en Orange Data Mining "
    "y **no constituyen un diagnóstico médico profesional**."
)

try:
    model = load_orange_model("Modelo_Infarto.pkcls")
except Exception as e:
    st.error(f"No se pudo cargar el archivo del modelo 'Modelo_Infarto.pkcls': {e}")
    st.stop()

tab_manual, tab_excel = st.tabs(["📝 Ingreso Manual", "📂 Carga de Archivo Excel"])

# --- PESTAÑA 1: INGRESO MANUAL ---
with tab_manual:
    st.subheader("Ingresar variables clínicas del paciente")
    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (Edad en años)", min_value=0.0, max_value=150.0, value=60.0, step=1.0)
            gender_label = st.selectbox("Gender (Sexo)", options=["1 - Hombre", "0 - Mujer"])
            gender = int(gender_label.split(" - ")[0])
            heart_rate = st.number_input("Heart rate (Frecuencia cardíaca)", min_value=0.0, max_value=300.0, value=75.0, step=1.0)
        with col2:
            sbp = st.number_input("Systolic blood pressure (Presión sistólica)", min_value=0.0, max_value=350.0, value=120.0, step=1.0)
            dbp = st.number_input("Diastolic blood pressure (Presión diastólica)", min_value=0.0, max_value=250.0, value=80.0, step=1.0)
            blood_sugar = st.number_input("Blood sugar (Glucosa en sangre)", min_value=0.0, max_value=1000.0, value=110.0, step=1.0)
        with col3:
            ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=1000.0, value=2.50, step=0.1, format="%.3f")
            troponin = st.number_input("Troponin (ng/mL)", min_value=0.0, max_value=100.0, value=0.010, step=0.001, format="%.3f")

        submitted = st.form_submit_button("Predecir Riesgo", type="primary")

    if submitted:
        manual_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Heart rate": heart_rate,
            "Systolic blood pressure": sbp,
            "Diastolic blood pressure": dbp,
            "Blood sugar": blood_sugar,
            "CK-MB": ck_mb,
            "Troponin": troponin,
        }])

        st.markdown("#### Datos ingresados")
        st.dataframe(manual_df, use_container_width=True)

        errors, warnings = validate_dataframe(manual_df)

        for w in warnings:
            st.warning(w)

        if errors:
            for err in errors:
                st.error(err)
            st.info("Corrija los errores señalados antes de ejecutar la predicción.")
        else:
            pred_df = predict_records(model, manual_df)
            pred_class = pred_df.loc[0, "Predicción"]
            prob_neg = float(pred_df.loc[0, "Prob_negative"])
            prob_pos = float(pred_df.loc[0, "Prob_positive"])

            st.markdown("#### Resultado de la Clasificación")
            if pred_class == "positive":
                st.error(f"Clasificación estimada: **POSITIVE** (Probabilidad: {prob_pos * 100:.2f}%)")
            else:
                st.success(f"Clasificación estimada: **NEGATIVE** (Probabilidad: {prob_neg * 100:.2f}%)")

            st.markdown("#### Probabilidades estimadas por clase")
            c_bar1, c_bar2 = st.columns(2)
            with c_bar1:
                st.write(f"**negative:** {prob_neg * 100:.2f}%")
                st.progress(min(max(prob_neg, 0.0), 1.0))
            with c_bar2:
                st.write(f"**positive:** {prob_pos * 100:.2f}%")
                st.progress(min(max(prob_pos, 0.0), 1.0))

            csv_manual = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar resultado en CSV",
                data=csv_manual,
                file_name="prediccion_paciente_individual.csv",
                mime="text/csv",
            )

# --- PESTAÑA 2: CARGA DE EXCEL ---
with tab_excel:
    st.subheader("Cargar archivo Excel con registros de pacientes")
    st.markdown(
        "El archivo debe contener las columnas exactas del conjunto de entrenamiento: "
        "`Age`, `Gender`, `Heart rate`, `Systolic blood pressure`, `Diastolic blood pressure`, `Blood sugar`, `CK-MB`, `Troponin` "
        "(opcionalmente puede incluir `DNI`, el cual será excluido al descargar los resultados)."
    )

    uploaded_file = st.file_uploader("Seleccione un archivo Excel (.xlsx o .xls)", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = "Datos" if "Datos" in excel_file.sheet_names else excel_file.sheet_names[0]
            if len(excel_file.sheet_names) > 1:
                sheet_name = st.selectbox("Seleccione la hoja a procesar", excel_file.sheet_names, index=excel_file.sheet_names.index(sheet_name))
            raw_df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        except Exception as e:
            st.error(f"Error al leer el archivo Excel: {e}")
            st.stop()

        st.markdown("#### Vista previa de los datos cargados")
        st.dataframe(raw_df, use_container_width=True)

        errors, warnings = validate_dataframe(raw_df)

        if warnings:
            with st.expander(f"⚠️ Advertencias de rango ({len(warnings)})", expanded=True):
                for w in warnings:
                    st.warning(w)

        if errors:
            st.error(f"Se detectaron {len(errors)} error(es) en los datos. Corrija las filas indicadas antes de predecir:")
            for err in errors:
                st.write(f"- {err}")
        else:
            st.success("✅ Validación exitosa. Todos los registros cumplen con el formato y codificación requeridos.")
            if st.button("Ejecutar Predicciones en Lote", type="primary"):
                results_df = predict_records(model, raw_df)

                st.markdown("#### Tabla de Predicciones")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    column_config={
                        "Prob_negative": st.column_config.ProgressColumn(
                            "Probabilidad negative",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "Prob_positive": st.column_config.ProgressColumn(
                            "Probabilidad positive",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    },
                )

                # Excluir identificadores personales (como DNI) en la descarga CSV
                id_cols = [c for c in results_df.columns if c.strip().lower() in ("dni", "id", "nombre", "paciente", "identificador")]
                export_df = results_df.drop(columns=id_cols, errors="ignore")

                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Descargar predicciones en CSV (sin identificadores personales)",
                    data=csv_bytes,
                    file_name="predicciones_infarto_resultados.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption(
    "This is for informational purposes only. For medical advice or diagnosis, consult a professional. AI responses may include mistakes."
)
