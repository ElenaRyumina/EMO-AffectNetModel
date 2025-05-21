import streamlit as st
import os
import pandas as pd
import tempfile
import numpy as np
from scipy import stats
import time

# Importar tus funciones
from functions import sequences
from functions import get_face_areas
from functions.get_models import load_weights_EE, load_weights_LSTM

# Configuración inicial de Streamlit
st.set_page_config(page_title="Reconocimiento Emocional", layout="centered")
st.title("🎭 Reconocimiento de emociones en video")

# Subir video
uploaded_video = st.file_uploader("Sube un video (formato .mp4)", type=["mp4"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.video(uploaded_video)

    # Parámetros predefinidos (puedes cambiarlos según tu necesidad)
    conf_d = 0.7
    path_FE_model = 'models/EmoAffectnet/weights_0_66_37_wo_gl.h5'
    path_LSTM_model = 'models/LSTM/RAVDESS_with_config.h5'

    if st.button("📊 Analizar emociones"):
        with st.spinner("Procesando el video..."):

            start_time = time.time()
            label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']

            detect = get_face_areas.VideoCamera(path_video=video_path, conf=conf_d)
            dict_face_areas, total_frame = detect.get_frame()
            name_frames = list(dict_face_areas.keys())
            face_areas = list(dict_face_areas.values())

            EE_model = load_weights_EE(path_FE_model)
            LSTM_model = load_weights_LSTM(path_LSTM_model)
            features = EE_model(np.stack(face_areas))
            seq_paths, seq_features = sequences.sequences(name_frames, features)
            pred = LSTM_model(np.stack(seq_features)).numpy()

            all_pred = []
            all_path = []
            for id, c_p in enumerate(seq_paths):
                c_f = [str(i).zfill(6) for i in range(int(c_p[0]), int(c_p[-1]) + 1)]
                c_pr = [pred[id]] * len(c_f)
                all_pred.extend(c_pr)
                all_path.extend(c_f)
            m_f = [str(i).zfill(6) for i in range(int(all_path[-1]) + 1, total_frame + 1)]
            m_p = [all_pred[-1]] * len(m_f)

            df = pd.DataFrame(data=all_pred + m_p, columns=label_model)
            df['frame'] = all_path + m_f
            df = df[['frame'] + label_model]
            df = sequences.df_group(df, label_model)

            end_time = time.time() - start_time
            mode = stats.mode(np.argmax(pred, axis=1), keepdims=False)[0]

            st.success("✅ Video procesado con éxito")
            st.write(f"**Emoción predominante:** {label_model[mode]}")
            st.write(f"⏱️ Tiempo de procesamiento: {round(end_time, 2)} s")

            # Mostrar tabla
            st.dataframe(df.head())

            # Botón para descargar
            st.download_button(
                label="📥 Descargar resultados (.csv)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="reporte_emociones.csv",
                mime="text/csv"
            )
