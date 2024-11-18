import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.title("Input Data")

# Upload dataset
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Menyimpan data di session_state agar tetap ada selama sesi
    if 'data' not in st.session_state:
        st.session_state.data = data

    # Menampilkan data awal
    st.write("Data Awal:", st.session_state.data.head())

    # Label Encoding untuk kolom kategori (contoh: Heart Disease)
    st.subheader("Preprocessing Data")
    label_encoder = LabelEncoder()
    if 'Heart Disease' in st.session_state.data.columns:
        st.session_state.data['Heart Disease'] = label_encoder.fit_transform(st.session_state.data['Heart Disease'])

    # Pilihan untuk Normalisasi
    st.write("Pilih kolom untuk normalisasi:")
    numeric_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns
    cols_to_normalize = st.multiselect("Kolom Fitur Numerik:", numeric_cols)

    # Normalisasi hanya jika ada kolom yang dipilih
    if cols_to_normalize:
        st.write(f"Melakukan normalisasi pada kolom: {cols_to_normalize}")
        scaler = MinMaxScaler()
        st.session_state.data[cols_to_normalize] = scaler.fit_transform(st.session_state.data[cols_to_normalize])
        st.write("Data setelah normalisasi:", st.session_state.data.head())
    else:
        st.write("Tidak ada kolom yang dipilih untuk normalisasi.")

    # Cek dan hapus nilai NaN pada kolom target ('Heart Disease') atau kolom lainnya
    if st.session_state.data.isnull().sum().any():
        st.warning("Data mengandung nilai NaN. Melakukan pembersihan data.")
        # Hapus baris yang memiliki NaN
        st.session_state.data = st.session_state.data.dropna()
        st.write("Data setelah pembersihan NaN:", st.session_state.data.head())

    # Simpan data yang telah diproses di session_state
    st.session_state.preprocessed_data = st.session_state.data

    # Menampilkan data setelah preprocessing (tampilan dataset final)
    st.subheader("Data setelah Preprocessing")
    st.dataframe(st.session_state.preprocessed_data)
