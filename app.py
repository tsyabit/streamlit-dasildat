import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Tubes Dasildat", layout="wide")

st.title("Prediksi Perkiraan Ongkos Kirim")
st.write("Dataset: Indonesia E-Commerce Sales")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

model_pilihan = st.sidebar.selectbox(
    "Pilih Model",
    ["KNN", "Decision Tree", "SVM", "Neural Network"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=";")

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    df = df.drop_duplicates()

    label_df = df["Perkiraan Ongkos Kirim"]

    fitur_df = df[[
        "order_id",
        "total_qty",
        "product_categories",
        "Metode Pembayaran",
        "Kota/Kabupaten",
        "Provinsi",
        "Waktu Pesanan Dibuat",
        "order_date"
    ]].copy()

    encoder = LabelEncoder()

    for col in fitur_df.columns:
        fitur_df[col] = encoder.fit_transform(fitur_df[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fitur_df)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled,
        label_df,
        test_size=0.3,
        random_state=42
    )

    st.subheader(f"Model yang Dipilih: {model_pilihan}")

    if model_pilihan == "KNN":
        k = st.sidebar.slider("Nilai K", 1, 20, 5)
        model = KNeighborsRegressor(
            n_neighbors=k,
            metric="manhattan"
        )

    elif model_pilihan == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
        model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=max_depth,
            random_state=42
        )

    elif model_pilihan == "SVM":
        c_value = st.sidebar.slider("Nilai C", 0.1, 10.0, 1.0)
        model = SVR(
            kernel="rbf",
            C=c_value
        )

    else:
        neuron = st.sidebar.slider("Jumlah Neuron", 10, 200, 100)
        model = MLPRegressor(
            hidden_layer_sizes=(neuron,),
            max_iter=500,
            random_state=42
        )

    model.fit(X_train, Y_train)

    hasil_prediksi = model.predict(X_test)

    mae = mean_absolute_error(Y_test, hasil_prediksi)
    mse = mean_squared_error(Y_test, hasil_prediksi)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, hasil_prediksi)

    st.subheader("Hasil Evaluasi Model")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("R²", f"{r2:.4f}")

    st.subheader("Tabel Hasil Prediksi")

    hasil_df = pd.DataFrame({
        "Label Asli": Y_test.values,
        "Prediksi": hasil_prediksi
    })

    st.dataframe(hasil_df)

else:
    st.info("Silakan upload dataset CSV terlebih dahulu.")