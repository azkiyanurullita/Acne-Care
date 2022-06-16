# Import Libraries
import cv2
import time
import requests
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from streamlit_lottie import st_lottie

# Load model
model = tf.keras.models.load_model("model_acne.h5")

# Nama website
st.set_page_config(page_title="AcneCare", page_icon=":sparkles:", layout="wide")

# Fungsi untuk menggunakan lottifiles
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Assets
lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_molzhsbm.json")
team1 = Image.open("img/azkiya.jpg")
team2 = Image.open("img/desti.jpg")
team3 = Image.open("img/galuh.JPG")
team4 = Image.open("img/meutia.jpeg")
team5 = Image.open("img/ulfa.jpg")


# ---- HEADER SECTION ----
with st.container():
    st.subheader("Selamat datang di AcneCare! :wave:")
    st.title("Ketahui Level Jerawatmu!")
    st.write(
        "Cukup upload foto wajahmu dan ketahui seberapa parah masalah jerawat milikmu :wink:"
    )
    st.write("*Scroll down!*")

# ---- WHAT WE DO? ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What we do?")
        st.write("##")
        st.write(
            """
            Di Website ini kamu akan mengetahui level jerawatmu, dengan cara:
            - Upload foto wajah yang ingin diidentifikasi
            - Ketahui seberapa parah masalah jerawatmu dengan *click* tombol PREDICT
            - Tunggu sistem untuk menganalisis masalah jerawatmu
            - Hasil prediksi masalah jerawatmu akan keluar dengan penjelasan singkat mengenai level tersebut.
            Tingkat jerawat pada website ini dibagi menjadi tiga, yaitu Level 0, Level 1, dan Level 2. *Scroll down* untuk memulai proses prediksi.
            """
        )
    with right_column:
        st_lottie(lottie_coding, height=400, key="woman")

# ---- KNOW YOUR ACNE ----
with st.container():
    st.write("---")
    st.header("Know Your Acne")
    st.write("##")
    left_column, right_column = st.columns(2)

    # label
    map_dict = {0: 'Level 0',
                1: 'Level 1',
                2: 'Level 2'}

    with left_column:
        # Upload file foto
        uploaded_file = st.file_uploader("Choose a image file", type="jpg")
        if uploaded_file is not None:
            # Convert foto dengan opencv
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(opencv_image,(224,224))
            # Menampilkan foto
            st.image(opencv_image, channels="RGB")

            img_reshape = resized[np.newaxis,...]
    with right_column:
        # menampilkan hasil prediksi
        st.write("##")
        Genrate_pred = st.button("PREDICT")
        if Genrate_pred:
            prediction = model.predict(img_reshape).argmax()
            level = map_dict [prediction]
            st.header("Level Jerawatmu adalah {}".format(level))            
            if prediction == 0:
                st.write("Level 0 merupakan level terkecil, hal ini menunjukkan bahwa kondisi jerawat pada wajah kamu masih aman bahkan tidak ada. Jumlah jerawat yang ada sangat sedikit dan dapat diatasi dengan obat jerawat yang ada.")
            elif prediction == 1:
                st.write("Level 1 merupakan level medium, hal ini menunjukkan bahwa kondisi jerawat pada wajah kamu cukup parah. Jumlah jerawat yang ada lumayan banyak serta letak jerawat menyebar atau mengumpul dalam satu bagian wajah saja.")
                st.write("Proses penyembuhan dapat menggunakan skincare khusus masalah jerawat dan obat/spot treatment jerawat")
            elif prediction == 2:
                st.write("Level 2 merupakan level tertinggi, hal ini menunjukkan bahwa kondisi jerawat pada wajah parah. Jumlah jerawat yang ada banyak serta letak jerawat menyebar hampir diseluruh wajah.")
                st.write("Proses penyembuhan dapat menggunakan skincare khusus masalah jerawat, serta dianjurkan menghubungi dokter spesialis kulit.")

# ---- OUR TEAMS ----
with st.container():
    st.write("---")
    st.header("Meet Our Teams!")
    st.write("##")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.image(team1, use_column_width='auto')
        st.subheader("Azkiya Nurullita")
    with t2:
        st.image(team2, use_column_width='auto')
        st.subheader("Destiana Sofianti")
    with t3:
        st.image(team3, use_column_width='auto')
        st.subheader("Galuh Kusuma Wardhani")
    with t4:
        st.image(team4, use_column_width='auto')
        st.subheader("Meutia Tri Mulyani")
    with t5:
        st.image(team5, use_column_width='auto')
        st.subheader("Ulfa Fadhilatul Mufidah")