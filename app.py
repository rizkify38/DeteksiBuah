import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "model_buah.h5")
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Gagal memuat model di: {model_path}\n{e}")
    st.stop()

# Daftar kelas (urutan sesuai training model)
class_names = [
    "apel busuk", "apel segar", "apel setengah segar",
    "jeruk busuk", "jeruk segar", "jeruk setengah segar",
    "melon busuk", "melon segar", "melon setengah segar",
    "pisang busuk", "pisang segar", "pisang setengah segar",
    "tomat busuk", "tomat segar", "tomat setengah segar"
]

# Fungsi prediksi
def predict_image(img):
    img = img.resize((224, 224))  # sesuaikan dengan input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    return class_names[pred_index], predictions[0][pred_index]

def prepare_square_image_from_path(path, size=180):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            img = img.crop((left, top, left + min_side, top + min_side))
            img = img.resize((size, size))
            return img
    except Exception:
        return None

# Navigasi halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Prediksi Buah"])

if page == "Beranda":
    st.title("Prediksi Kondisi Buah")
    st.write("""
    Aplikasi ini dapat mengenali kondisi buah berdasarkan gambar.
    Berikut kategori yang digunakan:
    """)
    
    buah_info = {
        "Apel": {
            "Segar": "Kulit mengkilap, warna cerah, tidak ada bercak busuk.",
            "Setengah Segar": "Warna sedikit pudar, mulai ada sedikit bercak atau kerutan.",
            "Busuk": "Banyak bercak hitam/coklat, tekstur lembek, berbau tidak sedap."
        },
        "Jeruk": {
            "Segar": "Kulit cerah, mulus, tidak ada kerutan.",
            "Setengah Segar": "Kulit mulai keriput, sedikit lunak di beberapa bagian.",
            "Busuk": "Kulit sangat keriput, berjamur atau bau asam tajam."
        },
        "Melon": {
            "Segar": "Kulit keras, warna sesuai jenis, aroma segar.",
            "Setengah Segar": "Mulai ada bercak gelap, aroma agak tajam.",
            "Busuk": "Banyak bercak busuk, kulit lembek, aroma menyengat."
        },
        "Pisang": {
            "Segar": "Kulit kuning cerah, tidak ada bercak hitam berlebihan.",
            "Setengah Segar": "Mulai muncul bercak coklat kecil.",
            "Busuk": "Kulit hitam hampir seluruhnya, tekstur sangat lembek."
        },
        "Tomat": {
            "Segar": "Warna merah cerah, kulit mulus, keras saat ditekan.",
            "Setengah Segar": "Kulit sedikit keriput, mulai lembek.",
            "Busuk": "Bercak hitam, kulit pecah, berair dan berbau."
        }
    }
    
    emoji_map = {"Apel": "🍎", "Jeruk": "🍊", "Melon": "🍈", "Pisang": "🍌", "Tomat": "🍅"}
    images_dir = os.path.join(base_dir, "images")
    fruit_to_image = {
        "Apel": "apel_segar_1.jpg",
        "Jeruk": "jeruk_segar_2.jpg",
        "Melon": "melon_segar_3.jpg",
        "Pisang": "pisang_segar_4.jpg",
        "Tomat": "tomat_segar_5.jpg",
    }

    if "selected_fruit" not in st.session_state:
        st.session_state["selected_fruit"] = None

    st.subheader("Pilih Buah")
    cols = st.columns(5)
    for idx, (buah, filename) in enumerate(fruit_to_image.items()):
        with cols[idx % 5]:
            st.caption(f"{emoji_map.get(buah, '')} {buah}")
            img_path = os.path.join(images_dir, filename)
            if os.path.exists(img_path):
                square_img = prepare_square_image_from_path(img_path, size=180)
                if square_img is not None:
                    st.image(square_img, width=180)
                else:
                    st.image(img_path, width=180)
            else:
                st.warning("Belum ada gambar")
            if st.button("Lihat penjelasan", key=f"btn_{buah}"):
                st.session_state["selected_fruit"] = buah

    selected = st.session_state.get("selected_fruit")
    if selected:
        st.markdown("---")
        st.subheader(f"{emoji_map.get(selected, '🍏')} {selected}")
        for status, deskripsi in buah_info[selected].items():
            st.markdown(f"**{status}:** {deskripsi}")
    else:
        st.info("Klik tombol di bawah gambar untuk melihat penjelasan.")

elif page == "Prediksi Buah":
    st.title("📷 Prediksi Kondisi Buah")
    uploaded_file = st.file_uploader("Upload gambar buah...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah")
        
        st.write("⏳ Sedang memproses...")
        label, confidence = predict_image(image)
        
        st.success(f"Prediksi: **{label}**")
        st.write(f"Tingkat keyakinan: **{confidence*100:.2f}%**")
