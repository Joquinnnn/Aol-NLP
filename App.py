import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
download_nltk_data()

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered"
)
st.markdown("""
<style>
    /* Tombol */
    div[data-testid="stButton"] button {
        background-color: #345790;
        color: white;
        border-radius: 12px;
        border: none;
        font-size: 16px;
        font-weight: bold;
        padding: 12px;
        transition: 0.3s;
    }

    /* Hover tombol */
    div[data-testid="stButton"] button:hover {
        background-color: #c9daf8;
        color: white;
        transform: scale(1.02);
    }

""", unsafe_allow_html=True)

en_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text)
    # Hapus HTML, URL, dan Karakter Khusus
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    text = text.lower() # Case folding
    tokens = word_tokenize(text) # Tokenisasi
    tokens = [word for word in tokens if word not in en_stopwords] 
    return " ".join(tokens)


@st.cache_resource
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

try:
    tfidf, model = load_models()
except FileNotFoundError:
    st.error("File model (.pkl) tidak ditemukan! Pastikan 'tfidf_vectorizer.pkl' dan 'best_model.pkl' ada di folder yang sama dengan app.py.")
    st.stop()


target_cols = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']


st.title("Multi-Label Toxic Comment Detection using SVM")
st.markdown("""
Aplikasi ini menggunakan Machine Learning untuk mendeteksi komentar berbahasa Inggris yang mengandung unsur toksik, ancaman, atau ujaran kebencian.
""")

# Input teks
user_input = st.text_area("Masukkan Komentar Anda:", height=150, placeholder="Ketik komentar berbahasa Inggris di sini...")

# Tombol Prediksi
if st.button("Analisis Komentar", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Mohon masukkan teks terlebih dahulu!")
    else:
        with st.spinner('Menganalisis teks...'):
            # 1. Preprocessing input user
            cleaned_text = preprocess_text(user_input)
            
            # 2. Transformasi dengan TF-IDF
            vectorized_text = tfidf.transform([cleaned_text])
            
            # 3. Prediksi dengan Model
            prediction = model.predict(vectorized_text)[0]
            
            st.markdown("---")
            st.subheader("📊 Hasil Analisis:")
            
            # Mengecek apakah ada setidaknya 1 label yang terdeteksi
            is_toxic = sum(prediction) > 0
            
            if not is_toxic:
                st.success("✅ **SAFE COMMENT!** No toxic content was detected in this comment.")
                st.balloons()
            else:
                st.error("🚫 **WARNING!** This comment is detected as containing the following toxic content:")
                st.snow()
               
                
                # Menampilkan hasil deteksi dalam format grid (3 kolom)
                cols = st.columns(3)
                for i, label in enumerate(target_cols):
                    with cols[i % 3]:
                        if prediction[i] == 1:
                            # Jika terdeteksi (1), tampilkan dengan warna merah
                            st.error(f"⚠️ {label} Detected")
                        else:
                            # Jika aman (0), tampilkan dengan warna hijau pucat
                            st.success(f"✔️ {label} Not Detected")
