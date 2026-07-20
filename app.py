import streamlit as st
import joblib
import json
import re
import numpy as np
import pandas as pd

# ── Import Keras dari tensorflow langsung ───────────────────
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Sastrawi (opsional) ─────────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer          = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    USE_SASTRAWI     = True
except ImportError:
    USE_SASTRAWI = False

# ── Kamus slang ─────────────────────────────────────────────
SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "udah": "sudah", "udh": "sudah", "dah": "sudah",
    "bgt": "banget", "bgtt": "banget", "bngt": "banget",
    "aku": "saya", "gue": "saya", "gw": "saya",
    "muka": "wajah", "mukaku": "wajah saya",
    "beli": "beli", "pake": "pakai", "pakee": "pakai",
    "bikin": "membuat", "cocok": "sesuai",
    "emang": "memang", "emg": "memang",
    "skrg": "sekarang", "skrng": "sekarang",
    "yg": "yang", "utk": "untuk", "dgn": "dengan",
    "dl": "dulu", "tpi": "tapi", "tp": "tapi",
    "jd": "jadi", "krn": "karena", "karna": "karena",
    "klo": "kalau", "kl": "kalau", "kalo": "kalau",
    "sm": "sama", "spy": "supaya", "sdh": "sudah",
    "lg": "lagi", "lgi": "lagi",
    "breakout": "jerawat", "brekout": "jerawat",
    "moist": "moisturizer",
}

MAX_LEN = 40

def normalize_slang(text):
    return " ".join(SLANG_DICT.get(w, w) for w in text.split())

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_slang(text)
    if USE_SASTRAWI:
        text = stopword_remover.remove(text)
        text = stemmer.stem(text)
    return text

def load_lstm(path):
    # Cara biasa dulu
    try:
        return load_model(path, compile=False)
    except Exception:
        pass
    # Kalau ada error quantization_config, patch Embedding
    try:
        from tensorflow.keras.layers import Embedding
        from tensorflow.keras.models import load_model as lm

        class PatchedEmbedding(Embedding):
            @classmethod
            def from_config(cls, config):
                config.pop("quantization_config", None)
                return super().from_config(config)

        return lm(path, compile=False,
                  custom_objects={"Embedding": PatchedEmbedding})
    except Exception as e:
        raise RuntimeError(f"Tidak bisa load {path}: {e}")

def load_tok(path):
    with open(path) as f:
        data = json.load(f)
    try:
        return tokenizer_from_json(data)
    except Exception:
        return tokenizer_from_json(json.dumps(data))

# ── Load model (cache) ───────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_all():
    nb_cw    = joblib.load("nb_model_cw.pkl")
    tfidf_cw = joblib.load("tfidf_cw.pkl")
    nb_bt    = joblib.load("nb_model_bt.pkl")
    tfidf_bt = joblib.load("tfidf_bt.pkl")
    le       = joblib.load("label_encoder.pkl")
    lstm_cw  = load_lstm("lstm_model_cw.keras")
    tok_cw   = load_tok("tokenizer_cw.json")
    lstm_bt  = load_lstm("lstm_model_bt.keras")
    tok_bt   = load_tok("tokenizer_bt.json")
    return nb_cw, tfidf_cw, nb_bt, tfidf_bt, le, lstm_cw, tok_cw, lstm_bt, tok_bt

# ── Prediksi ────────────────────────────────────────────────
def predict(text, nb_cw, tfidf_cw, nb_bt, tfidf_bt,
            lstm_cw, tok_cw, lstm_bt, tok_bt, CLASS_NAMES):
    clean = preprocess(text)

    # NB Class Weight
    v1 = tfidf_cw.transform([clean])
    nb_cw_lbl   = CLASS_NAMES[nb_cw.predict(v1)[0]]
    nb_cw_prob  = dict(zip(CLASS_NAMES, nb_cw.predict_proba(v1)[0]))

    # NB Back-Translation
    v2 = tfidf_bt.transform([clean])
    nb_bt_lbl   = CLASS_NAMES[nb_bt.predict(v2)[0]]
    nb_bt_prob  = dict(zip(CLASS_NAMES, nb_bt.predict_proba(v2)[0]))

    # LSTM Class Weight
    s1 = pad_sequences(tok_cw.texts_to_sequences([clean]),
                       maxlen=MAX_LEN, padding="post")
    p1 = lstm_cw.predict(s1, verbose=0)[0]
    lstm_cw_lbl  = CLASS_NAMES[int(np.argmax(p1))]
    lstm_cw_prob = dict(zip(CLASS_NAMES, p1.tolist()))

    # LSTM Back-Translation
    s2 = pad_sequences(tok_bt.texts_to_sequences([clean]),
                       maxlen=MAX_LEN, padding="post")
    p2 = lstm_bt.predict(s2, verbose=0)[0]
    lstm_bt_lbl  = CLASS_NAMES[int(np.argmax(p2))]
    lstm_bt_prob = dict(zip(CLASS_NAMES, p2.tolist()))

    return clean, {
        "NB + Class Weight"      : (nb_cw_lbl,  nb_cw_prob),
        "NB + Back-Translation"  : (nb_bt_lbl,  nb_bt_prob),
        "LSTM + Class Weight"    : (lstm_cw_lbl, lstm_cw_prob),
        "LSTM + Back-Translation": (lstm_bt_lbl, lstm_bt_prob),
    }

# ── UI ───────────────────────────────────────────────────────
EMOJI = {"positif": "😊", "negatif": "😠", "netral": "😐"}
COLOR = {"positif": "#28a745", "negatif": "#dc3545", "netral": "#fd7e14"}

st.set_page_config(page_title="Analisis Sentimen Dorskin",
                   page_icon="💄", layout="wide")

st.title("💄 Analisis Sentimen Ulasan Dorskin")
st.caption("Prediksi sentimen komentar TikTok Shop — Naive Bayes & LSTM + FastText")
st.divider()

try:
    nb_cw, tfidf_cw, nb_bt, tfidf_bt, le, lstm_cw, tok_cw, lstm_bt, tok_bt = load_all()
    CLASS_NAMES = list(le.classes_)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.info("Pastikan semua 9 file model ada di folder yang sama dengan app.py")
    st.stop()

teks_input = st.text_area(
    "✍️ Masukkan komentar:",
    placeholder="Contoh: Produknya bagus banget, kulit jadi glowing setelah pakai seminggu",
    height=120,
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    klik = st.button("🔍 Analisis", use_container_width=True, type="primary")

if klik:
    if not teks_input.strip():
        st.warning("Masukkan komentar terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            clean, hasil = predict(
                teks_input,
                nb_cw, tfidf_cw, nb_bt, tfidf_bt,
                lstm_cw, tok_cw, lstm_bt, tok_bt, CLASS_NAMES
            )

        st.divider()
        st.subheader("📝 Teks setelah preprocessing")
        st.code(clean, language=None)

        st.subheader("🎯 Hasil Prediksi")
        cols = st.columns(4)
        for col, (nama_model, (label, proba)) in zip(cols, hasil.items()):
            with col:
                st.markdown(f"""
                <div style="border:2px solid {COLOR[label]};border-radius:10px;
                            padding:16px;text-align:center;
                            background:{COLOR[label]}18;margin-bottom:12px;">
                    <div style="font-size:12px;color:#666;margin-bottom:6px;">{nama_model}</div>
                    <div style="font-size:32px;">{EMOJI[label]}</div>
                    <div style="font-size:18px;font-weight:bold;color:{COLOR[label]};
                                text-transform:uppercase;margin-top:4px;">{label}</div>
                </div>""", unsafe_allow_html=True)
                for cls in CLASS_NAMES:
                    pct = proba[cls] * 100
                    st.progress(int(pct), text=f"{cls.capitalize()}: {pct:.1f}%")

        st.divider()
        st.subheader("📊 Konsensus Model")
        semua_label   = [v[0] for v in hasil.values()]
        mayoritas     = max(set(semua_label), key=semua_label.count)
        jml_setuju    = semua_label.count(mayoritas)

        if jml_setuju == 4:
            st.success(f"Semua model sepakat: **{mayoritas.upper()}** {EMOJI[mayoritas]}")
        elif jml_setuju == 3:
            st.info(f"3 dari 4 model memprediksi **{mayoritas.upper()}** {EMOJI[mayoritas]}")
        else:
            st.warning("Model tidak sepakat — prediksi bervariasi antar model.")

        st.dataframe(
            pd.DataFrame([{"Model": m, "Prediksi": l.capitalize(),
                           "Sentimen": EMOJI[l]} for m, (l, _) in hasil.items()]),
            use_container_width=True, hide_index=True,
        )
