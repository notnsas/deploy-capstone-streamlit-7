import streamlit as st
import pandas as pd
import time
import copy
import utils  # Custom Module
import visualizer  # Custom Module
import setting  # Custom Module
import base64
import os

# ==========================================
# 0. ASSETS & ICONS SETUP (LOCAL FILES)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(BASE_DIR, "icons")

# Ensure you have these files in your 'icons/' folder
# I have mapped generic names to specific usage
ICONS = {
    "menu": os.path.join(ICON_DIR, "menu.png"),
    "home": os.path.join(ICON_DIR, "home.png"),
    "single_analysis": os.path.join(ICON_DIR, "search.png"),
    "batch_analysis": os.path.join(ICON_DIR, "folder.png"),
    "docs": os.path.join(ICON_DIR, "book.png"),
    "about": os.path.join(ICON_DIR, "info.png"),
    "settings": os.path.join(ICON_DIR, "settings.png"),
    "dual_lang": os.path.join(ICON_DIR, "translation.png"),
    "aspect": os.path.join(ICON_DIR, "mind-map.png"),
    "dataviz": os.path.join(ICON_DIR, "chart.png"),
    "add": os.path.join(ICON_DIR, "plus.png"),
    "download": os.path.join(ICON_DIR, "download.png"),
    "rocket": os.path.join(ICON_DIR, "rocket.png"),
    "time": os.path.join(ICON_DIR, "clock.png"),  # Generic clock icon
}


def get_img_as_base64(file_path):
    """
    Reads a local image file and converts it to a base64 string
    so it can be displayed in HTML (st.markdown).
    """
    if not os.path.exists(file_path):
        return ""

    with open(file_path, "rb") as f:
        data = f.read()

    encoded = base64.b64encode(data).decode()
    return f"data:image/png;base64,{encoded}"


def render_header_with_image(title, image_path, size=40):
    """Helper to render a title with a local image icon"""
    img_src = get_img_as_base64(image_path)
    # If image missing, just render text
    if not img_src:
        st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
        return

    st.markdown(
        f"""
        <h1 style="display: flex; align-items: center; gap: 15px;">
            <img src="{img_src}" width="{size}" height="{size}" style="vertical-align: middle; filter: drop-shadow(0px 0px 5px rgba(29, 185, 84, 0.3));"/>
            {title}
        </h1>
        """,
        unsafe_allow_html=True,
    )


def render_subheader_with_image(title, image_path, size=30):
    """Helper to render a subheader with a local image icon"""
    img_src = get_img_as_base64(image_path)
    if not img_src:
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
        return

    st.markdown(
        f"""
        <h3 style="display: flex; align-items: center; gap: 10px;">
            <img src="{img_src}" width="{size}" height="{size}" style="vertical-align: middle;"/>
            {title}
        </h3>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA
# ==========================================
st.set_page_config(
    page_title="Spotify Sentiment Intel",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown(
    """
<style>
    :root {
        --primary-color: #1DB954;
        --bg-color: #121212;
        --secondary-bg: #191414;
        --text-color: #FFFFFF;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-bg);
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #b3b3b3;
    }
    div[data-testid="stMetricValue"] {
        color: var(--primary-color);
        font-weight: bold;
    }

    /* Button styling */
    div.stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #1ed760;
        border: 1px solid white;
    }

    /* Card styling */
    .aspect-card-pos {
        background-color: #0d2e18;
        border-left: 5px solid #1DB954;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .aspect-card-neg {
        background-color: #3b0d10;
        border-left: 5px solid #ff4d4d;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .aspect-card-neutral {
        background-color: #2b2b2b;
        border-left: 5px solid #666666;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        opacity: 0.7;
    }
    .trigger-text {
        font-size: 0.85em;
        color: #b3b3b3;
        font-style: italic;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. INISIALISASI MODEL (CACHING)
# ==========================================
if "ASPECT_KEYWORDS" not in st.session_state:
    st.session_state["ASPECT_KEYWORDS"] = copy.deepcopy(setting.ASPECT_KEYWORDS)


@st.cache_resource
def initialize_ai_engine():
    return utils.load_all_models()


if "models_loaded" not in st.session_state:
    with st.spinner("Sedang Memanaskan Mesin AI (Loading Models)..."):
        dict_model = initialize_ai_engine()
        models_en = dict_model["en"]
        models_id = dict_model["id"]

        if models_en is None or models_id is None:
            st.error("Gagal memuat model. Pastikan folder 'models/' lengkap.")
            st.stop()

        st.session_state["models_en"] = models_en
        st.session_state["models_id"] = models_id
        st.session_state["models_loaded"] = True
    st.toast("Sistem AI Siap Digunakan!")
else:
    models_en = st.session_state["models_en"]
    models_id = st.session_state["models_id"]


# ==========================================
# 3. SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        st.image(
            "https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png",
            width=200,
        )

    st.markdown("---")
    render_subheader_with_image("Main Menu", ICONS["menu"])

    # Cleaned text options (No emojis here, they don't render images well in Radio)
    menu = st.radio(
        "Pilih Mode Analisis:",
        [
            "Beranda",
            "Analisis Teks (Single)",
            "Analisis File (Batch)",
            "Dokumentasi & Panduan",
        ],
        index=0,
    )

    st.markdown("---")
    render_subheader_with_image("Tentang Sistem", ICONS["about"], size=24)
    st.info(
        """
        Sistem ini menggunakan arsitektur **Hybrid**:
        - **IndoBERT / BERT** (High Accuracy)
        - **ABSA Engine** (Granular)
        """
    )

# ==========================================
# 4. HALAMAN UTAMA: BERANDA
# ==========================================
if menu == "Beranda":
    render_header_with_image(
        "Welcome to Spotify Review Intelligence", ICONS["home"], size=50
    )

    st.markdown(
        """
    Platform analisis sentimen tingkat lanjut yang dirancang untuk membedah ribuan ulasan pengguna Spotify 
    menjadi wawasan bisnis yang dapat ditindaklanjuti.
    """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        render_subheader_with_image("Dual Language", ICONS["dual_lang"])
        st.write("Mendeteksi otomatis Bahasa Indonesia & Inggris.")
    with c2:
        render_subheader_with_image("Aspect-Based", ICONS["aspect"])
        st.write("Mendeteksi Audio, Harga, Iklan, & Bug.")
    with c3:
        render_subheader_with_image("Interactive Viz", ICONS["dataviz"])
        st.write("Visualisasi WordCloud dan Chart interaktif.")

    st.divider()

    # Replaced Emoji Header with Image Header
    # Assuming 'rocket' is mapped in ICONS, or reuse 'about' if missing
    start_icon = ICONS.get("rocket", ICONS["about"])
    render_subheader_with_image("Cara Memulai", start_icon)

    st.markdown("1. Buka menu **Analisis Teks** untuk menguji satu kalimat review.")
    st.markdown(
        "2. Buka menu **Analisis File** untuk mengupload CSV berisi ribuan data."
    )

# ==========================================
# 5. HALAMAN KEDUA: ANALISIS TEKS TUNGGAL
# ==========================================
elif menu == "Analisis Teks (Single)":
    render_header_with_image("Granular Text Analysis", ICONS["single_analysis"])

    st.markdown(
        "Masukkan satu ulasan untuk melihat bagaimana AI membedah sentimen dan aspeknya."
    )

    with st.container():
        input_text = st.text_area(
            "Masukkan Ulasan User:",
            height=150,
            placeholder="Contoh: Aplikasinya bagus, tapi iklannya kebanyakan...",
        )

        col_btn, col_opt = st.columns([1, 4])
        with col_btn:
            analyze_btn = st.button(
                "🔍 Analisis Sekarang", type="primary", use_container_width=True
            )

    if analyze_btn and input_text:
        start_time = time.time()
        global_sentiment, confidence, aspect_results, lang = (
            utils.analyze_single_review_complete(
                st.session_state["ASPECT_KEYWORDS"], input_text, (models_en, models_id)
            )
        )
        end_time = time.time()
        print(f"Aspect result : {aspect_results}")
        st.divider()

        st.markdown("### 🎯 Hasil Analisis AI")

        m1, m2, m3 = st.columns(3)
        with m1:

            emoji = "😄" if global_sentiment == "Positive" else "😡"
            st.metric("Sentimen Global", f"{emoji} {global_sentiment}")
        with m2:
            st.metric("Keyakinan", f"{confidence:.1%}")
        with m3:
            lang_name = "Indonesia" if lang == "id" else "Inggris"
            st.metric("Bahasa", lang_name)

        render_subheader_with_image("Breakdown Per Aspek", ICONS["aspect"], size=28)

        # Logic Display Cards
        all_possible_aspects = list(st.session_state["ASPECT_KEYWORDS"][lang].keys())
        display_list = []

        for aspect, data in aspect_results.items():
            display_list.append({"name": aspect, "data": data, "status": "active"})
        for aspect in all_possible_aspects:
            if aspect not in aspect_results:
                display_list.append(
                    {"name": aspect, "data": None, "status": "inactive"}
                )

        col_left, col_right = st.columns(2)
        mid = (len(display_list) + 1) // 2
        left_items = display_list[:mid]
        right_items = display_list[mid:]

        def render_card(item):
            aspect_name = item["name"]
            if item["status"] == "active":
                data = item["data"]
                css_class = (
                    "aspect-card-pos"
                    if data["label"] == "Positive"
                    else "aspect-card-neg"
                )
                label_text = data["label"].upper()
                score_fmt = f"{data['score']:.1%}"
                trigger_text = (
                    f"Kata Pemicu: '{data['trigger']}'"
                    if data["trigger"]
                    else "Trigger implisit"
                )
                text_color = "white"
            else:
                css_class = "aspect-card-neutral"
                label_text = "NOT DETECTED"
                score_fmt = "-"
                trigger_text = "Tidak ditemukan dalam teks"
                text_color = "#888"

            return f"""
            <div class="{css_class}">
                <h4 style="margin:0; color:{text_color};">{aspect_name}</h4>
                <div style="display:flex; justify-content:space-between; margin-top:5px;">
                    <span style="font-weight:bold; color:{text_color};">{label_text}</span>
                    <span style="color:{text_color};">{score_fmt}</span>
                </div>
                <div class="trigger-text">{trigger_text}</div>
            </div>
            """

        with col_left:
            for item in left_items:
                st.markdown(render_card(item), unsafe_allow_html=True)
        with col_right:
            for item in right_items:
                st.markdown(render_card(item), unsafe_allow_html=True)

        st.caption(f"Waktu Pemrosesan: {end_time - start_time:.4f} detik")

    st.divider()
    render_subheader_with_image(
        "Konfigurasi Aspek & Keyword", ICONS["settings"], size=28
    )

    with st.expander("Buka Panel Manajemen Aspek", expanded=False):
        col_lang, _ = st.columns([1, 2])
        with col_lang:

            lang_choice = st.radio(
                "Bahasa",
                ["Indonesia", "English"],
                horizontal=True,
                label_visibility="collapsed",
            )

        target_lang = "id" if "Indonesia" in lang_choice else "en"
        current_aspects = st.session_state["ASPECT_KEYWORDS"][target_lang]

        tab_add_kw, tab_new_cat = st.tabs(
            ["➕ Tambah Keyword", "🆕 Buat Kategori Baru"]
        )

        with tab_add_kw:
            c1, c2 = st.columns(2)
            with c1:
                selected_cat = st.selectbox(
                    "Pilih Kategori:", list(current_aspects.keys())
                )
            with c2:
                new_keyword = st.text_input("Keyword Baru:")

            existing_kws = ", ".join([f"`{k}`" for k in current_aspects[selected_cat]])

            st.info(f"📂 **Keyword saat ini:** {existing_kws}")

            if st.button("Simpan Keyword", type="primary"):
                if new_keyword and new_keyword not in current_aspects[selected_cat]:
                    st.session_state["ASPECT_KEYWORDS"][target_lang][
                        selected_cat
                    ].append(new_keyword)

                    st.toast(f"Berhasil menambahkan!")
                    time.sleep(1)
                    st.rerun()

        with tab_new_cat:
            c_cat, c_kw = st.columns(2)
            with c_cat:
                new_cat_name = st.text_input("Nama Kategori Baru:")
            with c_kw:
                first_kw_input = st.text_input("Keyword Pertama:")

            if st.button("Buat Kategori", type="primary"):
                if new_cat_name and first_kw_input:
                    st.session_state["ASPECT_KEYWORDS"][target_lang][new_cat_name] = [
                        first_kw_input
                    ]

                    st.toast(f"Kategori '{new_cat_name}' berhasil dibuat!")
                    time.sleep(1)
                    st.rerun()

# ==========================================
# 6. HALAMAN KETIGA: ANALISIS BATCH (FILE)
# ==========================================
elif menu == "Analisis File (Batch)":
    render_header_with_image("Batch Sentiment Processing", ICONS["batch_analysis"])
    st.markdown(
        "Unggah file (CSV/Excel) ulasan aplikasi untuk analisis massal otomatis."
    )

    uploaded_file = st.file_uploader("Drop file di sini", type=["csv", "xlsx"])

    if uploaded_file:
        df = utils.load_uploaded_file(uploaded_file)
        if df is not None:
            text_col = utils.find_text_column(df)
            if text_col:

                st.success(f"File berhasil dimuat! **{len(df)}** baris.")

                if st.button("Jalankan Analisis AI (Batch)", type="primary"):
                    progress_text = "Memproses ulasan..."
                    my_bar = st.progress(0, text=progress_text)
                    results = []
                    df_to_process = df
                    total_items = len(df_to_process)

                    for idx, row in df_to_process.iterrows():
                        text = str(row[text_col])
                        gl_lbl, gl_conf, aspects, lang = (
                            utils.analyze_single_review_complete(
                                st.session_state["ASPECT_KEYWORDS"],
                                text,
                                (models_en, models_id),
                            )
                        )
                        res_row = {
                            "Original Text": text,
                            "Language": lang,
                            "Global Sentiment": gl_lbl,
                            "Confidence": gl_conf,
                            "Aspects JSON": str(aspects),
                        }
                        for asp, detail in aspects.items():
                            res_row[f"{asp}_Sentiment"] = detail["label"]
                        results.append(res_row)
                        my_bar.progress(
                            int(((idx + 1) / total_items) * 100),
                            text=f"Processing {idx+1}/{total_items}...",
                        )

                    my_bar.empty()
                    st.session_state["batch_result"] = pd.DataFrame(results)
            else:
                st.error("Tidak dapat menemukan kolom teks.")

    if "batch_result" in st.session_state:
        df_res = st.session_state["batch_result"]
        st.divider()
        render_subheader_with_image("Laporan & Dashboard", ICONS["dataviz"])

        tab_sum, tab_viz, tab_data = st.tabs(
            ["Summary & KPIs", "Visualisasi Mendalam", "Data Detail"]
        )

        with tab_sum:
            visualizer.display_kpi_metrics(df_res)
            c_d, c_b = st.columns(2)
            with c_d:
                visualizer.plot_sentiment_donut(df_res)
            with c_b:
                visualizer.plot_aspect_bar_chart(df_res)
            st.subheader("Apa yang Paling Sering Dibahas?")
            visualizer.plot_trigger_sentiment_chart(df_res)

        with tab_viz:
            st.write("#### Wordcloud Analisis")
            lang_wc = st.selectbox("Pilih Bahasa:", df_res["Language"].unique())
            subset_df = df_res[df_res["Language"] == lang_wc]
            wc1, wc2 = st.columns(2)
            with wc1:
                st.write("**Positif:**")
                visualizer.generate_wordcloud(subset_df, "Positive")
            with wc2:
                st.write("**Negatif:**")
                visualizer.generate_wordcloud(subset_df, "Negative")

        with tab_data:
            st.dataframe(df_res)

            st.download_button(
                label="Download CSV",
                data=utils.convert_df_to_csv(df_res),
                file_name=f"result_{int(time.time())}.csv",
                mime="text/csv",
            )

# ==========================================
# 7. HALAMAN KEEMPAT: DOKUMENTASI
# ==========================================
elif menu == "Dokumentasi & Panduan":
    render_header_with_image("Dokumentasi Sistem", ICONS["docs"])

    tab_guide, tab_dict = st.tabs(["Cara Penggunaan", "Kamus Aspek (Live)"])

    with tab_guide:
        st.header("Panduan Penggunaan")

        with st.expander("Analisis Teks (Single)"):
            st.write("Masukkan kalimat, klik tombol analisis, lihat hasil.")
        with st.expander("Analisis File (Batch)"):
            st.write("Upload CSV, klik jalankan, download hasil.")

    with tab_dict:
        st.header("Daftar Keyword")

        lc = st.radio("Bahasa:", ["Indonesia", "English"], horizontal=True)
        tl = "id" if "Indonesia" in lc else "en"

        if "ASPECT_KEYWORDS" in st.session_state:
            for cat, kws in st.session_state["ASPECT_KEYWORDS"][tl].items():
                with st.expander(f"Kategori: **{cat}**"):
                    html = "".join(
                        [
                            f"<code style='color:#1DB954;background:#191414;border:1px solid #333;margin:2px;padding:2px 6px;border-radius:4px;display:inline-block;'>{k}</code> "
                            for k in sorted(kws)
                        ]
                    )
                    st.markdown(html, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>&copy; 2025 Capstone Project</div>",
    unsafe_allow_html=True,
)
