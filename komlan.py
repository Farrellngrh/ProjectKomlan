import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from factor_analyzer.factor_analyzer import calculate_kmo
import statsmodels.api as sm
import plotly.graph_objects as go
import chardet
import io

st.set_page_config(page_title="K-Means Clustering", layout="wide")

st.title("📊 K-Means Clustering GUI")

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = []
if 'kmeans_model' not in st.session_state:
    st.session_state.kmeans_model = None
if 'daerah_col' not in st.session_state:
    st.session_state.daerah_col = None
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
if 'kmo_vif_done' not in st.session_state:
    st.session_state.kmo_vif_done = False
if 'kmo_score' not in st.session_state:
    st.session_state.kmo_score = None
if 'vif_df' not in st.session_state:
    st.session_state.vif_df = None
if 'silhouette_avg' not in st.session_state:
    st.session_state.silhouette_avg = None
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = None
if 'silhouette_scores' not in st.session_state:
    st.session_state.silhouette_scores = []

# --- Enhanced File Reading Functions ---
def detect_csv_separator(file_content):
    """Deteksi pemisah CSV yang paling cocok"""
    separators = [';', ',', '\t', '|']
    separator_scores = {}
    
    for sep in separators:
        try:
            # Test dengan beberapa baris pertama
            test_df = pd.read_csv(io.StringIO(file_content[:2000]), sep=sep, nrows=5)
            # Skor berdasarkan jumlah kolom yang masuk akal (2-50 kolom)
            if 2 <= len(test_df.columns) <= 50:
                separator_scores[sep] = len(test_df.columns)
        except:
            separator_scores[sep] = 0
    
    # Kembalikan separator dengan skor tertinggi
    best_sep = max(separator_scores, key=separator_scores.get)
    return best_sep if separator_scores[best_sep] > 1 else ';'

def read_csv_file(uploaded_file):
    """Baca file CSV dengan deteksi encoding dan separator otomatis"""
    try:
        # Baca file sebagai bytes
        file_bytes = uploaded_file.read()
        
        # Deteksi encoding
        encoding_result = chardet.detect(file_bytes)
        detected_encoding = encoding_result['encoding']
        
        # Fallback encodings jika deteksi gagal
        encodings_to_try = [detected_encoding, 'utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
        encodings_to_try = [enc for enc in encodings_to_try if enc is not None]
        
        df = None
        used_encoding = None
        used_separator = None
        
        for encoding in encodings_to_try:
            try:
                # Decode file content
                file_content = file_bytes.decode(encoding)
                
                # Deteksi separator
                separator = detect_csv_separator(file_content)
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Baca CSV
                df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
                
                used_encoding = encoding
                used_separator = separator
                break
                
            except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
        
        if df is not None:
            st.success(f"✅ CSV berhasil dibaca dengan encoding: {used_encoding}, separator: '{used_separator}'")
            return df
        else:
            st.error("❌ Gagal membaca file CSV dengan semua encoding yang dicoba")
            return None
            
    except Exception as e:
        st.error(f"❌ Error saat membaca CSV: {e}")
        return None

def read_excel_file(uploaded_file):
    """Baca file Excel dengan penanganan error yang lebih baik"""
    try:
        # Coba baca semua sheet dan ambil yang pertama
        excel_file = pd.ExcelFile(uploaded_file)
        
        if len(excel_file.sheet_names) > 1:
            st.info(f"📋 File Excel memiliki {len(excel_file.sheet_names)} sheet. Menggunakan sheet pertama: '{excel_file.sheet_names[0]}'")
        
        # Baca sheet pertama
        df = pd.read_excel(uploaded_file, sheet_name=0)
        st.success(f"✅ Excel berhasil dibaca dari sheet: '{excel_file.sheet_names[0]}'")
        return df
        
    except Exception as e:
        st.error(f"❌ Error saat membaca Excel: {e}")
        return None

def clean_and_convert_data(df):
    """Bersihkan dan konversi data ke format yang sesuai"""
    if df is None or df.empty:
        return df
    
    # Hapus baris yang sepenuhnya kosong
    df = df.dropna(how='all')
    
    # Bersihkan nama kolom
    df.columns = df.columns.astype(str).str.strip()
    
    # Konversi kolom numerik
    numeric_converted = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Bersihkan data: hapus spasi, karakter non-numerik kecuali titik dan minus
                cleaned_series = df[col].astype(str).str.strip()
                cleaned_series = cleaned_series.str.replace(r'[^\d\.\-\+eE]+', '', regex=True)
                cleaned_series = cleaned_series.replace('', np.nan)
                
                # Coba konversi ke numerik
                numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                
                # Jika lebih dari 50% berhasil dikonversi, gunakan versi numerik
                non_null_ratio = numeric_series.notna().sum() / len(numeric_series)
                if non_null_ratio > 0.5:
                    df[col] = numeric_series
                    numeric_converted += 1
                    
            except Exception as e:
                # Biarkan kolom tetap sebagai object jika konversi gagal
                pass
    
    if numeric_converted > 0:
        st.info(f"🔢 {numeric_converted} kolom berhasil dikonversi ke numerik")
    
    return df

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        filename = uploaded_file.name
        st.info(f"📁 Memproses file: {filename}")
        
        # Reset session state when new file is uploaded
        if st.session_state.df is None or st.button("🔄 Reset & Reload Data"):
            # Reset all session state
            st.session_state.df = None
            st.session_state.clustering_done = False
            st.session_state.kmo_vif_done = False
            st.session_state.kmo_score = None
            st.session_state.vif_df = None
            st.session_state.silhouette_avg = None
            st.session_state.optimal_k = None
            st.session_state.silhouette_scores = []
            
            # Baca file berdasarkan ekstensi
            if filename.lower().endswith(".csv"):
                st.session_state.df = read_csv_file(uploaded_file)
            elif filename.lower().endswith((".xls", ".xlsx")):
                st.session_state.df = read_excel_file(uploaded_file)
            else:
                st.error("❌ Format file tidak didukung")

            # Proses data jika berhasil dibaca
            if st.session_state.df is not None:
                # Bersihkan dan konversi data
                st.session_state.df = clean_and_convert_data(st.session_state.df)
                
                if st.session_state.df is not None and not st.session_state.df.empty:
                    st.success(f"✅ Data berhasil diproses: {st.session_state.df.shape[0]} baris, {st.session_state.df.shape[1]} kolom")
                    
                    # Set kolom daerah (kolom pertama)
                    if st.session_state.df.columns.size > 0:
                        st.session_state.daerah_col = st.session_state.df.columns[0]
                        st.info(f"🗺️ Kolom identifier: '{st.session_state.daerah_col}'")
                else:
                    st.error("❌ File kosong atau tidak dapat diproses.")

        # Tampilkan info data jika sudah ada
        if st.session_state.df is not None:
            st.markdown("---")
            
            # Tampilkan info kolom
            numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
            non_numeric_cols = st.session_state.df.select_dtypes(exclude=np.number).columns.tolist()
            
            st.markdown("**📊 Info Kolom:**")
            st.markdown(f"- Numerik: {len(numeric_cols)} kolom")
            st.markdown(f"- Non-numerik: {len(non_numeric_cols)} kolom")
            
            # Tampilkan preview data
            st.markdown("**👀 Preview Data:**")
            st.dataframe(st.session_state.df.head(10))
            
            # Tampilkan statistik missing values
            missing_counts = st.session_state.df.isnull().sum()
            if missing_counts.sum() > 0:
                st.markdown("**⚠️ Missing Values:**")
                missing_df = pd.DataFrame({
                    'Kolom': missing_counts.index,
                    'Missing': missing_counts.values,
                    'Persentase': (missing_counts.values / len(st.session_state.df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing'] > 0]
                st.dataframe(missing_df)
    else:
        st.info("📂 Silakan unggah file data Anda.")

# --- Fungsi KMO & VIF ---
def hitung_kmo_vif(df_num):
    """Hitung KMO dan VIF dengan error handling yang lebih baik"""
    try:
        # KMO calculation
        kmo_all, kmo_model = calculate_kmo(df_num)
        
        # VIF calculation
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_num.columns
        vif_values = []
        
        for col in df_num.columns:
            try:
                X = df_num.drop(columns=[col])
                y = df_num[col]
                
                # Skip jika ada missing values
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                if mask.sum() < 2:  # Perlu minimal 2 observasi
                    vif_values.append(np.nan)
                    continue
                
                X_clean = X[mask]
                y_clean = y[mask]
                
                model = sm.OLS(y_clean, sm.add_constant(X_clean)).fit()
                vif = 1 / (1 - model.rsquared) if model.rsquared < 0.9999 else np.inf
                vif_values.append(vif)
            except Exception:
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        return kmo_model, vif_data
        
    except Exception as e:
        st.error(f"❌ Error dalam perhitungan KMO/VIF: {e}")
        return None, None

# --- Area Utama ---
st.markdown("---")
st.subheader("📊 Proses Clustering")

if st.session_state.df is not None:
    col_kmo_vif, col_kmeans = st.columns(2)

    with col_kmo_vif:
        if st.button("1️⃣ Hitung KMO dan VIF"):
            df_num = st.session_state.df.select_dtypes(include=np.number).dropna()
            if df_num.empty:
                st.warning("⚠️ Tidak ada kolom numerik untuk dianalisis.")
            else:
                with st.spinner("🔄 Menghitung KMO dan VIF..."):
                    kmo_score, vif_df = hitung_kmo_vif(df_num)
                    if kmo_score is not None:
                        # Simpan hasil ke session state
                        st.session_state.kmo_score = kmo_score
                        st.session_state.vif_df = vif_df
                        st.session_state.kmo_vif_done = True
                        
                        st.success("✅ KMO dan VIF berhasil dihitung!")
                        st.rerun()  # Refresh untuk menampilkan hasil

        # Tampilkan hasil KMO dan VIF jika sudah dihitung
        if st.session_state.kmo_vif_done and st.session_state.kmo_score is not None:
            st.subheader("✔️ Hasil KMO dan VIF")
            
            # Interpretasi KMO
            if st.session_state.kmo_score >= 0.8:
                kmo_status = "Sangat Baik 🟢"
            elif st.session_state.kmo_score >= 0.7:
                kmo_status = "Baik 🟡"
            elif st.session_state.kmo_score >= 0.6:
                kmo_status = "Cukup 🟠"
            else:
                kmo_status = "Kurang Baik 🔴"
                
            st.markdown(f"**KMO Score: {st.session_state.kmo_score:.3f}** ({kmo_status})")
            st.dataframe(st.session_state.vif_df.round(3))

    with col_kmeans:
        # Input untuk range cluster yang akan dicoba
        st.markdown("**⚙️ Pengaturan Clustering:**")
        col_min, col_max = st.columns(2)
        with col_min:
            min_clusters = st.number_input("Min Cluster", min_value=2, max_value=10, value=2)
        with col_max:
            max_clusters = st.number_input("Max Cluster", min_value=2, max_value=15, value=8)
        
        if st.button("2️⃣ Jalankan K-Means Clustering (Optimal)"):
            df_num = st.session_state.df.select_dtypes(include=np.number).dropna()
            if df_num.empty:
                st.warning("⚠️ Tidak ada kolom numerik untuk clustering.")
            elif min_clusters >= max_clusters:
                st.error("❌ Min cluster harus lebih kecil dari max cluster!")
            elif len(df_num) < max_clusters:
                st.error(f"❌ Jumlah data ({len(df_num)}) harus lebih besar dari max cluster ({max_clusters})!")
            else:
                with st.spinner("🔄 Mencari jumlah cluster optimal..."):
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_num)

                    # Cari cluster optimal berdasarkan silhouette score
                    silhouette_scores = []
                    k_range = range(min_clusters, min(max_clusters + 1, len(df_num)))
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, k in enumerate(k_range):
                        status_text.text(f"🔍 Testing {k} clusters...")
                        
                        try:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                            labels = kmeans.fit_predict(X_scaled)
                            silhouette_avg = silhouette_score(X_scaled, labels)
                            silhouette_scores.append((k, silhouette_avg))
                        except Exception as e:
                            st.warning(f"⚠️ Error pada k={k}: {e}")
                            silhouette_scores.append((k, -1))  # Score negatif untuk error
                        
                        progress_bar.progress((i + 1) / len(k_range))
                    
                    # Hapus progress bar dan status
                    progress_bar.empty()
                    status_text.empty()
                    
                    if not silhouette_scores:
                        st.error("❌ Tidak ada cluster yang berhasil dibuat!")
                        # Use continue or break instead of return outside function
                        st.stop()
                    
                    # Temukan k optimal
                    best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
                    
                    # Tampilkan hasil evaluasi
                    st.subheader("📊 Evaluasi Jumlah Cluster")
                    scores_df = pd.DataFrame(silhouette_scores, columns=['Jumlah_Cluster', 'Silhouette_Score'])
                    scores_df['Silhouette_Score'] = scores_df['Silhouette_Score'].round(4)
                    
                    # Highlight best score
                    def highlight_best(row):
                        if row['Silhouette_Score'] == best_score:
                            return ['background-color: #90EE90'] * len(row)  # Light green
                        return [''] * len(row)
                    
                    st.dataframe(scores_df.style.apply(highlight_best, axis=1))
                    
                    # Plot silhouette scores
                    fig_scores = go.Figure()
                    fig_scores.add_trace(go.Scatter(
                        x=[score[0] for score in silhouette_scores],
                        y=[score[1] for score in silhouette_scores],
                        mode='lines+markers',
                        name='Silhouette Score',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Highlight optimal point
                    fig_scores.add_trace(go.Scatter(
                        x=[best_k],
                        y=[best_score],
                        mode='markers',
                        name=f'Optimal (k={best_k})',
                        marker=dict(size=15, color='red', symbol='star')
                    ))
                    
                    fig_scores.update_layout(
                        title="📈 Silhouette Score vs Jumlah Cluster",
                        xaxis_title="Jumlah Cluster",
                        yaxis_title="Silhouette Score",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)
                    
                    # Jalankan clustering dengan k optimal
                    st.info(f"🎯 Cluster optimal: **{best_k}** dengan Silhouette Score: **{best_score:.4f}**")
                    
                    kmeans_optimal = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
                    labels_optimal = kmeans_optimal.fit_predict(X_scaled)
                    
                    # Urutkan cluster berdasarkan centroid
                    centroids_mean = kmeans_optimal.cluster_centers_.mean(axis=1)
                    cluster_order = centroids_mean.argsort()
                    label_mapping = {old: new for new, old in enumerate(cluster_order)}
                    sorted_labels = [label_mapping[label] for label in labels_optimal]

                    # Update session state
                    st.session_state.cluster_labels = sorted_labels
                    st.session_state.optimal_k = best_k
                    st.session_state.silhouette_scores = silhouette_scores
                    
                    # Hapus kolom Cluster yang sudah ada jika ada
                    if "Cluster" in st.session_state.df.columns:
                        st.session_state.df = st.session_state.df.drop(columns=["Cluster"])
                    
                    st.session_state.df["Cluster"] = st.session_state.cluster_labels
                    st.session_state.silhouette_avg = best_score
                    st.session_state.clustering_done = True
                    
                    st.success(f"✅ Clustering optimal selesai dengan k={best_k}!")
                    st.rerun()  # Refresh untuk menampilkan hasil

        # Tampilkan hasil clustering jika sudah selesai
        if st.session_state.clustering_done and "Cluster" in st.session_state.df.columns:
            st.subheader("✔️ Hasil Clustering")
            
            # Tampilkan info cluster optimal
            if hasattr(st.session_state, 'optimal_k') and st.session_state.optimal_k:
                st.success(f"🎯 **Cluster Optimal: {st.session_state.optimal_k}**")
            
            # Interpretasi Silhouette Score
            if st.session_state.silhouette_avg >= 0.7:
                sil_status = "Sangat Baik 🟢"
            elif st.session_state.silhouette_avg >= 0.5:
                sil_status = "Baik 🟡"
            elif st.session_state.silhouette_avg >= 0.25:
                sil_status = "Cukup 🟠"
            else:
                sil_status = "Kurang Baik 🔴"
                
            st.markdown(f"**Silhouette Score: {st.session_state.silhouette_avg:.4f}** ({sil_status})")
            
            # Preview hasil
            numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
            if "Cluster" in numeric_cols:
                numeric_cols.remove("Cluster")
            
            preview_cols = [st.session_state.daerah_col, "Cluster"] + numeric_cols
            # Pastikan tidak ada duplikasi kolom
            preview_cols = list(dict.fromkeys(preview_cols))
            
            st.dataframe(st.session_state.df[preview_cols].head(10))

            # Bar Plot Cluster Count dengan warna dinamis
            cluster_counts = pd.Series(st.session_state.cluster_labels).value_counts().sort_index()
            
            # Generate colors dynamically based on number of clusters
            colors = [
                "#1F618D", "#196F3D", "#B9770E", "#8E44AD", "#E74C3C", 
                "#F39C12", "#2ECC71", "#3498DB", "#9B59B6", "#E67E22",
                "#1ABC9C", "#34495E", "#F1C40F", "#E91E63", "#FF5722"
            ][:len(cluster_counts)]
            
            fig = go.Figure(data=[
                go.Bar(x=[f"Cluster {i}" for i in cluster_counts.index], 
                       y=cluster_counts.values,
                       marker_color=colors,
                       text=cluster_counts.values,
                       textposition='auto')
            ])
            fig.update_layout(
                title=f"📊 Distribusi Data per Cluster (Total: {len(cluster_counts)} cluster)",
                xaxis_title="Cluster", 
                yaxis_title="Jumlah Data",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # Profil Cluster - Menggunakan checkbox untuk persistent state
    st.markdown("---")
    show_profile = st.checkbox("3️⃣ Tampilkan Profil Cluster", key="show_cluster_profile")
    
    if show_profile and st.session_state.clustering_done and "Cluster" in st.session_state.df.columns:
        st.subheader("📌 Profil Cluster")
        
        # Profil cluster untuk kolom numerik
        df_num_cluster = st.session_state.df.select_dtypes(include=np.number).copy()
        
        # Pastikan kolom Cluster tidak duplikat
        if "Cluster" in df_num_cluster.columns:
            df_num_cluster = df_num_cluster.drop(columns=["Cluster"])
        
        df_num_cluster["Cluster"] = st.session_state.df["Cluster"]
        cluster_profile = df_num_cluster.groupby("Cluster").agg(['mean', 'std', 'count']).round(3)
        
        st.subheader("📈 Statistik Fitur per Cluster")
        st.dataframe(cluster_profile)

        # Daftar entitas per cluster
        st.subheader(f"📍 Daftar {st.session_state.daerah_col} per Cluster")
        for cluster_num in sorted(st.session_state.df["Cluster"].unique()):
            cluster_data = st.session_state.df[st.session_state.df["Cluster"] == cluster_num]
            st.markdown(f"**Cluster {cluster_num}** ({len(cluster_data)} entitas):")
            
            # Tampilkan dalam kolom untuk menghemat ruang
            entities = cluster_data[st.session_state.daerah_col].tolist()
            cols = st.columns(3)
            for i, entity in enumerate(entities):
                cols[i % 3].write(f"• {entity}")
            st.markdown("---")

else:
    st.info("📂 Unggah file data untuk memulai analisis.")