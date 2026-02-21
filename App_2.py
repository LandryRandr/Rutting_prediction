import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Pavement Analysis Pro", page_icon="🛣️", layout="wide")

# --- STYLE CSS (Police agrandie + Largeur optimisée) ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    html, body, [class*="st-"] { font-size: 1.12rem; }
    h1 { font-size: 2.5rem !important; color: #1E3A8A; }
    .stMetric { background-color: #F0F4F8; padding: 15px; border-radius: 10px; border: 1px solid #D1D5DB; }
    </style>
    """, unsafe_allow_html=True)

# 2. CHARGEMENT DES ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('rf_rutting_model.pkl')
    scaler = joblib.load('scaler_rutting.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("🛣️ Analyse Prédictive de l'Orniérage")

# --- SÉLECTION DU NOMBRE DE COUCHES ---
nb_couches = st.sidebar.selectbox("📋 Nombre de couches", [3, 4, 5], index=0)
st.sidebar.divider()
st.sidebar.info("Remplissez le formulaire à droite pour mettre à jour les graphiques.")

# 3. MISE EN PAGE : FORMULAIRE (GAUCHE) | GRAPHIQUE (DROITE)
col_form, col_graph, tab2 = st.columns([1.2, 1, 1], gap="large")

with col_form:
    with st.form("rutting_form"):
        st.subheader("🏗️ Paramètres de la Structure")
        
        # Section Temps & Trafic (Age actuel pour le calcul ponctuel)
        age_actuel = st.number_input("Âge actuel de la route (ans)", 1.0, 10.0, 4.0)
        KESAL = st.number_input("Trafic cumulé (KESAL)", 0.0, 500000.0, 0.0)
        IRI = st.number_input("Roughness (IRI)", 0.0, 15.0, 0.0)

        # Couche 1
        st.markdown("---")
        st.markdown("**Couche 1 (Sol)**")
        choix_c1 = st.selectbox("Type de sol", ["Coarse Grained Soil", "Fine Grained Soil"])
        RMOD_1 = st.number_input("RMOD_1", 0.0, 5000.0, 0.0)
        M_FineGsoil_1 = 1.0 if choix_c1 == "Fine Grained Soil" else 0.0
        M_CoarseGsoil_1 = 1.0 if choix_c1 == "Coarse Grained Soil" else 0.0

        # Couche 2
        st.markdown("**Couche 2**")
        choix_c2 = st.selectbox("Matériau C2", ["Asphalt", "Ciment", "Gravel"])
        Layer2_Thick = st.number_input("Épaisseur C2 (mm)", 0.0, 600.0, 0.0)
        RMOD_2 = st.number_input("RMOD_2", 0.0, 25000.0, 0.0)
        M_Asphalt_2 = 1.0 if choix_c2 == "Asphalt" else 0.0
        M_Ciment_2 = 1.0 if choix_c2 == "Ciment" else 0.0
        M_Gravel_2 = 1.0 if choix_c2 == "Gravel" else 0.0

        # Couche 3
        st.markdown("**Couche 3**")
        choix_c3 = st.selectbox("Matériau C3", ["Gravel", "Asphalt", "Ciment"])
        Layer3_Thick = st.number_input("Épaisseur C3 (mm)", 0.0, 600.0, 0.0)
        RMOD_3 = st.number_input("RMOD_3", 0.0, 25000.0, 0.0)
        M_Asphalt_3 = 1.0 if choix_c3 == "Asphalt" else 0.0
        M_Ciment_3 = 1.0 if choix_c3 == "Ciment" else 0.0
        M_Gravel_3 = 1.0 if choix_c3 == "Gravel" else 0.0

        # Couches 4 et 5
        if nb_couches >= 4:
            st.markdown("**Couche 4**")
            choix_c4 = st.selectbox("Matériau C4", ["Gravel", "Asphalt", "Ciment"])
            Layer4_Thick = st.number_input("Épaisseur C4 (mm)", 0.0, 600.0, 0.0)
            RMOD_4 = st.number_input("RMOD_4", 0.0, 25000.0, 0.0)
            M_Asphalt_4, M_Ciment_4, M_Gravel_4, Aucun_C4 = (1.0 if choix_c4 == "Asphalt" else 0.0), (1.0 if choix_c4 == "Ciment" else 0.0), (1.0 if choix_c4 == "Gravel" else 0.0), 1.0
        else:
            Layer4_Thick, RMOD_4, M_Asphalt_4, M_Ciment_4, M_Gravel_4, Aucun_C4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if nb_couches == 5:
            st.markdown("**Couche 5**")
            choix_c5 = st.selectbox("Matériau C5", ["Asphalt", "Sol"])
            Layer5_Thick = st.number_input("Épaisseur C5 (mm)", 0.0, 1000.0, 0.0)
            RMOD_5 = st.number_input("RMOD_5", 0.0, 15000.0, 0.0)
            M_Asphalt_5, Aucun_C5 = (1.0 if choix_c5 == "Asphalt" else 0.0), 1.0
        else:
            Layer5_Thick, RMOD_5, M_Asphalt_5, Aucun_C5 = 0.0, 0.0, 0.0, 0.0

        # Environnement
        st.divider()
        choix_drain = st.selectbox("Drainage efficace ?", ["Non", "Oui"])
        DRAINAGE = 1.0 if choix_drain == "Oui" else 0.0
        
        # Données météo regroupées
        st.markdown("**Climat**")
        t_moy = st.number_input("Temp. Moy (°C)", 0.0, 65.0, 0.0)
        p_ann = st.number_input("Précip. Annuelle (mm)", 0.0, 42000.0, 0.0)
        p_max = st.number_input("Pluie Max Mensuelle", 0.0, 1000.0, 0.0)
        p_per = st.number_input("Période Précip (mois)", 0.0, 365.0, 0.0)
        h_rel = st.number_input("Humidité Moy (%)", 0.0, 100.0, 0.0)
        h_max = st.number_input("Humidité Max (%)", 0.0, 100.0, 0.0)
        h_min = st.number_input("Humidité Min (%)", 0.0, 100.0, 0.0)
        wind = st.number_input("Vitesse Vent (m/s)", 0.0, 150.0, 0.0)

        submit = st.form_submit_button("📊 ANALYSER L'ÉVOLUTION")

# 4. ANALYSE ET GRAPHIQUE (DROITE)
with col_graph:
    if submit:
        # --- RÉSUMÉ DES DONNÉES ---
        st.subheader("📝 Résumé de la Route")
        res_col1, res_col2 = st.columns(2)
        res_col1.write(f"**Structure:** {nb_couches} couches")
        res_col1.write(f"**Trafic:** {KESAL} KESAL")
        res_col2.write(f"**Sol Support:** {choix_c1}")
        res_col2.write(f"**Drainage:** {'Oui' if DRAINAGE == 1 else 'Non'}")
        
        # --- CALCUL DE L'ÉVOLUTION (0 à 15 ANS) ---
        ages_simulés = list(range(1, 10))
        predictions_rutting = []

        for a in ages_simulés:
            feat_sim = np.array([[
                float(a), IRI, M_FineGsoil_1, M_CoarseGsoil_1, RMOD_1,
                Layer2_Thick, M_Asphalt_2, M_Ciment_2, M_Gravel_2, RMOD_2,
                Layer3_Thick, M_Asphalt_3, M_Ciment_3, M_Gravel_3, RMOD_3,
                Layer4_Thick, M_Asphalt_4, M_Ciment_4, M_Gravel_4, RMOD_4,
                Layer5_Thick, RMOD_5, M_Asphalt_5,
                Aucun_C4, Aucun_C5, DRAINAGE,
                p_ann, p_max, p_per,
                t_moy, h_rel, h_max, h_min, wind, KESAL
            ]])
            
            val_scaled = scaler.transform(feat_sim)
            predictions_rutting.append(model.predict(val_scaled)[0])

        # --- TRACÉ DE LA COURBE ---
        df_plot = pd.DataFrame({'Années': ages_simulés, 'Orniérage (mm)': predictions_rutting})
        
        fig = px.line(df_plot, x='Années', y='Orniérage (mm)', 
                      title="Évolution de l'Orniérage sur 10 ans",
                      markers=True, template="plotly_white")
        
        fig.add_hline(y=8.0, line_dash="dash", line_color="red", annotation_text="Seuil Critique (8mm)")
        
        st.plotly_chart(fig, use_container_width=True)

        # Affichage valeur actuelle
        idx_actuel = int(age_actuel) if age_actuel <= 10 else 10
        val_actuelle = predictions_rutting[idx_actuel]
        st.metric("Profondeur à l'âge saisi", f"{val_actuelle:.2f} mm")
        
    else:

        st.info("Veuillez remplir le formulaire et cliquer sur 'Analyser' pour voir l'évolution graphique.")



with tab2:
    st.header("📲 Scanner pour tester sur mobile")
    
    # L'URL de votre application une fois déployée sur Streamlit Cloud
    # Remplacez par votre lien réel après le déploiement
    app_url = "https://ruttingprediction-dqtpfcj2mvd4y3goyq5tkv.streamlit.app/" 
    
    st.info(f"Lien de l'application : {app_url}")

    # Génération du QR Code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(app_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # Conversion pour Streamlit
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # Affichage centré du QR Code
    col_qr1, col_qr2, col_qr3 = st.columns([1, 2, 1])
    with col_qr2:
        st.image(byte_im, caption="Scannez ce QR Code avec votre téléphone", use_container_width=True)


