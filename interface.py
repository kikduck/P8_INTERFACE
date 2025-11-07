## Application Streamlit
import streamlit as st
import requests
from functools import lru_cache
import os

# --- Configuration de l'API ---
# URL de l'API (configurable via variable d'environnement)
API_URL = os.getenv("API_URL", "http://localhost:5000")

# --- Config de la page ---
st.set_page_config(
    page_title="Segmentation d'images",
    page_icon="🎈",
    layout="centered"
)


# Fonction mise en cache pour récupérer la liste des modèles
@st.cache_data(ttl=60)  # Cache pendant 60 secondes
def get_available_models():
    """Récupère la liste des modèles depuis l'API (avec cache)"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        if response.ok:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Erreur API: {e}")
        return None

# --- Interface chargement dossier + sélection image + prédiction segmentation ---

st.title("Segmentation d'images")

# Sélection du modèle
st.sidebar.header("Sélection du modèle")

# Récupérer la liste des modèles disponibles (avec cache)
models_data = get_available_models()

if models_data:
    available_models = [m for m in models_data["models"] if m["available"]]

    # Créer un dict pour le mapping
    model_options = {m["name"]: m["id"] for m in available_models}

    # Trouver le modèle actuel
    current_model_id = models_data.get("current_model", "hrnet")
    current_model_name = next((m["name"] for m in available_models if m["id"] == current_model_id), "HRNet-FPN")

    # Selectbox pour choisir le modèle
    selected_model_name = st.sidebar.selectbox(
        "Modèle de segmentation",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(current_model_name) if current_model_name in model_options.keys() else 0
    )

    selected_model_id = model_options[selected_model_name]

    # Afficher les infos du modèle sélectionné
    selected_model_info = next(m for m in available_models if m["id"] == selected_model_id)
    st.sidebar.write(f"**Description:** {selected_model_info['description']}")
    if 'miou' in selected_model_info:
        st.sidebar.metric("mIoU", f"{selected_model_info['miou']:.4f}")

    # Bouton pour charger le modèle si différent
    if selected_model_id != current_model_id:
        if st.sidebar.button("Charger ce modèle"):
            with st.spinner(f"Chargement du modèle {selected_model_name}..."):
                load_response = requests.post(
                    f"{API_URL}/load_model",
                    json={"model_name": selected_model_id},
                    timeout=30
                )
                if load_response.ok:
                    st.sidebar.success(f"✅ Modèle {selected_model_name} chargé !")
                    # Réinitialiser la segmentation pour forcer une nouvelle prédiction
                    if 'segmentation_result' in st.session_state:
                        st.session_state.segmentation_result = None
                    # Invalider le cache pour forcer une mise à jour
                    get_available_models.clear()
                else:
                    st.sidebar.error(f"❌ Erreur lors du chargement du modèle")
    else:
        st.sidebar.info("✓ Modèle actuellement chargé")
else:
    st.sidebar.error("Impossible de récupérer la liste des modèles")
    selected_model_id = "hrnet"

# 1. Sélection du dossier d'images
st.header("1. Charger un dossier d'images")

# File uploader avec une clé unique pour maintenir l'état
uploaded_files = st.file_uploader(
    "Glissez ou sélectionnez des images (plusieurs possibles)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="image_uploader"
)

if uploaded_files:
    # Initialisation de l'index dans session_state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # S'assurer que l'index est dans les limites
    if st.session_state.current_index >= len(uploaded_files):
        st.session_state.current_index = len(uploaded_files) - 1
    if st.session_state.current_index < 0:
        st.session_state.current_index = 0

    # 2. Sélection d'une image spécifique avec navigation
    st.header("2. Sélectionner une image pour la prédiction")

    # Boutons de navigation
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Précédent", disabled=(st.session_state.current_index == 0)):
            st.session_state.current_index -= 1
            st.rerun()

    with col2:
        st.markdown(f"**Image {st.session_state.current_index + 1} / {len(uploaded_files)}**")

    with col3:
        if st.button("Suivant ➡️", disabled=(st.session_state.current_index >= len(uploaded_files) - 1)):
            st.session_state.current_index += 1
            st.rerun()

    # Image sélectionnée
    file_selected = uploaded_files[st.session_state.current_index]

    # Initialiser le résultat de segmentation dans session_state si nécessaire
    if 'segmentation_result' not in st.session_state:
        st.session_state.segmentation_result = None
    if 'last_predicted_index' not in st.session_state:
        st.session_state.last_predicted_index = None

    # Réinitialiser la segmentation si on change d'image
    if st.session_state.last_predicted_index != st.session_state.current_index:
        st.session_state.segmentation_result = None
        st.session_state.last_predicted_index = st.session_state.current_index

    # 3. Affichage côte à côte de l'image originale et de la segmentation
    st.header("3. Prédire la segmentation")

    # Deux colonnes pour les images
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.image(file_selected, caption=f"Image originale: {file_selected.name}", use_container_width=True)

    with col_img2:
        if st.session_state.segmentation_result is not None:
            st.image(st.session_state.segmentation_result, caption="Résultat de la segmentation", use_container_width=True)
        else:
            # Placeholder vide si pas encore de prédiction
            st.info("Cliquez sur 'Prédire' pour voir la segmentation")

    # Bouton "Prédire" centré en dessous
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("🔮 Prédire la segmentation", use_container_width=True):
            # Envoyer l'image à l'API
            api_url = f"{API_URL}/predict"
            files = {"image": (file_selected.name, file_selected.getvalue(), file_selected.type)}
            try:
                with st.spinner("Segmentation en cours..."):
                    response = requests.post(api_url, files=files)
                if response.ok:
                    # L'API retourne directement une image PNG
                    from io import BytesIO
                    from PIL import Image
                    segment_img = Image.open(BytesIO(response.content))
                    st.session_state.segmentation_result = segment_img
                    st.rerun()
                else:
                    st.error(f"Erreur lors de la requête: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
else:
    st.info("Veuillez charger des images pour commencer.")

# --- Footer ---
st.caption("Segmentation d'images – © 2025")