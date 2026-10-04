import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
from sklearn.preprocessing import normalize


# Configuration de la page
st.set_page_config(
    page_title="Segmentation des clients - modèle DBSCAN",
    page_icon="📊",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de clustering (DBSCAN) permet de déterminer à quelle classe "
    "appartient un client à partir de ses caractéristiques d'utilisation."
)


# Chargement des artefacts (mis en cache : chargés une seule fois)

@st.cache_resource
def load_artifacts():
    artefacts = jb.load("modele_dbscan.joblib")
    return artefacts


artefacts = load_artifacts()
points_coeur = np.array(artefacts["points_coeur"])
labels_coeur = np.array(artefacts["labels_coeur"])
eps = artefacts["eps"]
colonnes = artefacts["colonnes"]
valeurs_defaut = artefacts["valeurs_defaut"]
noms_classes = artefacts["noms_classes"]


# Fonction de prédiction simple (basée sur predire_classe)

def pred_func(valeurs):
    client = np.array(valeurs, dtype=float).reshape(1, -1)
    client = normalize(client)

    distances = np.linalg.norm(points_coeur - client, axis=1)
    plus_proche = distances.argmin()

    if distances[plus_proche] <= eps:
        label = labels_coeur[plus_proche]
    else:
        label = -1  # anomalie / bruit

    nom_classe = noms_classes.get(label, f"Classe {label}") if isinstance(noms_classes, dict) else f"Classe {label}"
    return nom_classe


# Fonction de prédiction multiple

def pred_func_csv(file):
    df = pd.read_csv(file)
    prediction = []

    for i, row in enumerate(df[colonnes].values):
        try:
            resultat = pred_func(row)
            prediction.append(resultat)
        except Exception as e:
            print(f"Erreur ligne {i} : {row} → {e}")
            prediction.append(None)

    df['classe_predite'] = prediction
    return df


# Interface

st.title("📊 Segmentation des clients - modèle DBSCAN")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire la classe d'un client avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        valeurs_saisies = []
        # Génère deux colonnes de champs pour tenir le formulaire compact
        cols = st.columns(2)
        for i, col_nom in enumerate(colonnes):
            with cols[i % 2]:
                valeur = st.number_input(
                    col_nom,
                    value=float(valeurs_defaut[i]),
                    format="%.2f"
                )
                valeurs_saisies.append(valeur)

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = pred_func(valeurs_saisies)
            st.success(f"**Ce client appartient à :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire la classe de plusieurs clients")
    st.write(DESCRIPTION)
    st.caption(f"Le fichier CSV doit contenir, dans cet ordre, les colonnes : {', '.join(colonnes)}.")

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")