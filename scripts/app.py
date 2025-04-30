import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

# Chargement du modèle et du scaler
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Fonction pour obtenir les données nettoyées
def get_clean_data():
    # Charger les données d'entraînement
    df = pd.read_csv("data/data.csv")
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

# Fonction pour créer un radar chart
def get_radar_chart(input_data):
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    
    return fig

# Fonction pour obtenir les données d'entrée de l'utilisateur
def get_user_input():
    data = get_clean_data()

    # Liste des caractéristiques à utiliser pour les sliders (en excluant 'diagnosis')
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # Créer un dictionnaire d'entrées utilisateur
    input_dict = {}
    
    for label, key in slider_labels:
        # Assurer que la colonne existe dans les données
        if key in data.columns:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
    
    return input_dict


# Fonction pour prédire avec les données entrées par l'utilisateur
def predict(input_data):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    
    # Standardiser les données d'entrée
    scaled_dict = {}
    for key, value in input_data.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    input_array = np.array(list(scaled_dict.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    proba = model.predict_proba(input_array_scaled)[0]
    
    return prediction[0], proba

# Fonction principale de l'application Streamlit
def main():
    st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
    
    st.title("Breast Cancer Prediction App")
    st.write("This app helps predict whether a breast tumor is malignant or benign based on cell measurements.")
    
    # Récupérer les données d'entrée de l'utilisateur
    user_input = get_user_input()

    # Créer les colonnes pour la disposition à deux colonnes (gauche et droite)
    col1, col2 = st.columns([4, 1])  # 4: pour plus de largeur à gauche

    # Afficher le radar chart dans la première colonne
    with col1:
        radar_chart = get_radar_chart(user_input)
        st.plotly_chart(radar_chart)
    
    # Afficher les résultats de la prédiction dans la deuxième colonne
    with col2:
        prediction, proba = predict(user_input)

        # Affichage du type de tumeur
        if prediction == 0:
            st.write("✅ **Benign** Tumor")
        else:
            st.write("⚠️ **Malignant** Tumor")
        
        # Affichage des probabilités
        st.write(f"**Probability of Benign:** {proba[0]:.2f}")
        st.write(f"**Probability of Malignant:** {proba[1]:.2f}")
        
        st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

if __name__ == "__main__":
    main()
