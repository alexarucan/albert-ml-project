import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer

st.set_page_config(page_title="Analyse des tumeurs mammaires", layout="wide")

tab1, tab2 = st.tabs(["Prédicteur de tumeur", "Robustesse des modèles"])

@st.cache_data
def get_clean_data():
    df = pd.read_csv("data/raw/data.csv")
    df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

# ========================
# ONGLET 1 - PREDICTEUR
# ========================
with tab1:
    st.title("Prédicteur de cancer du sein")

    def add_sidebar():
        st.sidebar.title("Mesures des noyaux cellulaires")
        if st.sidebar.button("Réinitialiser les paramètres"):
            st.experimental_rerun()

        df = get_clean_data()

        grouped = {
            "Taille et forme du noyau": [
                ("Rayon (moyen)", "radius_mean"),
                ("Périmètre (moyen)", "perimeter_mean"),
                ("Aire (moyenne)", "area_mean"),
            ],
            "Texture et surface": [
                ("Texture (moyenne)", "texture_mean"),
                ("Lissage (moyen)", "smoothness_mean"),
                ("Symétrie (moyenne)", "symmetry_mean"),
                ("Dimension fractale (moyenne)", "fractal_dimension_mean"),
            ],
            "Compacité et concavité": [
                ("Compacité (moyenne)", "compactness_mean"),
                ("Concavité (moyenne)", "concavity_mean"),
                ("Points concaves (moyen)", "concave points_mean"),
            ],
            "Écarts types": [
                ("Rayon (écart-type)", "radius_se"),
                ("Texture (écart-type)", "texture_se"),
                ("Périmètre (écart-type)", "perimeter_se"),
                ("Aire (écart-type)", "area_se"),
                ("Lissage (écart-type)", "smoothness_se"),
                ("Compacité (écart-type)", "compactness_se"),
                ("Concavité (écart-type)", "concavity_se"),
                ("Points concaves (écart-type)", "concave points_se"),
                ("Symétrie (écart-type)", "symmetry_se"),
                ("Dimension fractale (écart-type)", "fractal_dimension_se"),
            ],
            "Mesures extrêmes (valeurs les plus élevées)": [
                ("Rayon (worst)", "radius_worst"),
                ("Texture (worst)", "texture_worst"),
                ("Périmètre (worst)", "perimeter_worst"),
                ("Aire (worst)", "area_worst"),
                ("Lissage (worst)", "smoothness_worst"),
                ("Compacité (worst)", "compactness_worst"),
                ("Concavité (worst)", "concavity_worst"),
                ("Points concaves (worst)", "concave points_worst"),
                ("Symétrie (worst)", "symmetry_worst"),
                ("Dimension fractale (worst)", "fractal_dimension_worst"),
            ]
        }

        inputs = {}
        for section, sliders in grouped.items():
            with st.sidebar.expander(section):
                for label, key in sliders:
                    inputs[key] = st.slider(
                        label,
                        min_value=0.0,
                        max_value=float(df[key].max()),
                        value=float(df[key].mean())
                    )
        return inputs

    def get_scaled_values(inputs):
        df = get_clean_data()
        X = df.drop('diagnosis', axis=1)
        return {k: (v - X[k].min()) / (X[k].max() - X[k].min()) for k, v in inputs.items()}

    def get_radar_chart(inputs):
        data = get_scaled_values(inputs)
        categories = ['Rayon', 'Texture', 'Périmètre', 'Aire',
                      'Lissage', 'Compacité', 'Concavité',
                      'Points concaves', 'Symétrie', 'Fractale']
        suffixes = [('mean', 'Valeur moyenne', '#f7a6b9'),
                    ('se', 'Écart-type', '#f9cad7'),
                    ('worst', 'Valeur extrême', '#d36b99')]

        fig = go.Figure()
        for suffix, label, color in suffixes:
            values = [data[f"{cat.lower()}_{suffix}"] for cat in [
                'radius', 'texture', 'perimeter', 'area',
                'smoothness', 'compactness', 'concavity',
                'concave points', 'symmetry', 'fractal_dimension']]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=label,
                line=dict(color=color),
                fillcolor=color,
                opacity=0.5
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600
        )
        return fig

    def add_predictions(inputs):
        model = pickle.load(open("modele/best_model.pkl", "rb"))
        scaler = pickle.load(open("modele/scaler.pkl", "rb"))

        X = np.array(list(inputs.values())).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        benign, malignant = round(proba[0], 3), round(proba[1], 3)

        st.subheader("Résultat du modèle")
        if pred == 0:
            if malignant >= 0.2:
                st.error("Suspicion forte de tumeur maligne (>= 20%)")
            elif malignant >= 0.1:
                st.warning("Risque modéré de malignité (entre 10% et 20%)")
            else:
                st.success("Tumeur bénigne selon le modèle")
        else:
            st.error("Tumeur maligne selon le modèle")

        st.write(f"Probabilité bénigne : {benign}")
        st.write(f"Probabilité maligne : {malignant}")
        st.write("---")

        if malignant >= 0.2:
            st.markdown("**Recommandation :** Consultez un professionnel de santé.")
        elif malignant >= 0.1:
            st.markdown("**Recommandation :** Surveillance ou second avis recommandé.")
        else:
            st.markdown("**Recommandation :** Aucun signe inquiétant immédiat.")

        st.markdown("> Cet outil ne remplace pas un avis médical professionnel.")

    inputs = add_sidebar()
    col1, col2 = st.columns([4, 1])
    with col1:
        st.plotly_chart(get_radar_chart(inputs))
    with col2:
        add_predictions(inputs)

# ========================
# ONGLET 2 - ROBUSTESSE
# ========================
with tab2:
    st.title("Impact des données manquantes sur les modèles")

    def add_noise(df, noise_level):
        df_noise = df.copy()
        num_cols = df.select_dtypes(include=np.number).columns
        total = df.shape[0] * len(num_cols)
        n_nan = int(total * noise_level)
        for _ in range(n_nan):
            i = np.random.randint(0, df.shape[0])
            j = np.random.choice(num_cols)
            df_noise.at[i, j] = np.nan
        return df_noise

    def impute_data(X_train, X_test):
        mean_cols = [c for c in X_train.columns if "_mean" in c]
        se_cols = [c for c in X_train.columns if "_se" in c]
        worst_cols = [c for c in X_train.columns if "_worst" in c]

        imp_mean = SimpleImputer(strategy="mean")
        imp_med = SimpleImputer(strategy="median")
        imp_knn = KNNImputer(n_neighbors=5)

        X_train_imp = pd.concat([
            pd.DataFrame(imp_mean.fit_transform(X_train[mean_cols]), columns=mean_cols, index=X_train.index),
            pd.DataFrame(imp_med.fit_transform(X_train[se_cols]), columns=se_cols, index=X_train.index),
            pd.DataFrame(imp_knn.fit_transform(X_train[worst_cols]), columns=worst_cols, index=X_train.index),
        ], axis=1)

        X_test_imp = pd.concat([
            pd.DataFrame(imp_mean.transform(X_test[mean_cols]), columns=mean_cols, index=X_test.index),
            pd.DataFrame(imp_med.transform(X_test[se_cols]), columns=se_cols, index=X_test.index),
            pd.DataFrame(imp_knn.transform(X_test[worst_cols]), columns=worst_cols, index=X_test.index),
        ], axis=1)

        scaler = StandardScaler()
        return scaler.fit_transform(X_train_imp), scaler.transform(X_test_imp)

    def compute_confusion(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    noise = st.slider("Niveau de bruit injecté :", 0.0, 0.5, 0.0, step=0.05)

    df = get_clean_data()
    y = df["diagnosis"]
    X = add_noise(df.drop(columns=["diagnosis"]), noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = impute_data(X_train, X_test)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=10000), {"C": [0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(), {"n_estimators": [100, 200]}),
        "SVM": (SVC(probability=True), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    }

    results = []
    confs = {}
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, param_grid=params, cv=5, scoring=make_scorer(f1_score))
        grid.fit(X_train_scaled, y_train)
        y_pred = grid.best_estimator_.predict(X_test_scaled)

        results.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Précision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        })

        confs[name] = compute_confusion(y_test, y_pred)

    df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    df_confs = pd.DataFrame(confs).T.reset_index().rename(columns={"index": "Modèle"})
    df_melt = df_confs.melt(id_vars="Modèle", var_name="Type", value_name="Valeur")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_results, x="F1 Score", y="Modèle", orientation="h",
                      color_discrete_sequence=["#f2a3b3"], range_x=[0.8, 1.0])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(df_melt, x="Type", y="Valeur", color="Modèle", barmode="group",
                      color_discrete_sequence=["#f7c5d1", "#f4a0b0", "#e6739f"])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Interprétations cliniques")
    col_a, col_b, col_c = st.columns(3)
    for i, row in enumerate(df_results.itertuples()):
        name = row.Modèle
        recall = round(row.Recall, 2)
        precision = round(row.Précision, 2)
        values = confs[name]

        msg = f"""
        <div style='background-color:#fde8e8; padding:15px; border-radius:10px;'>
        <strong>{name}</strong><br>
        Sensibilité : {recall} — détecte {int(recall * 100)}% des cancers<br>
        Précision : {precision} — {int(precision * 100)}% des cas positifs sont vrais<br>
        Détail : {values['TP']} TP | {values['FP']} FP | {values['FN']} FN | {values['TN']} TN<br>
        Avec {int(noise * 100)}% de bruit injecté
        </div>
        """
        if i == 0:
            col_a.markdown(msg, unsafe_allow_html=True)
        elif i == 1:
            col_b.markdown(msg, unsafe_allow_html=True)
        else:
            col_c.markdown(msg, unsafe_allow_html=True)

