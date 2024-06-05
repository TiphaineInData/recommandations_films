import streamlit as st
import pandas as pd
from unidecode import unidecode
import yaml
from yaml.loader import SafeLoader
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import base64
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# j'importe les données, unidecode sur le titre pour la recherche sans accent, minuscule...
df = pd.read_csv("https://raw.githubusercontent.com/TiphaineInData/recommandations_films/main/dataframe2.csv", sep=',')
df['originalTitle_normalized'] = df['originalTitle'].apply(lambda x: unidecode(x.lower()))
df_realisateurs = pd.read_csv("df_real_71.csv", sep=',')

# je retire les stop words sur les 3 colonnes
tfidf_overview = TfidfVectorizer(stop_words='english') 
tfidf_genre = TfidfVectorizer(stop_words='english')
tfidf_nconst = TfidfVectorizer(stop_words='english')

 #je transforme en matrice, et apprend les occurences de tokens
tfidf_overview_matrix = tfidf_overview.fit_transform(df['overview_lemm']) 
tfidf_genre_matrix = tfidf_genre.fit_transform(df['genre_lemm'])
tfidf_nconst_matrix = tfidf_nconst.fit_transform(df['nconst'])


#mise à l'échelle des variables numériques
scaler = StandardScaler() 
numeric_features = df[['averageRating', 'startYear']]
scaled_numeric_features = scaler.fit_transform(numeric_features)

# Conversion des caractéristiques numériques en matrice sparse pr stocker les éléments non nuls et réduire la mémoire
scaled_numeric_matrix = sp.csr_matrix(scaled_numeric_features)

# Création de la matrice pour original_language avec OneHotEncoder (matrice sparse comme un get_dummies mais qui ne retient que les valeurs non nulles)
onehot_encoder = OneHotEncoder()
original_language_matrix = onehot_encoder.fit_transform(df[['original_language']])

# Je donne du poids aux variables pour le ML
weights = {
    'tfidf_overview': 1,
    'tfidf_genre': 1,
    'tfidf_nconst': 1,
    'original_language': 3,
    'average_rating': 2,
    'start_year': 2
}

# j'applique les poids
weighted_tfidf_overview_matrix = tfidf_overview_matrix * weights['tfidf_overview']
weighted_tfidf_genre_matrix = tfidf_genre_matrix * weights['tfidf_genre']
weighted_tfidf_nconst_matrix = tfidf_nconst_matrix * weights['tfidf_nconst']
weighted_original_language_matrix = original_language_matrix * weights['original_language']
weighted_numeric_matrix = scaled_numeric_matrix * weights['average_rating']

# association des variables 
combined_features = sp.hstack([
    weighted_tfidf_overview_matrix,
    weighted_tfidf_genre_matrix,
    weighted_tfidf_nconst_matrix,
    weighted_original_language_matrix,
    weighted_numeric_matrix
])

# J'entraîne le modèle
knn = NearestNeighbors(metric='cosine', algorithm='auto')
knn.fit(combined_features)

# Spécifiez le chemin absolu du fichier config.yaml
config_path = 'C:/Users/Admin/Documents/pop_data/config.yaml'

# Charger la configuration
if os.path.exists(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
else:
    st.error(f"Le fichier config.yaml est introuvable à l'emplacement : {config_path}")

# Fonction simple pour vérifier les identifiants
def check_credentials(email, password):
    for user, info in config['credentials']['usernames'].items():
        if info['email'] == email and info['password'] == password:
            return info['name']
    return None

# Fonction pour ajouter une image de fond à partir d'un fichier local
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# je définis une fonction pour tronquer le texte à 200 mots si l'overview dépasse 200 mots
def truncate_text(text, max_length=200):
    return text if len(text) <= max_length else text[:max_length] + '...' 

# Fonction pour trouver les films similaires
def recommendations(df, titre, nb_recommendations=4): #nous souhaitons 4 nearest neighbors
    # Utiliser unidecode pour supprimer les accents du titre recherché
    titre_normalized = unidecode(titre.lower())
    
    # je créé une variable avec les titres insensibles à la casse et aux accents
    df['originalTitle_normalized'] = df['originalTitle'].apply(lambda x: unidecode(x.lower()))
    
    # Rechercher les indices correspondants au titre normalisé
    matched_indices = df[df['originalTitle_normalized'] == titre_normalized].index
    
    if matched_indices.empty:
        # Si aucun titre exact n'est trouvé, chercher les titres similaires
        possible_titles = df["originalTitle"][df['originalTitle_normalized'].str.contains(titre_normalized, case=False, na=False)].tolist()
        if not possible_titles:
            return f"Le titre '{titre}' n'existe pas dans le dataset."
        return f"Le titre '{titre}' n'existe pas dans le dataset. Peut-être cherchiez-vous : {possible_titles}"
    
    # Trouve l'indice du film, avec KNN trouve les plus proches voisins, s'exclut des plus proches voisins, et retourne les films similaires
    indice = matched_indices[0]
    distances, indices = knn.kneighbors(combined_features[indice], n_neighbors=nb_recommendations + 1)
    similar_indices = indices.flatten()[1:]
    return df['originalTitle'].iloc[similar_indices].tolist() 


# Fonction pour afficher la page principale
def main_page():
    # Lire le fichier CSV
    df = pd.read_csv("dataframe2.csv", sep=',')

    # Créer une nouvelle colonne avec
    df['primaryNameLower'] = df['primaryName'].apply(lambda x: ', '.join([unidecode(name.lower()) for name in x.split(', ')])) 

    # Vérifier et corriger les valeurs de `poster_path_film`
    default_image_path = 'img_film_manqu.jpg'  
    df['poster_path_test'] = df['poster_path_test'].apply(lambda x: x if isinstance(x, str) else default_image_path)

    st.markdown(
        """
        <style>
        .film-container {
            background-color: #FAE3C1; /* Couleur de fond */
            border-radius: 15px; /* Coins arrondis */
            padding: 20px; /* Espacement interne */
            margin: 20px 0; /* Espacement externe */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Ombre */
            display: flex;
            align-items: center;
            height: 400px; /* Hauteur fixe pour les carrés */
        }
        .film-image {
            flex: 1;
            margin-right: 20px;
        }
        .highlight {
            color: #983D30;
            font-size: 20px;
        }
        .highlight-large {
            color: #983D30;
            font-size: 24px;
        }
        .film-description {
            flex: 2;
            color: #003366; /* Couleur du texte */
            overflow: hidden; /* Masquer le texte débordant */
        }
        .film-title {
            font-size: 20px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        .block-container {
            background-color: #FAE3C1 !important;
            color: #983D30 !important;
            font-size: 20px !important;
            width: 100% !important;
            padding: 0 !important;
        }
        .main .block-container {
            padding: 1rem !important;
            width: calc(100% - 2rem) !important;
        }
        .st-emotion-cache-13ln4jf {
            max-width: none !important; /* Remove max-width limitation */
        }
        body {
            background-color: black !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .highlight {
            color: #983D30;
        }
        .highlight-large {
            color: #983D30;
            font-size: 24px;
        }
        .header {
            background-color: black !important;
            padding: 10px;
            text-align: center;
            width: 100%;
            top: 0;
            left: 0;
        }
        textarea {
            min-height: 196px !important;
        }
        .yellow-text {
            color: #black !important;
        }
        .stTextInput > div > div > input {
            color: black !important;
            background-color: white !important;
        }
        .stTextInput input::placeholder {
            color: black !important; /* Placeholder text color */
        }
        .content {
            background-color: black !important;
            color: #983D30 !important;
            width: 100% !important;
        }
        .stImage > div {
            background-color: black !important;
            width: 100% !important;
        }
        .stImage img {
            background-color: black !important;
            width: 100% !important;
        }
        .stButton button {
            background-color: #215362 !important;
            border: #215362 !important;
            color: #FAE3C1 !important;
        }
        .button-row {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #983D30 !important; /* Title colors */
        }
        .stTextInput {
            margin: auto;
        }
        .stTextInput label {
            color: #983D30 !important;
        }
        h1 {
            color: #983D30!important;
        }
        .stColumns {
            margin-bottom: 20px !important;
        }
        .film-row {
            margin-bottom: 40px !important; /* Ajouter de l'espace entre les lignes de films */
        }
        .result-title {
            font-size: 20px !important;
            font-weight: bold !important.
        }
        .film-title {
            font-size: 20px !important.
            font-weight: bold !important.
        .sub-header {
            font-size: 24px !important;  /* Taille de la police pour les sous-titres */
            color: #983D30 !important;  /* Couleur du texte pour les sous-titres */
        }
        .results-title {
            font-size: 22px;
            color: #003366; /* Couleur pour le texte "Résultats pour" */
            font-weight: bold;
            margin-bottom: 20px;
        }
        .similar-films-title {
            font-size: 22px;
            color: #983D30; /* Couleur pour le texte "Films similaires" */
            font-weight: bold;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Contenu de l'application
    st.markdown('<div class="content">', unsafe_allow_html=True)

    # En-tête avec l'image
    st.markdown('<div class="full-width-image">', unsafe_allow_html=True)
    st.image('https://raw.githubusercontent.com/TiphaineInData/recommandations_films/main/back_main4.jpg', use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bouton pour naviguer vers la page "Idées et indicateurs"
    col1, col2, col3 = st.columns([8, 1, 1])
    with col3:
        if st.button("Idées et indicateurs", key="btn_idees_indicateurs"):
            st.session_state.page = "idees"
            st.experimental_rerun()

    st.text("")
    st.text("")
    st.text("")
    
    # Recherche par titre de film
    st.markdown('<h2 class="sub-header">Recherche par titre de film :</h2>', unsafe_allow_html=True)

    if 'titre' not in st.session_state:
        st.session_state.titre = ''

    def clear_text2():
        st.session_state.titre = ''

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        titre = st.text_input(label ="",value=st.session_state.titre, placeholder="Entrez un titre de film", key='titre')

    if titre:
        titre = unidecode(titre)
        resultat = recommendations(df, titre, nb_recommendations=4)
        
        if isinstance(resultat, str):  # Vérifiez si resultat est un message d'erreur
            st.markdown(f"<div class='highlight'>{resultat}</div>", unsafe_allow_html=True) #prend les strings
        else:
            st.markdown("<div class='similar-films-title'>Films similaires :</div>", unsafe_allow_html=True)
            for i in range(0, len(resultat), 2):
                col1, col2 = st.columns(2)
                for j, col in enumerate([col1, col2]):
                    if i + j < len(resultat):
                        title = resultat[i + j]
                        film_df = df[df['originalTitle'] == title]
                        if not film_df.empty:
                            row = film_df.iloc[0]  # Récupérer la ligne correspondant au titre
                            primary_names = sorted(set(row['primaryName'].split(', ')))
                            primary_names_str = ', '.join(primary_names)
                            art_et_essai_text = 'FILM ART ET ESSAI' if row['art_et_essai'] == 1 else ''
                            with col:
                                st.markdown(
                                    f"""
                                    <div class="film-container">
                                        <div class="film-image">
                                            <img src="{row['poster_path_test']}" width="150px">
                                        </div>
                                        <div class="film-description">
                                            <div class="film-title">{row['originalTitle'].upper()}</div>
                                            <p><strong>{art_et_essai_text}</strong></p>
                                            <p><strong>Genres:</strong> {row['genres']}</p>
                                            <p><strong>Année:</strong> {row['startYear']}</p>
                                            <p><strong>Participants:</strong> {primary_names_str}</p>
                                            <p><strong>Description:</strong> {truncate_text(row['overview'], 300)}</p>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

            st.markdown("<div class='film-row'></div>", unsafe_allow_html=True)  # Ajouter de l'espace entre les lignes de films

            # Suppression des boutons "Voir plus" et "Voir moins" pour la recherche de films similaires

            st.markdown('<div class="button-row">', unsafe_allow_html=True)
            col1, col3 = st.columns([1, 1])
            with col1:
                if st.button('Rafraîchir', on_click=clear_text2, key='refresh_similar'):
                    st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.text("")
    st.text("")
    st.text("")

    # Recherche par nom
    st.markdown('<h2 class="sub-header">Recherche par nom :</h2>', unsafe_allow_html=True)

    # Input de l'utilisateur
    if 'name' not in st.session_state:
        st.session_state.name = ''

    def clear_text():
        st.session_state.name = ''
        st.session_state.num_movies = num_movies_to_display

    # Définir les colonnes
    col1, col2, col3 = st.columns([1, 2, 1])

    # Ajouter un input de texte dans la première colonne plus petite
    with col1:
        name = st.text_input(label="", placeholder="Entrez un nom d'actrice, acteur, réalisatrice (...)", value=st.session_state.name, key='name')

    # Convertir le texte en minuscules et retirer les accents
    name2 = unidecode(name.lower())

    num_movies_to_display = 4

    # je filtre le df en fonction de l'input
    if name:
        filtered_df = df[df["primaryNameLower"].str.contains(name2, case=False, na=False)]
        filtered_df = filtered_df.sort_values(by="averageRating", ascending=False)
        
        # Utiliser session state pour gérer la pagination
        if 'num_movies' not in st.session_state:
            st.session_state.num_movies = num_movies_to_display

        movies_to_show = filtered_df.head(st.session_state.num_movies)

        if not movies_to_show.empty:
            st.markdown(f"<div class='results-title highlight-large'>Résultats pour '{name}':</div>", unsafe_allow_html=True)

            for i in range(0, len(movies_to_show), 2):
                col1, col2 = st.columns(2)
                for j, col in enumerate([col1, col2]):
                    if i + j < len(movies_to_show):
                        row = movies_to_show.iloc[i + j]
                        art_et_essai_text = 'FILM ART ET ESSAI' if row['art_et_essai'] == 1 else ''
                        primary_names = sorted(set(row['primaryName'].split(', ')))
                        primary_names_str = ', '.join(primary_names)
                        with col:
                            st.markdown(
                                f"""
                                <div class="film-container">
                                    <div class="film-image">
                                        <img src="{row['poster_path_test']}" width="150px">
                                    </div>
                                    <div class="film-description">
                                        <div class="film-title">{row['originalTitle'].upper()}</div>
                                        <p><strong>{art_et_essai_text}</strong></p>
                                        <p><strong>Genres:</strong> {row['genres']}</p>
                                        <p><strong>Année:</strong> {row['startYear']}</p>
                                        <p><strong>Participants:</strong> {primary_names_str}</p>
                                        <p><strong>Description:</strong> {truncate_text(row['overview'], 300)}</p>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            st.markdown('<div class="button-row">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button('Voir moins', key='see_less'):
                    if st.session_state.num_movies > num_movies_to_display:
                        st.session_state.num_movies -= num_movies_to_display
                        st.experimental_rerun()
            with col2:
                if st.button('Voir plus', key='see_more'):
                    st.session_state.num_movies += num_movies_to_display
                    st.experimental_rerun()
            with col3:
                if st.button('Rafraîchir', on_click=clear_text, key='refresh'):
                    st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='highlight'>Aucun film trouvé pour '{name}'</div>", unsafe_allow_html=True)

    # Close the div
    st.markdown('</div>', unsafe_allow_html=True)






def idees_page():
    st.markdown(
        """
        <style>
        .block-container {
            background-color: #FAE3C1 !important;
            color: #983D30 !important;
            width: 100% !important;
            padding: 0 !important;
        }
        .main .block-container {
            padding: 1rem !important;
            width: calc(100% - 2rem) !important;
        }
        .st-emotion-cache-13ln4jf {
            max-width: none !important; /* Remove max-width limitation */
        }
        body {
            background-color: black !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .header {
            background-color: black !important;
            padding: 10px;
            text-align: center;
            width: 100%;
            top: 0;
            left: 0;
        }
        textarea {
            min-height: 196px !important;
        }
        .yellow-text {
            color: #black !important;
        }
        .stTextInput > div > div > input {
            color: black !important;
            background-color: white !important;
        }
        .stTextInput input::placeholder {
            color: black !important; /* Placeholder text color */
        }
        .content {
            background-color: black !important;
            color: #983D30 !important;
            width: 100% !important;
        }
        .stImage > div {
            background-color: black !important;
            width: 100% !important;
        }
        .stImage img {
            background-color: black !important;
            width: 100% !important;
        }
        .stButton button {
            background-color: #215362 !important;
            border: #215362 !important;
            color: #FAE3C1 !important;
        }
        .button-row {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #983D30 !important; /* Title colors */
        }
        .stTextInput {
            margin: auto;
        }
        .stTextInput label {
            color: #983D30 !important;
        }
        h1 {
            color: #983D30 !important;
        }
        .stColumns {
            margin-bottom: 20px !important;
        }
        .film-row {
            margin-bottom: 40px !important; /* Ajouter de l'espace entre les lignes de films */
        }
        .result-title {
            font-size: 20px !important;
            font-weight: bold !important.
        }
        .film-title {
            font-size: 20px !important.
            font-weight: bold !important.
        .sub-header {
            font-size: 24px !important;  /* Taille de la police pour les sous-titres */
            color: #983D30 !important;  /* Couleur du texte pour les sous-titres */
        }
        .results-title {
            font-size: 22px;
            color: #003366; /* Couleur pour le texte "Résultats pour" */
            font-weight: bold;
            margin-bottom: 20px;
        }
        .similar-films-title {
            font-size: 24px;
            color: #983D30; /* Couleur pour le texte "Films similaires" */
            font-weight: bold;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .director-container {
            background-color: #FAE3C1;
            padding: 20px;
            margin: 20px 0;
            display: flex;
            justify-content: space-around;
            align-items: center;
            height: 400px;
            width: 100%;
        }
        .director {
            text-align: center;
            margin: 0 10px;
        }
        .director img {
            width: 150px;
            height: 220px;
            object-fit: cover; /* Cette propriété assure que l'image s'ajuste sans déformation */
            border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="content">', unsafe_allow_html=True)

    # En-tête avec l'image
    st.markdown('<div class="full-width-image">', unsafe_allow_html=True)
    st.image('https://raw.githubusercontent.com/TiphaineInData/recommandations_films/main/back_main4.jpg', use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bouton pour naviguer vers la page "Recommandations"
    if st.button("Recommandations", key="btn_recommandations"):
        st.session_state.page = "main"
        st.experimental_rerun()
        st.title("")
        st.title("")

    # Sélection de films "art et essai"
    st.title("")
    st.title("")
    st.title("")
    st.markdown('<h2 class="sub-header">Sélection de films "art et essai"</h2>', unsafe_allow_html=True)


    if 'new_sample_films' not in st.session_state:
        st.session_state.new_sample_films = False

    if st.session_state.new_sample_films:
        sample_art = df.loc[df['art_et_essai'] == True, ['originalTitle', 'genres', 'poster_path_test']].sample(8)
        st.session_state.sample_art = sample_art
        st.session_state.new_sample_films = False
    else:
        if 'sample_art' not in st.session_state:
            sample_art = df.loc[df['art_et_essai'] == True, ['originalTitle', 'genres', 'poster_path_test']].sample(8)
            st.session_state.sample_art = sample_art
        else:
            sample_art = st.session_state.sample_art

    st.markdown('<div class="film-row">', unsafe_allow_html=True)
    cols = st.columns(8)
    for i, col in enumerate(cols):
        with col:
            row = sample_art.iloc[i]
            st.markdown(f"**{row['originalTitle'].upper()}**")
            st.markdown(f"*{row['genres']}*")
            st.image(row['poster_path_test'], width=150)
    st.markdown('</div>', unsafe_allow_html=True)


    if st.button("Générer d'autres films", key="btn_generer_films"):
        st.session_state.new_sample_films = True
        st.experimental_rerun()


    # Section des réalisateurs
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.markdown('<h2 class="sub-header">Sélection de réalisateurs plébiscités par les spectateurs</h2>', unsafe_allow_html=True)


    if 'new_sample_realisateurs' not in st.session_state:
        st.session_state.new_sample_realisateurs = False

    if st.session_state.new_sample_realisateurs:
        df_realisateurs = pd.read_csv("df_real_71.csv")
        sample_realisateurs = df_realisateurs.sample(8)
        st.session_state.sample_realisateurs = sample_realisateurs
        st.session_state.new_sample_realisateurs = False
    else:
        if 'sample_realisateurs' not in st.session_state:
            df_realisateurs = pd.read_csv("df_real_71.csv")
            sample_realisateurs = df_realisateurs.sample(8)
            st.session_state.sample_realisateurs = sample_realisateurs
        else:
            sample_realisateurs = st.session_state.sample_realisateurs

    st.markdown('<div class="director-container">', unsafe_allow_html=True)
    # j'affiche les réalisateurs dans une même ligne avec des colonnes
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    columns = [col1, col2, col3, col4, col5, col6, col7, col8]

    for col, (_, row) in zip(columns, sample_realisateurs.iterrows()):
        with col:
            st.markdown(f"<div class='director'>", unsafe_allow_html=True)
            st.markdown(f"<div class='director-name'>{row['nom_realisateur'].upper()}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='director-rating'>Moyenne : {row['averageRating']:.2f}</div>", unsafe_allow_html=True)
            st.title("")
            st.image(row['photo_url'], width=150)
            st.markdown(f"</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    if st.button("Générer d'autres réalisateurs", key="btn_generer_realisateurs"):
        st.session_state.new_sample_realisateurs = True
        st.experimental_rerun()
    
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")

        # Ajouter l'espace à la suite de la page et intégrer le lien vers la page web
    st.markdown('<h2 class="sub-header" style="text-align: left;">Fréquentation Cinématographique</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: left;">
            <a href="https://www.cnc.fr/cinema/etudes-et-rapports/statistiques/frequentation-cinematographique" target="_blank">Cliquez ici pour voir les statistiques de fréquentation cinématographique</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("")
    st.title("")
    st.title("")
    st.markdown('<h2 class="sub-header" style="text-align: left;">Association Art et Essai</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: left;">
            <a href="https://www.art-et-essai.org/" target="_blank">Cliquez ici pour accéder au site de l'association</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("")
    st.title("")
    st.title("")

# Interface de connexion
def login_page():
    st.markdown(
        """
        <style>
        .block-container {
            background-color: #FAE3C1 !important;
            color: #983D30 !important;
            width: 100% !important;
            padding: 0 !important;
        }
        .main .block-container {
            padding: 1rem !important;
            width: calc(100% - 2rem) !important;
        }
        .st-emotion-cache-13ln4jf {
            max-width: none !important; /* Remove max-width limitation */
        }
        body {
            background-color: black !important;
            margin: 0 !important;
            padding: 0 !important.
        }
        .header {
            background-color: black !important;
            padding: 10px.
            text-align: center.
            width: 100%.
            top: 0.
            left: 0.
        }
        textarea {
            min-height: 196px !important.
        }
        .yellow-text {
            color: #983D30 !important.
        }
        .stTextInput > div > div > input {
            color: black !important.
            background-color: white !important.
        }
        .stTextInput input::placeholder {
            color: black !important; /* Placeholder text color */
        }
        .content {
            background-color: black !important.
            color: #983D30 !important.
            width: 0% !important.
        }
        .stImage > div {
            background-color: black !important.
            width: 100% !important.
        }
        .stImage img {
            width: 100% !important;
            height: auto !important;
            background-color: black !important;
        }
        .stButton button {
            background-color: #215362 !important.
            border: 2px solid #215362 !important.
            color: #215362 !important.
        }
        .button-row {
            display: flex.
            justify-content: center.
            gap: 10px.
            margin-top: 20px.
        }
        h1, h2, h3, h4, h5, h6 {
            color: #983D30 !important; /* Title colors */
        }
        .stTextInput {
            margin: auto.
        }
        .stTextInput label {
            color: #983D30 !important.
        }
        h1 {
            color: #983D30 !important.
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="block-container">', unsafe_allow_html=True)
    st.image('https://raw.githubusercontent.com/TiphaineInData/recommandations_films/main/back_connexion.jpg', use_column_width=True)
    st.title("Connexion")
    st.text("")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:

        email = st.text_input("Adresse e-mail")
        password = st.text_input("Mot de passe", type="password")


        if st.button("Se connecter", key="btn_se_connecter"):
            nom = check_credentials(email, password)
            if nom:
                st.session_state.user = nom
                st.session_state.page = "main"
                st.experimental_rerun()
            else:
                st.error("Identifiants incorrects. Veuillez réessayer.")

    st.markdown("</div>", unsafe_allow_html=True)

# Gérer la navigation entre les pages
if 'page' not in st.session_state:
    st.session_state.page = "login"

if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "main":
    main_page()
elif st.session_state.page == "idees":
    idees_page()
