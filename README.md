# recommandations_films
Application de recommandations de films

Bienvenue sur notre application de recommandations de films !

Pour ce projet 2 à la wild code school, nous avons été appelés par un cinéma Creusois afin d'attirer plus de public au sein de son établissement.

Nous avons débuté par une étude de marché de la population en Creuse ainsi que l'état du cinéma en France (et en Creuse).

L'étude de marché fait apparaître que le nombre de cinémas en Creuse par rapport à la France est cohérent (1 siège pour 56 habitants en Creuse contre 1 pour 57 en France), mais que la fréquentation est largement moins importante en Creuse que dans le reste de la France.

Le cinéma nous a donc demandé de pouvoir lui recommander des films qui pourraient plaire à son public.
6 cinémas sur 7 en Creuse étant des cinémas art et essai, nous avons également intégré cette donnée dans notre application.

Nous avons utilisé des bases de données IMDB et TMDB, et nous avons webscrappé la base de données des films art et essai recommandés par l'association des films art et essai.

Après avoir exploré, nettoyé, filtré et traité les données nous avons pu obtenir un seul dataset orienté sur les films dont la langue est le français, l'anglais et toutes langues concernant les films art et essai (un film peut être recommandé en tant qu film art et essai pour protéger la diversité du cinéma, il était donc nécessaire de conserver également une diversité dans notre dataset).

Ensuite, place au machine learning !

Nous avons utilisé l'algorythme du plus proche voisin (KNN) pour trouver 4 films correspondant au film renseigné par l'utilisateur.
En plus des variables numériques, nous avons appliqué du TF-IDF après avoir lemmatizé le texte sur les descriptions, les genres, et les participants au film (acteur, réalisatrice ...). 

Nous avons pris l'initiative de rajouter une rubrique "recherche par nom". Qui ne s'est jamais dit que rarement un film avec Nicolas Cage est un navet ?

Enfin, la troisième page permet au Cinéma d'avoir des idées de films et d'effectuer une recherche par réalisateurs, par nom de film art et essai...
Elle permet également de se connecter sur la page du CNC où les derniers chiffres du cinéma en France apparaissent avec les films ayant le plus fonctionné.
ELle permet enfin de se connecter au site de l'association des films art et essai : ce site présente les derniers films recommandés art et essai et des informations sur le renouvellement du lable art et essai du cinéma (...).


