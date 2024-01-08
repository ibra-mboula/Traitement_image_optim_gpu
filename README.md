# Video (7s Max) >>> Panorama

## Aperçu
Ce projet à pour but de transformer une vidéo en panorama. Il est conçu pour travailler avec des vidéos de courte durée (7 secondes maximum) et utilise OpenCV compilé avec le support CUDA pour améliorer les performances de traitement d'image.

## Prérequis
- **OpenCV avec Support CUDA** : Installez OpenCV compilé avec le support CUDA pour utiliser le GPU
- **Python 3.x** : Installer Python 3.x 
- **Bibliothèques Python** : Numpy

## Scripts Principaux
- **app_gpu.py** : Utilise CUDA pour accélérer la detectection des keypoints et le calcul des descriptors. Il comprend des fonctions pour l'extraction de frames, la détection et l'association de caractéristiques, et la création de panoramas.
- **app_cpu.py** : Version alternative du script pour le traitement sans accélération GPU, utile pour comparer les performances.

## Utilisation
1. Placez votre vidéo (7 secondes maximum) dans le dossier `videos` à la racine du projet.
2. Exécutez `app_gpu.py` pour un traitement avec accélération GPU ou `app_cpu.py` pour un traitement sans GPU.
3. Le résultat sera sauvegardé dans le dossier `output` que vous devez aussi créer à la racine du projet.

## Profilage et Monitoring
Les scripts incluent un profilage de performance et affichent le temps d'exécution ainsi que les statistiques d'utilisation GPU.

# Exemple de sortie console sans CUDA  :

![cn](https://github.com/ibra-mboula/Traitement_image_optim_gpu/assets/78673312/cb5e126d-5785-4eb1-a1d7-32da266bfb89)


