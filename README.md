# Traitement d'Image Optimisé par GPU : Panorama

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

# Exemple de sortie console :

Temps d'exécution sans CUDA: 32.45 secondes

Mon Jan  8 08:57:57 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 546.01                 Driver Version: 546.01       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2070 ...  WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   75C    P8              11W /  80W |    741MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      6812    C+G   ...oogle\Chrome\Application\chrome.exe    N/A      |
|    0   N/A  N/A      7592      C   ...brah\anaconda3\envs\cuda\python.exe    N/A      |
|    0   N/A  N/A     10616    C+G   C:\Windows\explorer.exe                   N/A      |
|    0   N/A  N/A     10884    C+G   ...nt.CBS_cw5n1h2txyewy\SearchHost.exe    N/A      |
|    0   N/A  N/A     17532    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe    N/A      |
|    0   N/A  N/A     17828    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe    N/A      |
|    0   N/A  N/A     20432    C+G   ...inaries\Win64\EpicGamesLauncher.exe    N/A      |
|    0   N/A  N/A     20664    C+G   ...ne\Binaries\Win64\EpicWebHelper.exe    N/A      |
|    0   N/A  N/A     21580    C+G   ...ta\Local\Programs\Notion\Notion.exe    N/A      |
|    0   N/A  N/A     22392    C+G   ...crosoft\Edge\Application\msedge.exe    N/A      |
|    0   N/A  N/A     23400    C+G   ...5n1h2txyewy\ShellExperienceHost.exe    N/A      |
3d8bbwe\WindowsTerminal.exe    N/A      |
|    0   N/A  N/A     49864    C+G   ...Programs\Microsoft VS Code\Code.exe    N/A      |
+---------------------------------------------------------------------------------------+
