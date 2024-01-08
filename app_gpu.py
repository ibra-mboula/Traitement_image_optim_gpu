import cv2
import numpy as np
import os
import time
import cProfile
import subprocess
import cv2.cuda

start_time = time.time()


#SECTION - Fonction pour découper la vidéo en frames
def extract_frames(video_path, num_frames=100):
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Je stocke tous les frames pour commencer
    skip_frames = total_frames // num_frames # pouuur avoir le nombre de frames à sauter

    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames * i) #sauter les frames et recup ce que je veux
        ret, frame = cap.read() #stocker les frames dans frame, et ret return T/F
        if ret:
            frames.append(frame)

    cap.release() #liberer la memoire
    return frames


# Chemin de la vidéo et dossier de sortie
video_path = './videos/video2.mp4'

#ANCHOR - Extraire les frames
frames = extract_frames(video_path)


#LINK - https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html

orb = cv2.cuda_ORB.create(nfeatures=2000) # Initialise ORB detector version GPU

#NOTE - On cree un objet bfmatch, avec ca on va trouver les corespondance en comparant les descripteurs 
bf = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING) #Brute-Force Matcher



#SECTION - Modifie la perspective de l'img1 pour corresondre a img2 mais a besoin de la matrice de l'homographie
def warpImages(img1, img2, H):

  shape1 = img1.shape #(1920, 1080, 3) dim de l'image
  
  rows1 = shape1[0] #1920

  cols1 = shape1[1] #1080

  shape2 = img2.shape
  rows2 = shape2[0]
  cols2 = shape2[1]

   #NOTE - On va creer une matrice de float32 avec les 4 coins de l'image 1
   # sup gauche,inf gauche,inf droite, sup droite  
  corners1 = [[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]
  list_of_points_1 = np.float32(corners1).reshape(-1, 1, 2)

  corners2 = [[0,0], [0,rows2], [cols2,rows2], [cols2,0]]
  temp_points = np.float32(corners2).reshape(-1,1,2)

  # 
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  #  min coord
  min_coords = list_of_points.min(axis=0).ravel() - 0.3 #j'aplati le tableau et le 0.5 pour depplacer les coord
  x_min, y_min = np.int32(min_coords)

  # max coord
  max_coords = list_of_points.max(axis=0).ravel() + 0.3
  x_max, y_max = np.int32(max_coords)
  
  translation_dist = [-x_min,-y_min] # pour etre sure que l'image est deplacé dans le cadran avec les coord positif
  
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # j'appllique la perspect pour transfom img2
  transformed_img2 = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_img = np.copy(transformed_img2)

    # Copy img1 dans output image
  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1 #LINK - https://docs.scipy.org/doc/numpy/reference/generated/numpy.copy.html

  return output_img




#SECTION - Fonction pour superposer les images
def stitchImages(img1, img2):
    min_homo = 10 # je fixe le nbr minim de correspond pour calculer l'homographie
    
    #on va convertir les img en nive de gris
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Upload vers GPU
    img1_gpu = cv2.cuda_GpuMat()
    img1_gpu.upload(img1_gray)
    
    img2_gpu = cv2.cuda_GpuMat()
    img2_gpu.upload(img2_gray)

    # version gpu de detectAndCompute 
    #NOTE - Grace à orb.detectAndCompute on va trouver les keypoints et les descripteurs des deux imag
    keypoints1, descriptors1 = orb.detectAndComputeAsync(img1_gpu, None)# none car il n'y a pas de masque, donc tous les points sont pris
    keypoints2, descriptors2 = orb.detectAndComputeAsync(img2_gpu, None)

    # on le Download vers CPU
    keypoints1 = orb.convert(keypoints1)
    keypoints2 = orb.convert(keypoints2)

    #NOTE - On va trouver les correspondances entre les descripteurs des deux images
    # notre knnMatch renvoie un objet DMatch qui contient les correspondances entre les descripteurs des deux images
    #NOTE - GPU version de knnMatch, il envoie direct les objetect DMatch en CPU
    matches = bf.knnMatch(descriptors1, descriptors2, k=2) # il est mis a 2 car on veut les deux meilleurs correspondances

    good = []
    
    #NOTE - Chaque element de matches est une liste de deux objet ,le  best match 1 et 2
    
    #NOTE - On va filtrer les correspondances en utilisant le ratio de Lowe
    #Donc si la distance du match1 est inf au ratio * la dist du match2, on deduit que c'est un bon match
    for i, j in matches:
        if i.distance < 0.6 * j.distance: #LINK - https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
            good.append(i)

    
    
    #NOTE - On va calculer l'homographie si on a plus de 10 correspondances
    # c'est une sorte de transfo pour que 2 img se ressemble si il sont pris de #points de vu, translation, rot..
    #donc ici on va alligner 2img en fonction de la corresp de leurs keypoints
    if len(good) > min_homo:
        
        src_pts_list = []
        for m in good:
            keypoint = keypoints1[m.queryIdx] #keypoint :  < cv2.KeyPoint 0000022435A77480 >
            point = keypoint.pt #point :  (910.0, 932.0) (x,y) du keypoint
            src_pts_list.append(point)
            
        #je met les points dans une matrice de float32
        src_pts_array = np.array(src_pts_list, dtype=np.float32)
        # je redimensi en 1x2 la matrice pour avoir un tab compatible avec la fct findHomography
        # -1 que numpy calcule automat le nbr de ligne
        src_pts = src_pts_array.reshape(-1, 1, 2)#LINK - https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

        dst_pts_list = []
        for m in good:
            keypoint = keypoints2[m.trainIdx]
            point = keypoint.pt
            dst_pts_list.append(point)
            
        dst_pts_array = np.array(dst_pts_list, dtype=np.float32)
        dst_pts = dst_pts_array.reshape(-1, 1, 2)
        #NOTE - On va calculer l'homographie a présent, elle renvoie 2 valeurs, la matrice de l'homographie et le mask (points ignorés)
        # RANSAC est un algo qui va rservir a trouver l'homographie et 5.0 est la distance seuil de proxmt pour considerer un pts ignoré
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #LINK - python findHomography https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
        result = warpImages(img2, img1, M)
        return result
    else:
        return None
    
#on charge la premiere image
img1 = frames[0]

#SECTION - Mon main pour lancer le process de création du panorama
def process_images():
    global img1
    for i in range(1, len(frames)):  # lit toutes les images
        img2 = frames[i]
        result = stitchImages(img1, img2) # superposer les imgs 1 temp panorama et l'imge 2 la frame

        if result is not None: 
            img1 = result 


    cv2.imwrite(f"./output/panorama_final.jpg", img1)

# Profiling
profiler = cProfile.Profile()
profiler.enable()
process_images()
profiler.disable()
profiler.print_stats(sort='time')


end_time = time.time()
print("Temps d'exécution AVEC CUDA: {:.2f} secondes".format(end_time - start_time))

# Monitor CPU and GPU usage
gpu_stats = subprocess.check_output("nvidia-smi", shell=True)
print(gpu_stats.decode("utf-8"))