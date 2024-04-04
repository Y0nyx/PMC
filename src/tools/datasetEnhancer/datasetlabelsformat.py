import os

def filter_labels(input_dir, output_dir):
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Liste des sous-dossiers dans le dossier d'entrée
    subdirectories = os.listdir(input_dir)
    
    # Parcourir chaque sous-dossier dans le dossier d'entrée
    for subdirectory in subdirectories:
        # Chemin d'accès complet du sous-dossier
        subdirectory_path = os.path.join(input_dir, subdirectory)
        
        # Vérifier si c'est un dossier
        if os.path.isdir(subdirectory_path):
            # Chemin d'accès au dossier labels dans le sous-dossier actuel
            labels_dir = os.path.join(subdirectory_path, "labels")
            
            # Chemin de sortie pour le dossier labels filtré
            output_subdirectory = os.path.join(output_dir, subdirectory, "labels")
            if not os.path.exists(output_subdirectory):
                os.makedirs(output_subdirectory)
            
            # Filtrer les fichiers dans le dossier labels
            filter_labels_in_directory(labels_dir, output_subdirectory)

def filter_labels_in_directory(input_dir, output_dir):
    # Liste des fichiers dans le dossier d'entrée
    files = os.listdir(input_dir)
    
    # Parcourir chaque fichier dans le dossier d'entrée
    for file in files:
        # Chemin d'accès complet du fichier d'entrée
        input_path = os.path.join(input_dir, file)
        
        # Vérifier si c'est un fichier .txt
        if os.path.isfile(input_path) and file.lower().endswith('.txt'):
            # Chemin de sortie pour le fichier filtré
            output_path = os.path.join(output_dir, file)
            
            # Ouvrir le fichier d'entrée en lecture
            with open(input_path, 'r') as f_in:
                # Lire les lignes du fichier
                lines = f_in.readlines()
                
                # Filtrer les lignes commençant par "0"
                filtered_lines = [line for line in lines if line.startswith("0")]
            
            # Écrire les lignes filtrées dans le fichier de sortie
            with open(output_path, 'w') as f_out:
                f_out.writelines(filtered_lines)
                print(f"Filtrage terminé pour {file}")

# Dossier d'entrée et de sortie
dataset_path = 'D:\dataset\WeldingDefectDetection.v2i.yolov8'
output_path = 'D:\dataset\output_2'

# Filtrer les fichiers dans les sous-dossiers train, valid et test
filter_labels(dataset_path, output_path)
