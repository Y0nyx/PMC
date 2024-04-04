import os
import cv2

def apply_processing(input_dir, output_dir):
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
            # Liste des fichiers dans le sous-dossier
            files = os.listdir(subdirectory_path)
            
            # Parcourir chaque fichier dans le sous-dossier
            for file in files:
                # Chemin d'accès complet du fichier d'entrée
                input_path = os.path.join(subdirectory_path, file)
                
                # Vérifier si c'est un fichier image
                if os.path.isfile(input_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Lire l'image
                    image = cv2.imread(input_path)
                    
                    # Appliquer les traitements
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    image = cv2.equalizeHist(image)
                    
                    # Chemin de sortie pour l'image traitée
                    output_path = os.path.join(output_dir, subdirectory, file)
                    
                    # Créer le dossier de sortie s'il n'existe pas
                    output_subdirectory = os.path.join(output_dir, subdirectory)
                    if not os.path.exists(output_subdirectory):
                        os.makedirs(output_subdirectory)
                    
                    # Enregistrer l'image traitée
                    cv2.imwrite(output_path, image)
                    print(f"Traitement terminé pour {file}")

# Fonction principale pour traiter le dataset
def process_dataset(dataset_path, output_path):
    # Vérifier si le chemin d'accès au dataset est valide
    if not os.path.exists(dataset_path):
        print("Le chemin d'accès spécifié au dataset est invalide.")
        return
    
    # Traiter les images dans les dossiers train, valid et test
    apply_processing(os.path.join(dataset_path, "train"), os.path.join(output_path, "train"))
    apply_processing(os.path.join(dataset_path, "valid"), os.path.join(output_path, "valid"))
    apply_processing(os.path.join(dataset_path, "test"), os.path.join(output_path, "test"))

# Spécifier le chemin d'accès au dataset et au dossier de sortie
dataset_path = 'D:\dataset\old_dataset'
output_path = 'D:\dataset\old_dataset_2'

# Traiter le dataset
process_dataset(dataset_path, output_path)
