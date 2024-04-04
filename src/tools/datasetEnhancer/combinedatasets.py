import os
import shutil

def combine_datasets(dataset_paths, output_path):
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Parcourir chaque dataset
    for i, dataset_path in enumerate(dataset_paths, start=1):
        contents = os.listdir(dataset_path)
        directories = [item for item in contents if os.path.isdir(os.path.join(dataset_path, item))]
        for file in directories:
            # Chemin d'accès au dossier images du dataset actuel
            images_dir = os.path.join(dataset_path, file, "images")
            labels_dir = os.path.join(dataset_path, file, "labels")

            output_images = os.path.join(output_path, file, "images")
            output_labels = os.path.join(output_path, file, "labels")
            if not os.path.exists(output_images):
                os.makedirs(output_images)
            if not os.path.exists(output_labels):
                os.makedirs(output_labels)

            # Copier les fichiers images dans le dossier de sortie
            copy_files(images_dir, output_images)
            copy_files(labels_dir, output_labels)

            print(f"Images du fichier {file} dataset {i} copiées avec succès.")

def copy_files(source_dir, destination_dir):
    # Liste des fichiers dans le dossier source
    files = os.listdir(source_dir)

    # Parcourir chaque fichier dans le dossier source
    for file in files:
        # Chemin d'accès complet du fichier source
        source_path = os.path.join(source_dir, file)
        # Chemin d'accès complet du fichier de destination
        destination_path = os.path.join(destination_dir, file)

        # Copier le fichier source dans le dossier de destination
        shutil.copy2(source_path, destination_path)

# Fonction principale pour combiner les datasets
def main():

    dataset_paths = ["D:\dataset\output_2",
                     "D:\dataset\dataset_with_equalize"]
    output_path = "D:\dataset\dofa_syntethic_welding_v1"

    # Combinaison des datasets
    if dataset_paths and output_path:
        combine_datasets(dataset_paths, output_path)
        print("Combinaison des datasets terminée avec succès.")
    else:
        print("Aucun dataset spécifié ou aucun chemin d'accès au dossier de sortie spécifié.")

if __name__ == "__main__":
    main()
