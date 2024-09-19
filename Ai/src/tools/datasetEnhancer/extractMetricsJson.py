import json
import pandas as pd

def extract_data(files):
    # Créer un objet ExcelWriter pour écrire dans un fichier Excel
    with pd.ExcelWriter('output.xlsx') as writer:
        for i, filename in enumerate(files):
            # Charger les données depuis le fichier JSON
            with open(filename, 'r') as file:
                data = json.load(file)

            # Créer un DataFrame pandas pour chaque fichier JSON
            df = pd.DataFrame()

            # Ajouter les données à DataFrame
            for entry in data:
                # Ajouter une nouvelle colonne avec le nom de la clé 'name' et les valeurs de 'y'
                df[entry['name']] = entry['y']

            # Écrire les données dans un worksheet avec un nom basé sur le nom du fichier
            df.to_excel(writer, sheet_name=f'Sheet_{i+1}', index=False)


if __name__ == "__main__":
    json_files = ['metrics.json', 'without_equ.json']

    extract_data(json_files)
