const ipcRenderer = window.require("electron").ipcRenderer;

export async function pieceParser(piece) {
  let images = [];
  let base64Images = await ipcRenderer.invoke("readImages", piece.photo);
  let boundingBox = await ipcRenderer.invoke(
    "readBoundingBox",
    piece.boundingbox
  );


  for (image of base64Images){
    if (image) {
      const imageFileNameWithoutExt = image.fileName.split('.').slice(0, -1).join('.');

    // Find the matching bounding box based on the filename (ignoring the extension)
    const matchedBoundingBox = boundingBoxes.find(box => {
      const boxFileNameWithoutExt = box.fileName.split('.').slice(0, -1).join('.');
      return imageFileNameWithoutExt === boxFileNameWithoutExt;
    });

      images.push({url:`data:image/jpeg;base64,${image}` , boundingBox: matchedBoundingBox});
    } else {
      console.error("No image content received.");
    }
  }

  let date = new Date(piece.date);

  let result = piece.resultat == "1" ? "succès" : "échec";

  let type = result ? piece.nom_erreur_soudure : "";

  let row = {
    id: piece.id,
    images: images,
    box: boundingBox,
    date: date.toISOString().split("T")[0],
    hour: date.getHours() + ":" + date.getMinutes(),
    result: result,
    errorType: type,
    errorDescription: piece.description_erreur_soudure,
    nom_client: piece.nom_client,
    nom_type_piece: piece.nom_type_piece,
    description_type_piece: piece.description_type_piece,
    id_log: piece.id_log,
  };

  console.log()
  return row;
}

export function idSubstring(id) {
  return id.substring(0, 8) + "..." + id.substring(id.length - 4);
}

export async function piecesParser(array) {
  let rows = [];

  for (let i = 0; i < array.length; i++) {
    let row = await pieceParser(array[i]);
    rows.push(row);
  }
  return rows;
}
