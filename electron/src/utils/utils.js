const ipcRenderer = window.require("electron").ipcRenderer;

export async function pieceParser(piece) {
  let srcBase64;
  let base64Image = await ipcRenderer.invoke("readImage", piece.photo);
  let boundingBox = await ipcRenderer.invoke(
    "readBoundingBox",
    piece.boundingbox
  );

  if (base64Image) {
    srcBase64 = `data:image/jpeg;base64,${base64Image}`;
  } else {
    console.error("No image content received.");
  }

  let date = new Date(piece.date);

  let result = piece.resultat == "1" ? "succès" : "échec";

  let type = result ? piece.nom_erreur_soudure : "";

  let row = {
    id: piece.id,
    url: srcBase64,
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
