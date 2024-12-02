const ipcRenderer = window.require("electron").ipcRenderer;

export async function pieceParser(piece, loadImage) {
  let images = [];
  // Run both IPC calls in parallel

  if (loadImage) {
    const [base64Images, boundingBox] = await Promise.all([
      ipcRenderer.invoke("readImages", piece.photo),
      ipcRenderer.invoke("readBoundingBox", piece.boundingbox),
    ]);

    for (let image of base64Images) {
      if (image) {
        const imageFileNameWithoutExt = image.fileName
          .split(".")
          .slice(0, -1)
          .join(".");

        // Find the matching bounding box based on the filename (ignoring the extension)
        const matchedBoundingBox = boundingBox.find((box) => {
          const boxFileNameWithoutExt = box.fileName
            .split(".")
            .slice(0, -1)
            .join(".");
          return imageFileNameWithoutExt === boxFileNameWithoutExt;
        });

        images.push({
          url: `data:image/jpeg;base64,${image.base64Image}`,
          boundingBox: matchedBoundingBox,
        });
      } else {
        console.error("No image content received.");
      }
    }
  }

  // Process the date and result
  let date = new Date(piece.date);
  let result = piece.resultat === 1 ? "succès" : "échec";
  let type = result === "échec" ? piece.nom_erreur_soudure : "";

  // Construct the row object
  let row = {
    id: piece.id,
    images: images,
    img_folder: piece.photo,
    date: date.toISOString().split("T")[0],
    hour: `${(date.getHours() - 4) % 24}:${date
      .getMinutes()
      .toString()
      .padStart(2, "0")}`,
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

export async function piecesParser(array, rowsPerPage, page) {
  page = page + 1;
  let limit = rowsPerPage * page;

  console.log("limit", limit);
  console.log("rowsPerPage", rowsPerPage);
  console.log("page", page);
  // Map each item to a promise
  let compteur = 0;
  //console.log(array)
  const promises = array.map((item) => {
    if (compteur < limit) {
      compteur++;
      console.log(item);
      return pieceParser(item, true);
    } else {
      compteur++;
      return pieceParser(item, false);
    }
  });

  // Wait for all promises to resolve
  const rows = await Promise.all(promises);

  console.log("STOP");
  return rows;
}
