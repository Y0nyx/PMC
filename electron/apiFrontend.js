const { ipcMain } = require("electron");
const fs = require("fs");
const { generateUUID } = require("./utils");
const { writeToPython } = require("./apiAi");
const { exec } = require("child_process");

//API avec Frontend
const query = require("./queries/queries");

function apiFrontend(mainWindow, configReact) {
  ipcMain.on("fetchConfig", (event) => {
    mainWindow.webContents.send("ReceiveConfig", configReact);
  });

  ipcMain.handle("readImages", (event, folderPath) => {
    try {
      // Read all files in the folder
      const files = fs.readdirSync(folderPath);
  
      // Map through all files, read each image, and convert to base64
      const base64Images = files.map(file => {
        const filePath = path.join(folderPath, file);
        const imageBuffer = fs.readFileSync(filePath);
        const base64Image = Buffer.from(imageBuffer).toString("base64");
  
        return { fileName: file, base64Image };
      });
  
      return base64Images; // Return the list of base64 images
    } catch (error) {
      console.error("Error reading images from folder:", error);
      return null;
    }
  });

  ipcMain.handle("readBoundingBox", (event, folderPath) => {
    try {
      // Read all files in the folder
      const files = fs.readdirSync(folderPath);
  
      // Map through all files and read bounding boxes from each
      const boundingBoxes = files.map(file => {
        const filePath = path.join(folderPath, file);
        const data = fs.readFileSync(filePath, "utf8");
  
        // Split the file data by lines, each line is a bounding box
        const boxes = data.trim().split("\n").map(line => {
          const [classId, xCenter, yCenter, width, height, confidence] = line
            .trim()
            .split(" ")
            .map(Number);
  
          return {
            classId,
            xCenter,
            yCenter,
            width,
            height,
            confidence,
          };
        });
  
        return {
          fileName: file,
          boundingBoxes: boxes,
        };
      });
  
      return boundingBoxes; // Return the list of bounding boxes for each file
    } catch (error) {
      console.error("Error reading bounding boxes from folder:", error);
      return null;
    }
  });

  ipcMain.handle("createClient", async (event, client) => {
    let uuid = generateUUID();
    let _client = {
      id: uuid,
      nom: client.nom,
      telephone: client.telephone,
      email: client.email,
    };

    await query.createClient(_client);

    return _client;
  });
  ipcMain.handle("createPiece", async (event, piece) => {
    let uuid = generateUUID();
    let currentDate = new Date()
      .toISOString()
      .replace("T", " ")
      .replace(/\.\d+Z$/, "");

    console.log("piece", piece);
    let _piece = {
      id: uuid,
      date: currentDate,
      url: piece.url,
      boundingbox: piece.boundingbox,
      resultat: piece.resultat,
      id_client: piece.id_client,
      id_log: piece.id_log,
      id_type_piece: piece.id_type_piece,
      id_erreur_soudure: piece.id_erreur_soudure,
    };
    await query.createPiece(_piece);
    return _piece;
  });

  ipcMain.handle("createLog", async (event, log) => {
    let uuid = generateUUID();
    let _log = { id: uuid, id_client: log.id_client, nom: log.nom };

    await query.createLog(_log);

    return _log;
  });

  ipcMain.on("fetchPieces", async (event, id_client, id_log) => {
    let result = await query.fetchPieces(id_client, id_log);
    mainWindow.webContents.send("receivePieces", result);
  });

  ipcMain.on("fetchPiece", async (event, id) => {
    let result = await query.fetchPiece(id);
    console.log("for id", id, result);
    mainWindow.webContents.send("receivePiece", result);
  });

  ipcMain.on("fetchClients", async (event) => {
    let result = await query.fetchClients();
    mainWindow.webContents.send("receiveClients", result);
  });

  ipcMain.on("fetchLogs", async (event, client_id) => {
    let result = await query.fetchLogs(client_id);
    mainWindow.webContents.send("receivelogs", result);
  });

  ipcMain.on("fetchTypesPiece", async (event) => {
    let result = await query.fetchTypesPiece();
    mainWindow.webContents.send("receiveTypesPiece", result);
  });

  ipcMain.on("deletePiece", async (event, selected) => {
    await query.deletePiece(selected);
  });

  ipcMain.on("restart", async (event, id) => {
    await query.deletePiece(id);
    writeToPython({ code: "start", data: "" });
  });

  ipcMain.on("powerOffMachine", async () => {
    exec("sudo poweroff", (error, stdout, stderr) => {
      if (error || stderr) {
        console.log(error);
        return;
      }
    });
  });

  ipcMain.on("rebootMachine", async () => {
    exec("sudo reboot", (error, stdout, stderr) => {
      if (error || stderr) {
        console.log(error);
        return;
      }
    });
  });

  ipcMain.on("resetData", async () => {
    let result = await query.resetData();
  });

  ipcMain.on("resetAll", async () => {
    exec(
      "sudo docker stop $(sudo docker ps -q) && sudo docker rm $(sudo docker ps -aq) && sudo docker volume rm $(sudo docker volume ls -q) && sudo docker-compose -f ../docker/docker-compose.yml up -d",
      (error, stdout, stderr) => {
        if (error || stderr) {
          console.log(error);
          return;
        }
      }
    );
  });

  ipcMain.on("command", (event, req) => {
    console.log(`${req.code} command send`);
    writeToPython(req);
  });
}

module.exports = {
  apiFrontend,
};
