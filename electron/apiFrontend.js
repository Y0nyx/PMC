const { ipcMain } = require("electron");
const fs = require("fs");
const { generateUUID } = require("./utils");
const { writeToPython } = require("./apiAi");
const {writeToPLC}   = require("./apiPLC")
const { exec } = require("child_process");
const {parse} = require('json2csv')
const path = require("path");
const diskusage = require('diskusage');

let appPath = global.appPath;
let sqlPath;
let sqlDefaultValuesPath;
let dockerComposePath;
const isDev = process.defaultApp;

if (!isDev) {
  sqlPath = path.join(appPath, "generateDatabase.sql");
  sqlDefaultValuesPath = path.join(appPath, "defaultValues.sql");
  dockerComposePath = path.join(appPath, "docker-compose.yml");
} else {
  sqlPath = "../sql/generateDatabase.sql";
  sqlDefaultValuesPath = "../sql/defaultValues.sql";
  dockerComposePath = "../docker-composeDev.yml";
}

//API avec Frontend
const query = require("./queries/queries");
const { type } = require("os");




function apiFrontend(mainWindow, configReact) {



  ipcMain.on("fetchConfig", (event) => {
    mainWindow.webContents.send("ReceiveConfig", configReact);
  });

  ipcMain.handle("readImages", (event, folderPath) => {
    try {
      // Read all files in the folder
      const files = fs.readdirSync(folderPath);

      // Map through all files, read each image, and convert to base64
      const base64Images = files.map((file) => {
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


  ipcMain.on("diskspace",()=> {
    const drivePath=path.parse(__dirname).root
    diskusage.check(drivePath, (err, info) => {
      if (err) {
          console.error("Error checking disk space:", err);
          mainWindow.webContents.send("error", `Error checking disk space: ${err} `);
      } else {

          const availablePercentage = (info.available / info.total) * 100;
          console.log("Available disk size:",availablePercentage)

          if(availablePercentage < 5){
            mainWindow.webContents.send("error", `le disque dur est presque plein, Veuillez dans les options pour réinitialiser les données`)
          }
   
      }
  });
  })


  ipcMain.handle("readBoundingBox", (event, folderPath) => {
    try {
      // Read all files in the folder
      const files = fs.readdirSync(folderPath);

      // Map through all files and read bounding boxes from each
      const boundingBoxes = files.map((file) => {
        const filePath = path.join(folderPath, file);
        const data = fs.readFileSync(filePath, "utf8");

        // Split the file data by lines, each line is a bounding box
        const boxes = data
          .trim()
          .split("\n")
          .map((line) => {
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
          box: boxes,
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


// Function to find the USB drive path dynamically
function findUsbDrivePath() {
  const mediaDir = '/media/dofa';
  const usbDrives = fs.readdirSync(mediaDir); // Read directories in /media

  for (const drive of usbDrives) {
    const drivePath = path.join(mediaDir, drive);
    try {
      const stats = fs.statSync(drivePath);

      // Check if it's a directory
      if (stats.isDirectory()) {
        // Check if it's writable
        fs.accessSync(drivePath, fs.constants.W_OK);
        console.log(drivePath, "found");
        return drivePath; // Return the first found USB drive
      }
    } catch (err) {
      // Handle the error (like logging it)
      console.error(`Error accessing ${drivePath}: ${err.message}`);
    }
  }

  return 0
}

  ipcMain.on("exportData", async (event) => {
    let client = await query.exportClient();
    let log = await query.exportLog()
    let piece = await query.exportPiece();

    
    let usb = findUsbDrivePath()
    if(usb == 0) {
      mainWindow.webContents.send("error","No USB Found");
      console.log("No USB found")
      return
    }
    

    // Get the current date
    const currentDate = new Date();
    const formattedDate = currentDate.toISOString().slice(0, 10); // Format: YYYY-MM-DD
    // Create a file name with the current date
    const outputDir = `${formattedDate}`;
    // Full path to the output file on the USB drive
    const outputFilePath = path.join(usb, outputDir);

    if(!fs.existsSync(outputFilePath)){
      fs.mkdirSync(outputFilePath)
    }
    
    const outputClient = path.join(outputFilePath, `client_${formattedDate}.csv`);
    const outputLog = path.join(outputFilePath, `log_${formattedDate}.csv`);
    const outputPiece = path.join(outputFilePath, `piece_${formattedDate}.csv`);

    const csvClient = parse(client)
    const csvLog = parse(log)
    let csvPiece
    if(piece.lentgh > 0){
      csvPiece = parse(piece)
    }
    else{
      
      csvPiece =  [
      'ID', 
      'Date', 
      'Photo', 
      'BoundingBox', 
      'Resultat', 
      'ID_client', 
      'ID_log', 
      'ID_type_piece', 
      'ID_erreur_Soudure'
    ].map(item => `"${item}"`).join(", ");
    }

    try{
      fs.writeFileSync(outputClient,csvClient)
      fs.writeFileSync(outputLog,csvLog)
      fs.writeFileSync(outputPiece,csvPiece)
      console.log("CSV generated ! ", outputFilePath)
    } catch(err) {
      console.log(err.message)
      mainWindow.webContents.send("error",err.message)
    }
   

    mainWindow.webContents.send("exportFinish");
  });



  ipcMain.on("deletePiece", async (event, selected) => {
    await query.deletePiece(selected);
  });

  ipcMain.on("restart", async (event, id) => {
    await query.deletePiece([id]);
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

  async function deleteAllimages() {
    try {
      let directories = await query.getAllimages(); // Assuming this returns an array of directory paths

      for (const dir of directories) {
        await fs.rmSync(dir.photo, { recursive: true, force: true });
        await fs.rmSync(dir.boundingbox, { recursive: true, force: true });
      }
      await deleteAllimages();
      console.log("All directories deleted successfully.");
    } catch (error) {
      console.error("Error deleting directories:", error);
    }
  }

  ipcMain.on("resetData", async () => {
    await deleteAllimages();
    await query.resetData();
    let connection = false;
        let sql = fs.readFileSync(sqlDefaultValuesPath, "utf8");

        console.log("REGENERATING DATABASE");
        while (!connection) {
          try {
            await query.generateDatabase(sql);
            connection = true;
            console.log("DATABASE CREATED ! ");
          } catch {
            console.log("ATTEMPT TO CONNECT...");
            await new Promise((resolve) => setTimeout(resolve, 5000));
          }
        }
  });

  ipcMain.on("resetAll", async () => {
    await deleteAllimages();

    exec(
      `sudo docker stop $(sudo docker ps -q) && sudo docker rm $(sudo docker ps -aq) && sudo docker volume rm $(sudo docker volume ls -q) && sudo docker-compose -f ${dockerComposePath} up -d`,
      async (error, stdout, stderr) => {
        console.log("EXEC");
        if (error || stderr) {
          console.log(error);
          console.log(stderr);
        }
        console.log(stdout);
        let connection = false;
        let sql = fs.readFileSync(sqlPath, "utf8");

        console.log("REGENERATING DATABASE");
        while (!connection) {
          try {
            await query.generateDatabase(sql);
            connection = true;
            console.log("DATABASE CREATED ! ");
          } catch {
            console.log("ATTEMPT TO CONNECT...");
            await new Promise((resolve) => setTimeout(resolve, 5000));
          }
        }

        console.log("REBOOT...");
        if (!isDev) exec("sudo reboot");
      }
    );
  });


  ipcMain.on("forward",() => {
    writeToPLC({ code: "forward", data: "" });
  })

  ipcMain.on("backward",() => {
    writeToPLC({ code: "backward", data: "" });
  })


  ipcMain.on("model",(event, req) =>{
    writeToPython({code:"model",data:req.model})
  })


  ipcMain.on("command", (event, req) => {
    console.log(`${req.code} command send`);

    if(req.code == "start")
      {
        writeToPython(req);
      }
    if(req.code == "stop")
      {
        writeToPython(req);
      }
    
  });

}

module.exports = {
  apiFrontend,
};
