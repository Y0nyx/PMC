const path = require("path");
const { spawn } = require("child_process");
const { app, BrowserWindow, Menu } = require("electron");
isDevelopment = require.main.filename.indexOf("app.asar") === -1; // Use require.main instead of process.mainModule
global.appPath = isDevelopment ? process.cwd() : app.getAppPath() + "/../";

const { apiFrontend } = require("./apiFrontend");
const { apiAi } = require("./apiAi");
const { apiPLC } = require("./apiPLC");

let mainWindow;
let pythonProcess;

let configReact;
const mainMenuTemplate = [
  {
    label: "File",
    submenu: [
      {
        label: "Quit",
        click: () => app.quit(),
      },
    ],
  },
];

function createMainWindow() {
  //Create Menu
  const mainMenu = Menu.buildFromTemplate(mainMenuTemplate);
  // Menu.setApplicationMenu(mainMenu);
  //create window
  mainWindow = new BrowserWindow({
    title: "Dofa",
    kiosk: true,
    fullscreen: true,
    webPreferences: {
      sandbox: false,
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "./build/index.html"));
  mainWindow.maximize();
}

app.whenReady().then(() => {
  configReact = require(path.join(appPath, "configReact.js"));
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });

  plcPythonPath = path.join(appPath, "plc.py");
  pythonProcess = spawn("python3", [plcPythonPath]);
  pythonProcess.stdout.on("data", (data) => {
    console.log(`Python stdout: ${data}`);
    // You can send this data to the frontend using mainWindow.webContents.send if needed
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Python stderr: ${data}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`Python process exited with code ${code}`);
  });
  apiFrontend(mainWindow, configReact);
  apiAi(mainWindow);
  apiPLC(mainWindow);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    //if MacOs
    app.quit();
  }
});
