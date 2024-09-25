const path = require("path");
const { app, BrowserWindow, Menu } = require("electron");
isDevelopment = require.main.filename.indexOf("app.asar") === -1; // Use require.main instead of process.mainModule
global.appPath = isDevelopment
  ? process.cwd()
  : app.getAppPath()+"/../";

const {apiFrontend} = require("./apiFrontend");
const {apiAi} = require("./apiAi");



let mainWindow;

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
    kiosk: false,
    fullscreen: false,
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

  apiFrontend(mainWindow,configReact);
  apiAi(mainWindow);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    //if MacOs
    app.quit();
  }
});
