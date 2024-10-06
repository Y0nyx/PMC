//Socket avec Python
const WebSocket = require('ws');

let pythonsocket = null;
function apiAi(mainWindow)
{
  const wss = new WebSocket.Server({ port: 8002 });

  wss.on('connection', ws => {
      console.log('Client connected');

      ws.on('message', data => {
          let receivedData = JSON.parse(data.toString());
          
          switch (receivedData.code) {
              case 'start':
                mainWindow.webContents.send("start");
                console.log("start received");
                  break;
              case 'stop':
                mainWindow.webContents.send("stop");
                console.log("stop received");
                  break;
              case 'train':
                mainWindow.webContents.send("train", receivedData.data);
                console.log("train received");
                  break;
              case 'init':
                mainWindow.webContents.send("init");
                console.log("init received");
                  break;
              case 'error':
                mainWindow.webContents.send("error",receivedData.data.message);
                console.log(`error ${receivedData.data.message}`);
                  break;
              case 'resultat':
                mainWindow.webContents.send("resultat",receivedData.data);
                console.log("resultat received",receivedData)
                  break;
              default:
                console.log("Message inconnu received: ", receivedData);
                  break;
          }

          receivedData = undefined
      });

      ws.on('close', () => {
          console.log('Client disconnected');
      });

      // Store the WebSocket connection
      pythonsocket = ws;
  });
}

function writeToPython(message) {
  if (pythonsocket) {
      pythonsocket.send(JSON.stringify(message));
  }
}


module.exports = {apiAi,writeToPython};