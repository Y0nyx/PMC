//Socket avec Python
const WebSocket = require('ws');

let pythonsocket = null;
function apiPLC(mainWindow)
{
  const wss = new WebSocket.Server({ port: 8003 });

  wss.on('connection', ws => {
      console.log('Client connected');

      ws.on('message', data => {
          let receivedData = JSON.parse(data.toString());
          
          switch (receivedData.code) {
              case 'ready':
                mainWindow.webContents.send("ready");
                console.log("ready received");
                break;
              case 'error':
                mainWindow.webContents.send("error",receivedData.data.message);
                console.log(receivedData);
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

function writeToPLC(message) {
  if (pythonsocket) {
      pythonsocket.send(JSON.stringify(message));
  }
}


module.exports = {apiPLC,writeToPLC};