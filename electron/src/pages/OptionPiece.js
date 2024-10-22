import React, { useContext, useEffect,useState } from "react";
import UIStateContext from "../Context/context";
import BackButton from "../components/BackButton";
import { useNavigate } from "react-router-dom";
import CreerClientButton from "../components/CreerClientButton";
import CreerLogButton from "../components/CreerLogButton";
import Box from "@mui/material/Box";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";
import CancelIcon from "@mui/icons-material/Cancel";
import { IconButton, Dialog, DialogContent, DialogTitle } from "@mui/material";

export default function OptionPiece() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const uicontext = useContext(UIStateContext);
  const [openError,setOpenError] = useState(false)
  const [errorMessage, setErrorMessage] = useState();
  const navigate = useNavigate();

  const [listClient, setListClient] = React.useState([]);
  const [listLogs, setListLogs] = React.useState([]);
  const [listTypesPiece, setTypesPiece] = React.useState([]);

  const [error, setError] = React.useState();


  function fetchClient() {
    ipcRenderer.send("fetchClients");
  }

  function fetchTypePiece() {
    ipcRenderer.send("fetchTypesPiece");
  }

  function fetchLogs(client_id) {
    ipcRenderer.send("fetchLogs", client_id);
  }



  useEffect(() => {
    ipcRenderer.on("error", (event,error) => {
      console.log(error)
      setErrorMessage(error);
      setOpenError(true)
    });

    ipcRenderer.on("receiveClients", async (event, _client) => {
      setListClient(_client);
    });
  
    ipcRenderer.on("receivelogs", async (event, _logs) => {
      setListLogs(_logs);
    });
  
    ipcRenderer.on("receiveTypesPiece", async (event, _typesPiece) => {
      setTypesPiece(_typesPiece);
    });
  

    fetchClient();
    fetchLogs(uicontext.state_client.id);
    fetchTypePiece();


    return () => {
      ipcRenderer.removeAllListeners('receiveClients');
      ipcRenderer.removeAllListeners('receivelogs');
      ipcRenderer.removeAllListeners('receiveTypesPiece');
      ipcRenderer.removeAllListeners("error");
    };
  }, []);
  function back() {
    if (uicontext.state_client.id && uicontext.state_log) {
      setError("");
      navigate("/");
    } else {
      setError("Veuillez choisir un client et un log pour continuer.");
    }
  }

  function handleClientChange(event) {
    uicontext.setClient(event.target.value);
    fetchLogs(event.target.value.id);
    uicontext.setLog("");
  }

  function handleLogChange(event) {
    uicontext.setLog(event.target.value);
  }

  function handleTypePiece(event) {
    uicontext.setTypePiece(event.target.value);
  }

  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="box-border flex flex-col w-full h-full justify-center items-center bg-gray-200 p-5">
        <div className="flex items-center w-5/6 border-gray-300 bg-gray-100 rounded-lg p-5 my-4">
          <div className="w-1/3"></div>
          <div className="w-1/3 text-4xl text-gray-800 uppercase font-normal flex justify-center items-center">
            {"Modifier les options en cours: "}
          </div>
          <BackButton back={back}></BackButton>
        </div>

        <div className=" shadow-xl rounded-lg flex flex-col justify-center items-center p-5 w-5/6 h-full  border-gray-300 bg-gray-100">
          <div className="flex w-1/2 flex-col justify-center items-center mx-20 p-5">
            {error && (
              <div className="text-red-600 font-normal font-bold text-xl">
                {"ERROR : " + error}
              </div>
            )}
            <div className="flex justify-between items-center w-full bg-white rounded-lg p-2 text-2xl text-gray-800  font-normal m-1">
              <span>{"Client:"}</span>

              <Box className="w-72 my-1 rounded-3 border-gray-400">
                <FormControl fullWidth>
                  <InputLabel className="text-2xl" id="Client">Client</InputLabel>
                  <Select
                    id="Client"
                    className="w-full h-20 text-2xl"
                    value={
                      (listClient &&
                        listClient.find(
                          (c) => c.id === uicontext.state_client.id
                        )) ||
                      ""
                    }
                    label="Client"
                    onChange={handleClientChange}
                  >
                    {listClient &&
                      listClient.map((client) => {
                        return <MenuItem value={client}>{client.nom}</MenuItem>;
                      })}
                  </Select>
                </FormControl>
              </Box>
            </div>
            <div className=" flex justify-between items-center w-full bg-white rounded-lg p-2 text-2xl text-gray-800  font-normal m-1">
              <span>{"#Log:"}</span>
              <Box className="w-72 my-2 rounded-3 border-gray-400">
                <FormControl fullWidth>
                  <InputLabel className="text-2xl" id="Log">#Log</InputLabel>
                  <Select
                    id="Log"
                    className="w-full h-20 text-2xl"
                    value={
                      (listLogs &&
                        listLogs.find(
                          (c) => c.id === uicontext.state_log.id
                        )) ||
                      ""
                    }
                    label="#Log"
                    onChange={handleLogChange}
                  >
                    {listLogs &&
                      listLogs.map((log) => {
                        return <MenuItem value={log}>{log.nom}</MenuItem>;
                      })}
                  </Select>
                </FormControl>
              </Box>
            </div>
            <div className="flex justify-between items-center w-full bg-white m-1 rounded-lg p-2 text-2xl text-gray-800  font-normal">
              <span>{"Type de Pièce: "}</span>
              <Box className="w-72 rounded-3 border-gray-400">
                <FormControl fullWidth>
                  <InputLabel className="text-2xl" id="Type de Pièce">Type de Pièce</InputLabel>
                  <Select
                    id="Type de Pièce"
                    className="w-full h-20 text-2xl"
                    value={
                      (listTypesPiece &&
                        listTypesPiece.find(
                          (c) => c.id === uicontext.state_type_piece.id
                        )) ||
                      ""
                    }
                    label="#Type de Pièce"
                    onChange={handleTypePiece}
                  >
                    {listTypesPiece &&
                      listTypesPiece.map((tyoePiece) => {
                        return (
                          <MenuItem value={tyoePiece}>{tyoePiece.nom}</MenuItem>
                        );
                      })}
                  </Select>
                </FormControl>
              </Box>
            </div>
          </div>
          <div className="flex justify-center items-center my-4">
            <CreerClientButton></CreerClientButton>
            <CreerLogButton></CreerLogButton>
          </div>
        </div>
      </div>
      <Dialog
        open={openError}
        aria-describedby="alert-dialog-slide-description"
      >
        <DialogTitle className="font-normal font-bold text-lg text-red-500 ">
          ERREUR
          <IconButton
            aria-label="close"
            onClick={() => setOpenError(false)}
            sx={{
              position: "absolute",
              right: 8,
              top: 8,
            }}
          >
            <div className="flex justify-center items-center w-full">
              <CancelIcon className="text-red-500 text-6xl hover:scale-105" />
            </div>
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <div className="flex flex-col p-2 justify-center items-center text-gray-400 font-normal font-bold ">
            <p className="text-justify  font-Cairo leading-normal text-3xl">
              {errorMessage}
            </p>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
