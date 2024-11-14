import React, { useContext, useEffect, useState } from "react";
import StopButton from "../components/StopButton";
import HistoryButton from "../components/HistoryButton";
import UIStateContext from "../Context/context";
import protocol from "../Protocol/protocol";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import { useNavigate } from "react-router-dom";
import CancelIcon from "@mui/icons-material/Cancel";
import { IconButton, Dialog, DialogContent, DialogTitle } from "@mui/material";
import NoMeetingRoomIcon from '@mui/icons-material/NoMeetingRoom';
export default function Analyse() {
  const uicontext = useContext(UIStateContext);
  const [texte, setTexte] = useState("Analyse en cours ...");
  const ipcRenderer = window.require("electron").ipcRenderer;
  const navigate = useNavigate();
  const [openError, setOpenError] = useState(false);
  const [errorMessage, setErrorMessage] = useState();
  useEffect(() => {
    ipcRenderer.on("error", (event, error) => {
      console.log(error);
      setErrorMessage(error);
      setOpenError(true);
    });

    ipcRenderer.on("porte", (event, error) => {
      setErrorMessage(error);
      setOpenError(true);
    });

    ipcRenderer.on("resultat", async (event, data) => {
      const newPiece = {
        url: data.url,
        boundingbox: data.boundingbox,
        resultat: data.resultat == true ? 1 : 0,
        id_client: uicontext.state_client.id,
        id_log: uicontext.state_log.id,
        id_type_piece: uicontext.state_type_piece.id,
        id_erreur_soudure: data.erreurSoudure,
      };

      let piece = await ipcRenderer.invoke("createPiece", newPiece);

      if (data.resultat == true) {
        const resultat = piece;
        uicontext.setState(protocol.state.analysePass);
        navigate("/loading", { state: { command: "backward", resultat } });
      } else {
        const resultat = piece;
        uicontext.setState(protocol.state.analyseFailed);
        navigate("/loading", { state: { command: "backward", resultat } });
      }
    });

    return () => {
      ipcRenderer.removeAllListeners("resultat");
      ipcRenderer.removeAllListeners("error");
    };
  }, []);

  useEffect(() => {
    // uicontext.setState(protocol.state.analysePass) // For testing verification pass
    switch (uicontext.ref_state.current) {
      case protocol.state.analyseInProgress:
        setTexte("Analyse en cours ... Ne pas ouvrir la porte");
        break;

      case protocol.state.analysePass:
        setTexte("Pièce 11 : Succès");
    }
  }, [uicontext.state_state]);
  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="box-border flex flex-col justify-center items-center w-full h-full p-5 bg-gray-100">
        <div className="flex justify-center items-center w-5/6 border-gray-200 bg-white rounded-lg p-5 my-4 h-24 shadow-sm">
          <div className="flex justify-start items-center w-1/3 h-full"></div>
          <div className="w-1/3 text-4xl text-gray-800 uppercase font-normal flex justify-center items-center">
          <NoMeetingRoomIcon className="text-4xl text-red-600 my-3"></NoMeetingRoomIcon>
            {"Analyse"}
          </div>
          <div className="flex justify-end items-center w-1/3"></div>
        </div>
        <div className="shadow-xl rounded-lg flex justify-around items-center w-5/6 h-full border-gray-300 bg-gray-100 p-5">
          <div className="flex flex-col justify-center items-center w-1/5 h-full">
            <div className="flex justify-between items-center w-5/6 bg-white rounded-lg p-4 text-xl text-gray-800  font-normal my-1 ">
              <span>{"Client:"}</span>
              <span>{uicontext.state_client.nom}</span>
            </div>
            <div className="flex justify-between items-center w-5/6 bg-white rounded-lg p-4 text-xl text-gray-800  font-normal my-1">
              <span>{"#Log:"}</span>
              <span>{uicontext.state_log.nom}</span>
            </div>
            <div className="flex justify-between items-center w-5/6 bg-white rounded-lg p-4 text-xl text-gray-800  font-normal my-1">
              <span>{"Pièce:"}</span>
              <span>{uicontext.state_type_piece.nom}</span>
            </div>
          </div>
          <div className="flex flex-col justify-around items-center w-3/5 my-10">
            <div className=" my-1 flex font-normal font-bold justify-center items-center text-3xl text-center">
              <div className="flex font-normal font-bold justify-center items-center text-3xl my-20 text-center m-2">
              <NoMeetingRoomIcon className="text-6xl text-red-600 my-3"></NoMeetingRoomIcon>
                {texte}
              </div>
              {uicontext.state_state === protocol.state.analysePass && (
                <CheckCircleIcon className="text-green-500 text-5xl m-2" />
              )}
            </div>
            <div className="flex p-3 justify-center items-center my-1">
              <StopButton />
            </div>
          </div>
          <div className="w-1/5 h-full"></div>
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
