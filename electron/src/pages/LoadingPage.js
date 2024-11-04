import React, { useContext, useEffect, useState } from "react";
import UIStateContext from "../Context/context";
import Loading from "../components/Loading";
import { useNavigate } from "react-router-dom";
import protocol from "../Protocol/protocol";
import CancelIcon from "@mui/icons-material/Cancel";
import { useLocation } from "react-router-dom";
import { IconButton, Dialog, DialogContent, DialogTitle } from "@mui/material";
export default function LoadingPage() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  let navigate = useNavigate();
  const [openError, setOpenError] = useState(false);
  const [errorMessage, setErrorMessage] = useState();
  const uicontext = useContext(UIStateContext);
  const location = useLocation();
  const { command, resultat } = location.state || {};

  useEffect(() => {
    if (command == "forward") {
      ipcRenderer.send("forward");
      uicontext.ref_plc_ready = false;
    }

    if (command == "backward") {
      ipcRenderer.send("backward");
      uicontext.ref_plc_ready = false;
    }

    if (command == "stop") {
      ipcRenderer.send("command", { code: "stop" });
    }

    ipcRenderer.on("stop", () => {
      uicontext.ref_plc_ready = false;
      ipcRenderer.send("backward");
    });

    ipcRenderer.on("init", () => {
      if (command == "forward") {
        ipcRenderer.send("command", { code: "start" });
      }
    });
    ipcRenderer.on("ready", () => {
      uicontext.ref_plc_ready = true;
      if (command == "forward") {
        ipcRenderer.send("command", { code: "start" });
      }

      if (command == "backward") {
        if (resultat.resultat == 1) navigate("/", { state: { resultat } });
        if (resultat.resultat == 0) navigate("/analysefailed/" + resultat.id);
      }

      if (command == "stop") {
        uicontext.setState(protocol.state.idle);
        navigate("/");
      }
    });

    ipcRenderer.on("error", (event, error) => {
      console.log(error);
      setErrorMessage(error);
      setOpenError(true);
    });
    ipcRenderer.on("start", () => {
      console.log("start");
      uicontext.setState(protocol.state.analyseInProgress);
      navigate("/analyse");
    });

    return () => {
      ipcRenderer.removeAllListeners("stop");
      ipcRenderer.removeAllListeners("start");
      ipcRenderer.removeAllListeners("error");
      ipcRenderer.removeAllListeners("ready");
    };
  }, []);

  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="flex flex-col justify-center items-center w-full h-full">
        <span className="text-5xl font-normal text-black m-4">
          Chargement... Ne pas ouvrir la porte
        </span>
        <Loading></Loading>
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
