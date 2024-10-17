import React, { useContext, useState,useEffect } from "react";
import UIStateContext from "../Context/context";
import BackButton from "../components/BackButton";
import { useNavigate } from "react-router-dom";
import CancelIcon from "@mui/icons-material/Cancel";
import {
  IconButton,
  Dialog,
  DialogContent,
  DialogTitle,
} from "@mui/material";

export default function SettingMachine() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const uicontext = useContext(UIStateContext);
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [open2, setOpen2] = useState(false);
  const [reboot, setReboot] = useState(false);
  const [openError,setOpenError] = useState(false)
  const [errorMessage, setErrorMessage] = useState();
  const [exportLoading,setExportLoading] = useState(false)

  useEffect(() => {
    ipcRenderer.on("error", (event,error) => {
      console.log(error)
      setErrorMessage(error);
      setOpenError(true)
    });

    ipcRenderer.on("noUSB",()=>{
      setExportLoading(false)
      setErrorMessage("Aucune clé USB détecté")
    })
   
    ipcRenderer.on("exportData",()=>{
      setExportLoading(false)
    })

    return () => {
      ipcRenderer.removeAllListeners("error");
      ipcRenderer.removeAllListeners("noUSB")
    };
  }, []);

  const powerOffMachine = () => {
    ipcRenderer.send("powerOffMachine");
  };

  const rebootClick = () => {
    ipcRenderer.send("rebootMachine");
  };

  const resetData = () => {
    ipcRenderer.send("resetData");
  };

  const resetAll = () => {
    ipcRenderer.send("resetAll");
  };

  const exportData = () => {
    setExportLoading(true)
    ipcRenderer.send("export Data");
  };
  
  function back() {
    navigate("/");
  }

  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="box-border flex flex-col w-full h-full justify-center items-center bg-gray-200 p-5">
        <div className="flex items-center w-5/6 border-gray-300 bg-gray-100 rounded-lg p-5 my-4">
          <div className="w-1/3"></div>
          <div className="w-1/3 text-4xl text-gray-800 uppercase font-normal flex justify-center items-center">
            {"Options de la machine "}
          </div>
          <BackButton back={back}></BackButton>
        </div>

        <div className=" shadow-xl rounded-lg flex justify-around items-center p-5 w-5/6 h-full  border-gray-300 bg-gray-100">
          <div className="flex flex-col  justify-center items-center h-full w-1/3  ">
            <span className="text-4xl font-normal font-bold text-black my-5">
              GÉNÉRAUX
            </span>

            <div className="flex flex-col justify-around items-center h-full w-full  rounded-lg border-3 border-solid p-3">
              <div
                className=" flex mx-6 font-bold justify-center items-center font-normal text-3xl text-white hover:scale-110 bg-black rounded-lg hover:bg-white w-96 h-64 "
                onClick={powerOffMachine}
              >
                <span>ÉTEINDRE LA MACHINE</span>
              </div>
              <div
                className=" flex mx-6 font-bold justify-center items-center font-normal text-3xl text-white hover:scale-110 bg-black rounded-lg hover:bg-white w-96 h-64 "
                onClick={rebootClick}
              >
                <span>REDÉMARRER LA MACHINE</span>
              </div>
              <div
                className=" flex mx-6 font-bold justify-center items-center font-normal text-3xl text-white hover:scale-110 bg-black rounded-lg hover:bg-white w-96 h-64 "
                onClick={exportData}
              >
                <span>Exporter les données sur USB</span>
              </div>
              
            </div>
          </div>

          <div className="flex flex-col  justify-center items-center h-full w-1/3">
            <span className="text-4xl font-normal font-bold text-red-500 my-5">
              ZONE DANGER
            </span>
            <div className="flex flex-col justify-around items-center w-full rounded-lg h-full border-3 p-3 border-solid border-red-500">
              <div
                className=" flex mx-6 font-bold justify-center items-center font-normal text-center text-2xl text-white hover:scale-110 bg-red-400 rounded-lg hover:bg-red-700 w-96 h-64 my-2 "
                onClick={() => {
                  setOpen(true);
                }}
              >
                <span>RÉINITIALISER LES DONNÉES</span>
              </div>
              <Dialog
                open={open}
                aria-describedby="alert-dialog-slide-description"
              >
                <DialogTitle className="font-normal font-bold text-lg text-red-500 ">
                  RÉINITIALISER LES DONNÉES
                  <IconButton
                    aria-label="close"
                    onClick={() => setOpen(false)}
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
                      Vous êtes sur le point de réinitialiser les données de la
                      machine (Client,log,historique). Êtes-vous sûr de vouloir
                      continuer ?
                    </p>
                    <div
                      className=" flex mx-6 font-bold justify-center items-center font-normal text-center text-2xl text-white hover:scale-110 bg-red-400 rounded-lg hover:bg-red-700 w-80 h-56 "
                      onClick={() => {
                        setOpen(false);
                        resetData();
                      }}
                    >
                      <span>RÉINITIALISER LES DONNÉES</span>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
              <div
                className=" flex mx-6 font-bold justify-center items-center font-normal text-3xl text-white hover:scale-110 bg-red-400 rounded-lg hover:bg-red-700 w-96 h-64 my-2 "
                onClick={() => {
                  setOpen2(true);
                }}
              >
                <span>RÉINITIALISER LA MACHINE</span>
              </div>
              <Dialog
                open={open2}
                aria-describedby="alert-dialog-slide-description"
              >
                <DialogTitle className="font-normal font-bold text-lg text-red-500 ">
                  RÉINITIALISER LES DONNÉES
                  <IconButton
                    aria-label="close"
                    onClick={() => setOpen2(false)}
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
                  {!reboot && (
                    <div className="flex flex-col p-2 justify-center items-center text-gray-400 font-normal font-bold ">
                      <p className="text-justify  font-Cairo leading-normal text-3xl">
                        Vous êtes sur le point de réinitialiser la machine. À
                        utiliser seulement si la machine ne fonctionne plus.
                        Êtes-vous sûr de vouloir continuer ?
                      </p>
                      <div
                        className=" flex mx-6 font-bold justify-center items-center font-normal text-center text-2xl text-white hover:scale-110 bg-red-400 rounded-lg hover:bg-red-700 w-80 h-56 "
                        onClick={() => {
                          setReboot(true);
                          resetAll();
                        }}
                      >
                        <span>RÉINITIALISER LA MACHINE</span>
                      </div>
                    </div>
                  )}
                  {reboot && (
                    <div className="flex flex-col p-2 justify-center items-center text-gray-400 font-normal font-bold ">
                      <p className="text-justify  font-Cairo leading-normal text-3xl">
                        REBOOT...
                      </p>
                    </div>
                  )}
                </DialogContent>
              </Dialog>
            </div>
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


      <Dialog
        open={exportLoading}
        aria-describedby="alert-dialog-slide-description"
      >
        <DialogTitle className="font-normal font-bold text-lg text-red-500 ">
          Exporter sur clé USB
          <IconButton
            aria-label="close"
            onClick={() => setExportLoading(false)}
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
              Exportation en cours....
            </p>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
