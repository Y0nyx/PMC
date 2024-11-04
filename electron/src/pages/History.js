import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useContext } from "react";
import UIStateContext from "../Context/context";
import HistoryTable from "../components/HistoryTable";
import { piecesParser } from "../utils/utils";
import BackButton from "../components/BackButton";
import Loading from "../components/Loading";
import CancelIcon from "@mui/icons-material/Cancel";
import { IconButton, Dialog, DialogContent, DialogTitle } from "@mui/material";

export default function History() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const uicontext = useContext(UIStateContext);
  const navigate = useNavigate();
  const [pieces, setPieces] = useState([]);
  const [openError,setOpenError] = useState(false)
  const [errorMessage, setErrorMessage] = useState();
  const [loading, setLoading] = useState(true);
  const page_ref = useRef(0)
  const rowsPerPage_ref = useRef(4)


  useEffect(() => {

    ipcRenderer.on("error", (event,error) => {
      console.log(error)
      setErrorMessage(error);
      setOpenError(true)
    });

    ipcRenderer.on("receivePieces", async (event, message) => {
      setPieces(await piecesParser(message,rowsPerPage_ref.current,page_ref.current));
      setLoading(false);
    });

    fetchNextPieces()

    return () => {
      ipcRenderer.removeAllListeners("receivePieces");
      ipcRenderer.removeAllListeners("error");
    };
  }, []); // Empty dependency array ensures the effect runs once on mount


  function fetchNextPieces()
  {
    setLoading(true);
    ipcRenderer.send(
      "fetchPieces",
      uicontext.ref_client.current.id,
      uicontext.ref_log.current.id,
    );

  }
  

function setPageParent(value){
  page_ref.current = value
}

function setRowsPerPageParent(value){
  rowsPerPage_ref.current = value
}


  function back() {
    navigate("/");
  }

  return (
    <div className="box-border w flex flex-col justify-start items-center w-screen min-h-screen p-5 bg-gray-100">
      <div className="flex justify-center items-center w-5/6 border-gray-200 bg-white rounded-lg p-5 my-4 h-24 shadow-sm">
        <div className="flex justify-start items-center w-1/3 h-full"></div>
        <div className="w-1/3 text-4xl text-gray-800 uppercase font-normal flex justify-center items-center">
          {"Historique"}
        </div>
        <div className="flex justify-end items-center w-1/3">
          <BackButton back={back} />
        </div>
      </div>
      {loading ? (
        <div className="box-border flex flex-col justify-center items-center w-full h-full my-auto ">
          <span className="text-5xl font-normal text-black m-4">
            Chargement...
          </span>
          <Loading></Loading>
        </div>
      ) : (
        <div className="shadow-xl rounded-lg flex justify-center items-center w-5/6  border-gray-300 bg-gray-100">
          {pieces.length > 0 && <HistoryTable rows_={pieces} rowsPerPage={rowsPerPage_ref.current} setRowsPerPage={setRowsPerPageParent} page={page_ref.current
          } setPage={setPageParent} fetchNextPieces={fetchNextPieces}></HistoryTable>}
        </div>
      )}
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
