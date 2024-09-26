import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useContext } from "react";
import UIStateContext from "../Context/context";
import protocol from "../Protocol/protocol";
import HistoryTable from "../components/HistoryTable";
import { piecesParser } from "../utils/utils";
import BackButton from "../components/BackButton";

export default function History() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const uicontext = useContext(UIStateContext);
  const navigate = useNavigate();
  const [pieces, setPieces] = useState([]);



  useEffect(() => {

    ipcRenderer.on("receivePieces", async (event, message) => {
      setPieces(await piecesParser(message));
    });
  
    ipcRenderer.send(
      "fetchPieces",
      uicontext.ref_client.current.id,
      uicontext.ref_log.current.id
    );

    return () => {
      ipcRenderer.removeAllListeners('receivePieces');
    };
  }, []); // Empty dependency array ensures the effect runs once on mount

  function back() {
      navigate("/"); 
    }
  

  return (
<div className="w-screen">
<div className="box-border flex flex-col justify-center items-center w-full h-full p-5 bg-gray-100">
<div className="flex justify-center items-center w-5/6 border-gray-200 bg-white rounded-lg p-5 my-4 h-24 shadow-sm">
  <div className="flex justify-start items-center w-1/3 h-full">
    
  </div>
  <div className="w-1/3 text-4xl text-gray-800 uppercase font-normal flex justify-center items-center">
    {"Historique"}
  </div>
  <div className="flex justify-end items-center w-1/3">
    <BackButton back={back} />
  </div>
</div>
<div className="shadow-xl rounded-lg flex justify-center items-center w-5/6  border-gray-300 bg-gray-100">
{pieces.length > 0 && <HistoryTable rows_={pieces}></HistoryTable>}
</div>
</div>

</div>
  );
}
