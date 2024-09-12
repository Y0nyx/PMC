import React, { useContext, useEffect } from "react";
import CancelIcon from "@mui/icons-material/Cancel";
import { idSubstring, pieceParser } from "../utils/utils";
import UIStateContext from "../Context/context";
import { useNavigate } from "react-router-dom";
import { useParams } from "react-router";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import protocol from "../Protocol/protocol";
import ContinueButton from "../components/ContinueButton";

export default function AnalyseFailed() {
  const uicontext = useContext(UIStateContext);
  const ipcRenderer = window.require("electron").ipcRenderer;
  const { id } = useParams();
  const [piece, setPiece] = React.useState({
    id: "7631E40B-BB8A-4278-A5B7-1B2B8E15FAEF",
    nom: "",
    description: "",
    url: "",
    date: "",
  });
  const navigate = useNavigate();
  ipcRenderer.on("receivePiece", async (event, message) => {
    console.log("message", message);
    setPiece(await pieceParser(message));
  });

  useEffect(() => {
    console.log("state", uicontext.state_state);
    ipcRenderer.send("fetchPiece", id);
  }, []);


  function RestartCommand() {
    ipcRenderer.send("restart", piece.id);
    uicontext.setState(protocol.state.analyseInProgress);
    navigate("/analyse");
  }

  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="box-border flex flex-col justify-center items-center w-full h-full bg-gray-200 p-5">
        <div className="box-border  shadow-xl rounded-lg flex justify-between items-center p-5 w-5/6 h-4/6 border-gray-300 bg-gray-100">
          <img
            className="box-border object-cover w-7/12 h-full rounded-lg flex justify-end items-end"
            src={piece.url}
          ></img>
          <div className="flex w-5/12 flex-col justify-center items-center box-border p-3  ">
            <span className=" box-border font-normal text-xl m-2 flex justify-center items-center">{"La pièce " + idSubstring(piece.id) + " a échoué"}</span>
            <div className="box-border flex justify-between items-center w-full bg-white rounded-lg p-2 text-lg text-gray-800  font-normal m-1">
              <span>{"ID:"}</span>
              <span>{piece.id}</span>
            </div>
            <div className="box-border flex justify-between items-center w-full bg-white m-1 rounded-lg p-2 text-lg text-gray-800  font-normal ">
              <span>{"Résultat:"}</span>
              <span className="flex justify-center items-center">
                {piece.result}
                <CancelIcon className="text-red-500 text-3xl m-2" />
              </span>
            </div>

            <div className="box-border flex justify-between items-center w-full bg-white m-1 rounded-lg p-2 text-lg text-gray-800  font-normal">
              <span>{"Erreur:"}</span>
              <span>{piece.errorType}</span>
            </div>

            <div className="box-border flex justify-between items-center w-full bg-white m-1 rounded-lg p-2 text-lg text-gray-800  font-normal">
              <span>{"Description de l'erreur:"}</span>
              <span>{piece.errorDescription}</span>
            </div>
          </div>
        </div>
        <div className="box-border  flex border-gray-200 shadow-xl  rounded-2xl border p-3 justify-center items-center h-2/6 m-5">
        <div
          className="border-box flex justify-center items-center mx-6 font-bold font-normal text-3xl text-white text- hover:scale-110 bg-yellow-500 rounded-lg hover:bg-yellow-500 w-80 h-64 "
          onClick={RestartCommand}
        >
          <span>RECOMMENCER</span>
        </div>
        <ContinueButton />
      </div>
      </div>

    </div>
  );
}
