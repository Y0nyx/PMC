import React, { useContext, useEffect } from "react";
import CancelIcon from "@mui/icons-material/Cancel";
import { idSubstring, pieceParser } from "../utils/utils";
import { useNavigate } from "react-router-dom";
import { useParams } from "react-router";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import BackButton from "../components/BackButton";

export default function PagePiece() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const { id } = useParams();
  const [piece, setPiece] = React.useState({
    id: "7631E40B-BB8A-4278-A5B7-1B2B8E15FAEF",
    nom: "",
    description: "",
    boundingbox: "",
    box: { xCenter: 0, yCenter: 0, width: 0, height: 0 },
    url: "",
    date: "",
  });
  const navigate = useNavigate();
  ipcRenderer.on("receivePiece", async (event, message) => {
    setPiece(await pieceParser(message));
  });

  useEffect(() => {
    ipcRenderer.send("fetchPiece", id);
  }, []);

  useEffect(() => {
    console.log(piece);
  }, [piece]);
  function back() {
    navigate("/history");
  }
  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="box-border flex flex-col justify-center items-center w-full h-full bg-gray-200 p-5">
        <div className="flex items-center w-5/6 border-gray-300 bg-gray-100 rounded-lg p-5 my-4">
          <div className="w-1/3"></div>
          <div className="w-1/3 text-xl  text-gray-800 uppercase font-normal flex justify-center items-center">
            {"ID: " + idSubstring(piece.id)}
          </div>
          <BackButton back={back}></BackButton>
        </div>

        <div className=" shadow-xl rounded-lg flex justify-center items-center p-5 w-5/6  border-gray-300 bg-gray-100">
          {piece.result == "succès" ? (
            <img src={piece.url}></img>
          ) : (
            <div className="relative w-full h-full">
              <img src={piece.url} className="w-full h-full" />
              <div
                style={{
                  top: `${piece.box.yCenter * 100}%`,
                  left: `${piece.box.xCenter * 100}%`,
                  width: `${piece.box.width * 100}%`,
                  height: `${piece.box.height * 100}%`,
                }}
                className="absolute  bg-opacity-75 border-4 border-solid border-red-600 rounded"
              ></div>
            </div>
          )}
          <div className="flex w-5/12 flex-col justify-center items-center mx-20 p-5">
            <div className="flex justify-between items-center w-full bg-white rounded-lg p-4 text-gray-800  font-normal m-1">
              <span className=" text-xl">{"ID:"}</span>
              <span className=" text-sm">{piece.id}</span>
            </div>
            <div className="flex justify-between items-center w-full bg-white rounded-lg p-4 text-xl text-gray-800  font-normal m-1">
              <span>{"Client:"}</span>
              <span>{piece.nom_client}</span>
            </div>
            <div className=" flex justify-between items-center w-full bg-white rounded-lg p-4 text-xl text-gray-800  font-normal m-1">
              <span>{"#Log:"}</span>
              <span>{piece.id_log}</span>
            </div>
            <div className="flex justify-between items-center w-full bg-white m-1 rounded-lg p-4 text-xl text-gray-800  font-normal">
              <span>{"Type de Pièce: "}</span>
              <span>{piece.nom_type_piece}</span>
            </div>
            <div className=" flex justify-between items-center w-full bg-white m-1 rounded-lg p-4 text-xl text-gray-800  font-normal">
              <span>{"Date:"}</span>
              <span>{piece.date + " " + piece.hour}</span>
            </div>
            <div className="flex justify-between items-center w-full bg-white m-1 rounded-lg p-4 text-xl text-gray-800  font-normal ">
              <span>{"Résultat:"}</span>
              <span className="flex justify-center items-center">
                {piece.result}
                {piece.result == "succès" ? (
                  <CheckCircleIcon className="text-green-500 text-3xl m-2" />
                ) : (
                  <CancelIcon className="text-red-500 text-3xl m-2" />
                )}
              </span>
            </div>

            <div className=" flex justify-between items-center w-full bg-white m-1 rounded-lg p-4 text-xl text-gray-800  font-normal">
              <span>{"Erreur:"}</span>
              <span>{piece.errorType}</span>
            </div>

            <div className=" flex justify-between items-center w-full bg-white m-1 rounded-lg p-4 text-xl text-gray-800  font-normal">
              <span>{"Description de l'erreur:"}</span>
              <span>{piece.errorDescription}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
