import React, { useContext, useEffect, useState } from "react";
import CancelIcon from "@mui/icons-material/Cancel";
import { idSubstring, pieceParser } from "../utils/utils";
import { useNavigate } from "react-router-dom";
import { useParams } from "react-router";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import BackButton from "../components/BackButton";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { IconButton, Dialog, DialogContent, DialogTitle } from "@mui/material";
export default function PagePiece() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const { id } = useParams();
  const [imageSelected, setImageSelected] = useState(0);
  const [piece, setPiece] = React.useState();
  const [openError, setOpenError] = useState(false);
  const [errorMessage, setErrorMessage] = useState();
  const navigate = useNavigate();

  useEffect(() => {
    ipcRenderer.on("error", (event, error) => {
      console.log(error);
      setErrorMessage(error);
      setOpenError(true);
    });
    ipcRenderer.on("receivePiece", async (event, message) => {
      let parser = await pieceParser(message,true);

      if (parser.result === "succès") {
        setImageSelected(0);
      } else {
        let index = parser.images.findIndex(
          (image) => image.boundingBox !== undefined
        );
        if (index < 0) index = 0;
        setImageSelected(index);
      }
      setPiece(parser);
    });
    ipcRenderer.send("fetchPiece", id);

    return () => {
      ipcRenderer.removeAllListeners("receivePiece");
      ipcRenderer.removeAllListeners("error");
    };
  }, []);

  function back() {
    navigate("/history");
  }
  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="box-border flex flex-col justify-center items-center w-full h-full bg-gray-200 p-5">
        <div className="flex items-center w-5/6 border-gray-300 bg-gray-100 rounded-lg p-5 my-4">
          <div className="w-1/3"></div>
          <div className="w-1/3 text-xl  text-gray-800 uppercase font-normal flex justify-center items-center">
            {piece && "ID: " + idSubstring(piece.id)}
          </div>
          <BackButton back={back}></BackButton>
        </div>

        {piece && (
          <div className=" shadow-xl rounded-lg flex justify-center items-center p-5 w-5/6  border-gray-300 bg-gray-100">
            <div className="flex justify-center items-center flex-col w-7/12 h-full">
              {piece.result === "succès" ? (
                <div className=" box-border  flex justify-center items-center w-full h-full">
                  <img
                    className="box-border object-contain w-full h-full"
                    src={piece.images[imageSelected].url}
                  ></img>
                </div>
              ) : (
                <div className="overflow-hidden relative flex justify-center items-center w-full h-full">
                  <img
                    src={piece.images[imageSelected].url}
                    className="object-cover w-full h-full"
                  />

                  {piece.images[imageSelected].boundingBox &&
                    piece.images[imageSelected].boundingBox.box.map((box) => {
                      return (
                        <div
                          style={{
                            top: `${(box.yCenter - box.height / 2) * 100}%`,
                            left: `${(box.xCenter - box.width / 2) * 100}%`,
                            width: `${box.width * 100}%`,
                            height: `${box.height * 100}%`,
                          }}
                          className="absolute  bg-opacity-75 border-4 border-solid border-red-600 rounded"
                        ></div>
                      );
                    })}
                </div>
              )}

              <div className="flex flex-col justify-center items-center my-1 w-full h-1/6 p-2">
                <span className="flex w-full items-center justify-center text-gray-500 text-sm">
                  {imageSelected + 1 + "/" + piece.images.length}
                </span>
                <div className="flex justify-center items-center w-full h-full ">
                  <div
                    onClick={() => {
                      setImageSelected(Math.max(imageSelected - 1, 0));
                    }}
                    className="flex justify-center items-center w-60 h-full bg-gray-900 rounded-lg text-white mx-4"
                  >
                    <ArrowBackIcon></ArrowBackIcon>
                  </div>
                  <div
                    onClick={() => {
                      let index = Math.min(
                        imageSelected + 1,
                        piece.images.length - 1
                      );

                      setImageSelected(index);
                    }}
                    className="flex justify-center items-center w-60 h-full bg-gray-900 rounded-lg text-white mx-4"
                  >
                    <ArrowForwardIcon></ArrowForwardIcon>
                  </div>
                </div>
              </div>
            </div>

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
                  {piece.result === "succès" ? (
                    <CheckCircleIcon className="text-green-500 text-3xl m-2" />
                  ) : (
                    <CancelIcon className="text-red-500 text-3xl m-2" />
                  )}
                </span>
              </div>

              {/*        <div className=" flex justify-between items-center w-full bg-white m-1 rounded-lg p-4 text-xl text-gray-800  font-normal">
                <span>{"Erreur:"}</span>
                <span>{piece.errorType}</span>
              </div>

              <div className=" flex justify-between items-center w-full bg-white m-1 rounded-lg p-4 text-xl text-gray-800  font-normal">
                <span>{"Description de l'erreur:"}</span>
                <span>{piece.errorDescription}</span>
              </div> */}
            </div>
          </div>
        )}
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
