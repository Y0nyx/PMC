import React, { useContext, useEffect, useState } from "react";
import CancelIcon from "@mui/icons-material/Cancel";
import { idSubstring, pieceParser } from "../utils/utils";
import UIStateContext from "../Context/context";
import { useNavigate } from "react-router-dom";
import { useParams } from "react-router";
import protocol from "../Protocol/protocol";
import ContinueButton from "../components/ContinueButton";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";

export default function AnalyseFailed() {
  const uicontext = useContext(UIStateContext);
  const ipcRenderer = window.require("electron").ipcRenderer;
  const [imageSelected, setImageSelected] = useState(0);
  const { id } = useParams();
  const [piece, setPiece] = React.useState();
  const navigate = useNavigate();

  useEffect(() => {
    ipcRenderer.on("receivePiece", async (event, message) => {
      let parser = await pieceParser(message);
      setPiece(parser);

      let index = parser.images.findIndex(
        (image) => image.boundingBox != undefined
      );
      if (index < 0) index = 0;
      setImageSelected(index);
    });
    ipcRenderer.send("fetchPiece", id);

    return () => {
      ipcRenderer.removeAllListeners("receivePiece");
    };
  }, []);

  function RestartCommand() {
    ipcRenderer.send("restart", piece.id);
    uicontext.setState(protocol.state.analyseInProgress);
    navigate("/analyse");
  }

  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="box-border flex flex-col justify-center items-center w-full h-full bg-gray-200 p-5">
        {piece && (
          <div className="box-border  shadow-xl rounded-lg flex justify-between items-center p-5 w-5/6 h-4/6 border-gray-300 bg-gray-100">
            <div className="box-border flex justify-center items-center flex-col w-7/12 h-full">
              {piece.result == "succès" ? (
                <div className="box-border flex justify-center items-center w-full h-full">
                  <img
                    className="box-border object-contain w-full h-full"
                    src={piece.images[imageSelected].url}
                  ></img>
                </div>
              ) : (
                <div className="overflow-hidden box-border relative flex justify-center items-center w-full h-full">
                  <img
                    src={piece.images[imageSelected].url}
                    className="object-contain w-full h-full"
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

              <div className="box-border flex flex-col justify-center items-center my-1 w-full h-2/6 p-2">
                <span className="flex w-full items-center justify-center text-gray-500 text-sm">
                  {imageSelected + 1 + "/" + piece.images.length}
                </span>
                <div className="flex justify-center items-center w-full h-full ">
                  <div
                    onClick={() => {
                      setImageSelected(Math.max(imageSelected - 1, 0));
                    }}
                    className="flex justify-center items-center w-60 h-20 bg-gray-900 rounded-lg text-white mx-4"
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
                    className="box-border flex justify-center items-center w-60 h-20 bg-gray-900 rounded-lg text-white mx-4"
                  >
                    <ArrowForwardIcon></ArrowForwardIcon>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex w-5/12 flex-col justify-center items-center box-border p-3  ">
              <span className=" box-border font-normal text-xl m-2 flex justify-center items-center">
                {"La pièce " + idSubstring(piece.id) + " a échoué"}
              </span>
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
        )}

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
