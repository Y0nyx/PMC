import {React,useEffect} from "react";
import StartButton from "../components/StartButton";
import SettingsIcon from "@mui/icons-material/Settings";
import HistoryButton from "../components/HistoryButton";
import OptionPieceButton from "../components/OptionPieceButton";
import { useContext } from "react";
import UIStateContext from "../Context/context";
import protocol from "../Protocol/protocol";
import Loading from "../components/Loading";
import { useLocation } from "react-router-dom";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import { idSubstring } from "../utils/utils";

export default function Home() {
  const uicontext = useContext(UIStateContext);
  const ipcRenderer = window.require("electron").ipcRenderer;
  const { state } = useLocation();
  const { resultat } = state || {};

  useEffect(() => {
  
   if(uicontext.ref_dev.current) uicontext.setState(protocol.state.idle);
  }, []); // Empty dependency array ensures the effect runs once on mount
  
  ipcRenderer.on("init", () => {
     uicontext.setState(protocol.state.idle);
  });

  return (
    <div className="w-screen h-screen overflow-scroll">
      {uicontext.state_state != protocol.state.init ? (
        <div className="box-border flex flex-col justify-center items-center w-full h-full p-5 bg-gray-100">
          <div className="flex justify-center items-center w-5/6 border-gray-200 bg-white rounded-lg p-5 my-4 h-24 shadow-sm">
            <div className="flex justify-start items-center w-1/3 h-full">
              <OptionPieceButton></OptionPieceButton>
            </div>
            <div className="w-1/3 text-4xl text-gray-800 uppercase font-normal flex justify-center items-center">
              {"Accueil"}
            </div>
            <div className="flex justify-end items-center w-1/3">
              <SettingsIcon className="text-gray-700 text-6xl hover:scale-105" />
            </div>
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
                {resultat ? (
                  <div className=" my-1 flex font-normal font-bold justify-center items-center text-3xl text-center">
                    <div className="flex font-normal font-bold justify-center items-center text-3xl my-20 text-center m-2">
                      {resultat && "La pièce " + idSubstring(resultat.id) + " succès"}
                    </div>
                    {uicontext.state_state === protocol.state.analysePass && (
                      <CheckCircleIcon className="text-green-500 text-5xl m-2" />
                    )}
                  </div>
                ) : (
                  <span>Cliquer sur DÉMARRER pour analyser</span>
                )}
              </div>
              <div className="flex p-3 justify-center items-center my-1">
                <StartButton />
                <HistoryButton />
              </div>
            </div>
            <div className="w-1/5 h-full"></div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col justify-center items-center w-full h-full">
          <span className="text-5xl font-normal text-black m-4">
            Initialisation...
          </span>
          <Loading></Loading>
        </div>
      )}
    </div>
  );
}
