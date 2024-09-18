import React, { useContext, useEffect } from "react";
import UIStateContext from "../Context/context";
import BackButton from "../components/BackButton";
import { useNavigate } from "react-router-dom";
import CreerClientButton from "../components/CreerClientButton";
import CreerLogButton from "../components/CreerLogButton";
import Box from "@mui/material/Box";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";

export default function SettingMachine() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const uicontext = useContext(UIStateContext);
  const navigate = useNavigate();

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

            <div className="flex flex-col justify-around items-center h-full w-full  rounded-lg border border-solid">
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
            </div>
          </div>

          <div className="flex flex-col  justify-center items-center h-full w-1/3">
            <span className="text-4xl font-normal font-bold text-red-500 my-5">
              ZONE DANGER
            </span>
            <div className="flex flex-col justify-around items-center w-full rounded-lg h-full border border-solid border-red-500">
              <div
                className=" flex mx-6 font-bold justify-center items-center font-normal text-center text-2xl text-white hover:scale-110 bg-red-400 rounded-lg hover:bg-red-700 w-96 h-64 "
                onClick={resetData}
              >
                <span>RÉINITIALISER LES DONNÉES</span>
              </div>
              <div
                className=" flex mx-6 font-bold justify-center items-center font-normal text-3xl text-white hover:scale-110 bg-red-400 rounded-lg hover:bg-red-700 w-96 h-64 "
                onClick={resetAll}
              >
                <span>RÉINITIALISER LA MACHINE</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
