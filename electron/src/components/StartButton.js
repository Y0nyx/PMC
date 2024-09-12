import React from "react";
import { useNavigate } from "react-router-dom";
import { useContext } from "react";
import UIStateContext from "../Context/context";
import protocol from "../Protocol/protocol";
export default function StartButton() {
  const uicontext = useContext(UIStateContext);

  const ipcRenderer = window.require("electron").ipcRenderer;
  let navigate = useNavigate();

  function startCommand() {
    ipcRenderer.send("command",{"code":"start",data:""});
    uicontext.setState(protocol.state.loading);
    navigate("/loading")
  }
  return (
    <div
      className=" flex mx-6 font-bold justify-center items-center font-normal text-3xl text-white hover:scale-110 bg-green-400 rounded-lg hover:bg-green-700 w-80 h-64 "
      onClick={startCommand}
    >
      <span>DÉMARRER</span>
    </div>
  );
}
