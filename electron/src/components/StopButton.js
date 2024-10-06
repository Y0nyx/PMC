import React from "react";
import { useNavigate } from "react-router-dom";
import { useContext } from "react";
import UIStateContext from "../Context/context";
import protocol from "../Protocol/protocol";
export default function StopButton() {
  const uicontext = useContext(UIStateContext);
  const ipcRenderer = window.require("electron").ipcRenderer;
  let navigate = useNavigate();

  function stopCommand() {
    uicontext.setState(protocol.state.loading);
    navigate("/loading",{ state: { command: "backward" } })
  }

  return (
    <div
      className="mx-6 flex justify-center items-center font-bold text-3xl text-white font-normal bg-red-400 rounded-lg hover:bg-red-700 hover:scale-110 w-80 h-64 "
      onClick={stopCommand}
    >
      <span>ARRÊTER</span>
    </div>
  );
}
