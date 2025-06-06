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
    uicontext.setState(protocol.state.loading);
    navigate("/loading",{ state: { command: "forward" } })
  }
  return (
    <div
      className=" flex mx-6 font-extrabold justify-center items-center font-normal text-4xl text-black hover:scale-110 bg-green-400 rounded-lg hover:bg-green-700 w-96 h-64 "
      onClick={startCommand}
    >
      <span>DÉMARRER</span>
    </div>
  );
}
