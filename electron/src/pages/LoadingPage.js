import React, { useContext } from "react";
import UIStateContext from "../Context/context";
import Loading from "../components/Loading";
import { useNavigate } from "react-router-dom";
import protocol from "../Protocol/protocol";
export default function LoadingPage() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  let navigate = useNavigate();
   const uicontext = useContext(UIStateContext);
 console.log("loading page")
  ipcRenderer.on("start", () => {
    console.log("start");
    uicontext.setState(protocol.state.analyseInProgress)
    navigate("/analyse");
  })

  ipcRenderer.on("stop", () => {
    uicontext.setState(protocol.state.idle)
    console.log("stop");
    navigate("/");
  })


  return (
    <div className="w-screen h-screen overflow-hidden">
      <div className="flex flex-col justify-center items-center w-full h-full">
        <span className="text-5xl font-normal text-black m-4">
          Chargement...
        </span>
        <Loading></Loading>
      </div>
    </div>
  );
}
