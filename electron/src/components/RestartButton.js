import React from "react";
import protocol from "../Protocol/protocol";
export default function RestartButton() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  let navigate = useNavigate();
  function RestartCommand() {
    ipcRenderer.send("restart",Pieceid);
    uicontext.setState(protocol.state.analyseInProgress);
    navigate("/analyse")
  }
  return (
    <div
      className="flex justify-center items-center mx-6 font-bold font-normal text-3xl text-white text- hover:scale-110 bg-yellow-500 rounded-lg hover:bg-yellow-500 w-80 h-64 "
      onClick={RestartCommand}
    >
      <span>RECOMMENCER</span>
    </div>
  );
}
