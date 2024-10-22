import React from "react";
import { Link } from "react-router-dom";

export default function HistoryButton() {
  const ipcRenderer = window.require("electron").ipcRenderer;

  return (
    <Link
      className="flex text-4xl text-black font-extrabold font-normal justify-center items-center no-underline"
      to={"/history"}
    >
      <div className="mx-6 flex justify-center items-center bg-blue-500 rounded-lg hover:bg-blue-900 hover:scale-110 w-96 h-64 ">
        HISTORIQUE
      </div>
    </Link>
  );
}
