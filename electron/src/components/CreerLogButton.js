import React from "react";
import { useNavigate } from "react-router-dom";
export default function CreerLogButton() {
  let navigate = useNavigate();

  function handleClick() {
    navigate("/creerlog");
  }
  return (
    <div
      className="mx-6 flex justify-center items-center font-bold text-3xl text-white font-normal bg-purple-500 rounded-lg hover:bg-purple-700 hover:scale-110 w-80 h-48 "
      onClick={handleClick}
    >
      <span>Créer un Log</span>
    </div>
  );
}
