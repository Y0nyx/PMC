import React from "react";
import { useNavigate } from "react-router-dom";

export default function OptionPieceButton() {

  let navigate = useNavigate();

  function handleClick() {
    navigate("/optionpiece");
  }
  return (
    <div
      className="mx-6 flex justify-center items-center font-bold text-2xl text-white font-normal bg-gray-900  rounded-lg hover:bg-slate-600 hover:scale-110 h-full p-2 "
      onClick={handleClick}
    >
      <span>Pièce/Client/Log</span>
    </div>
  );
}