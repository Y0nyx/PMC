import React from "react";
import { useNavigate } from "react-router-dom";
export default function CreerClientButton() {
  let navigate = useNavigate();

  function handleClick() {
    navigate("/creerclient");
  }
  return (
    <div
      className="mx-6 flex justify-center items-center font-bold text-3xl text-white font-normal bg-blue-600 rounded-lg hover:bg-blue-700 hover:scale-110 w-96 h-52 "
      onClick={handleClick}
    >
      <span>Créer un Client</span>
    </div>
  );
}
