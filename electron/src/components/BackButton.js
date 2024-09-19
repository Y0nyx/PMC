import React from "react";
import CancelIcon from "@mui/icons-material/Cancel";

export default function BackButton({back}) {
  return (
    <div onClick={back} className="flex justify-end items-center w-1/3">
    <CancelIcon className="text-red-500 text-6xl hover:scale-105" />
  </div>
  );
}
