import React from "react";
import { useNavigate } from "react-router-dom";
import SettingsIcon from "@mui/icons-material/Settings";
export default function SettingButton() {
  let navigate = useNavigate();

  function handleClick() {
    navigate("/setting");
  }
  return (
    <SettingsIcon
      onClick={handleClick}
      className="text-gray-700 text-6xl hover:scale-105"
    />
  );
}
