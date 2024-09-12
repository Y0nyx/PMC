import React from "react";
import CircularProgress from '@mui/material/CircularProgress';

export default function Loading() {
  return (
    <div className="flex justify-center items-center">
    <CircularProgress style={{'color': 'black'}} size={100} />
  </div>
  );
}
