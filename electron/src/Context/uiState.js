import { useState, useRef } from "react";
import protocol from "../Protocol/protocol";

function useUIState() {
  //State pour l'affichage
  const ipcRenderer = window.require("electron").ipcRenderer;

  const [state_state, set_state] = useState(protocol.state.init);
  const [state_client, set_client] = useState({
    id: "1",
    nom: "Client1",
    email: "client1@example.com",
    telephone: "1234567890",
  });
  const [state_log, set_log] = useState({
    id: "1",
    id_client: "1",
    nom: "log1",
  });
  const [state_type_piece, set_type_piece] = useState({
    id: "1",
    nom: "1",
    description: "",
  });
  //Ref pour la logique
  const ref_state = useRef(protocol.state.init);
  const ref_config = useRef();
  const ref_client = useRef({
    id: "1",
    nom: "Client1",
    email: "client1@example.com",
    telephone: "1234567890",
  });
  const ref_log = useRef({ id: "1", id_client: "1", nom: "log1" });
  const ref_type_piece = useRef({ id: "1", nom: "1", description: "" });
  const ref_dev = useRef(true);
  ipcRenderer.send("fetchConfig");
  ipcRenderer.on("ReceiveConfig", (event, config) => {
    ref_config.current = config;
  });

  function setState(value) {
    ref_state.current = value;
    set_state(value);
  }

  function setClient(value) {
    ref_client.current = value;
    set_client(value);
  }

  function setLog(value) {
    ref_log.current = value;
    set_log(value);
  }

  function setTypePiece(value) {
    ref_type_piece.current = value;
    set_type_piece(value);
  }
  return {
    state_state,
    state_client,
    state_log,
    state_type_piece,
    ref_state,
    ref_config,
    ref_client,
    ref_log,
    ref_dev,
    setState,
    setClient,
    setLog,
    setTypePiece,
  };
}

export default useUIState;
