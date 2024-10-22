import React, { useEffect, useContext,useState,useRef } from "react";
import BackButton from "../components/BackButton";
import { useNavigate } from "react-router-dom";
import TextField from "@mui/material/TextField";
import Box from "@mui/material/Box";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";
import UIStateContext from "../Context/context";
import Keyboard from "react-simple-keyboard";
import "react-simple-keyboard/build/css/index.css";
import CancelIcon from "@mui/icons-material/Cancel";
import { IconButton, Dialog, DialogContent, DialogTitle } from "@mui/material";

export default function CreerLog() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const uicontext = useContext(UIStateContext);
  const [nom, setNom] = React.useState("");
  const [listClient, setListClient] = React.useState([]);
  const [openError,setOpenError] = useState(false)
  const [errorMessage, setErrorMessage] = useState();
  const [selectedClient, setSelectedClient] = React.useState(
    uicontext.state_client
  );
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  useEffect(() => {
    ipcRenderer.on("error", (event,error) => {
      console.log(error)
      setErrorMessage(error);
      setOpenError(true)
    });

    return () => {
      ipcRenderer.removeAllListeners("error");
    };
  }, []);

    // Create refs for the input fields
    const nomRef = useRef(null);

    const onFocus = (name, elementRef) => {
      setKeyboardVisible(true);
  
      // Scroll to the center of the screen
      elementRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    };

    const handleInputChange = (event) => {
        setNom(event.target.value)
    };
  
    const onChange = (inputValue) => {
      setNom(inputValue);
    
    };

  function fetchClients() {
    ipcRenderer.send("fetchClients");
  }



  function handleClick() {
    createLog();
  }

  async function createLog() {
    let result = await ipcRenderer.invoke("createLog", {
      nom: nom,
      id_client: selectedClient.id,
    });
    if (result) {
      back();
    }
  }

  const navigate = useNavigate();

  function back() {
    navigate("/optionpiece");
  }

  function handleNom(event) {
    setNom(event.target.value);
  }

  function handleClientChange(event) {
    setSelectedClient(event.target.value);
  }

  useEffect(() => {
    ipcRenderer.on("receiveClients", async (event, _client) => {
      setListClient(_client);
    });
    fetchClients();

    return () => {
      ipcRenderer.removeAllListeners('receiveClients');
    };
  }, []);

  return (
    <div className=" flex flex-col justify-center items-center w-screen h-screen">
      <div className=" box-border flex flex-col w-full h-full justify-center items-center bg-gray-200 px-5 py-10">
        <div className="box-border flex items-center w-5/6 h-1/6 border-gray-300 bg-gray-100 rounded-lg p-5 my-4">
          <div className="w-1/3"></div>
          <div className="w-1/3 text-3xl text-gray-800 uppercase font-normal flex justify-center items-center">
            {"Créer un Log "}
          </div>
          <BackButton back={back}></BackButton>
        </div>

        <div className=" box-border shadow-xl rounded-lg flex flex-col justify-center items-center w-5/6 h-5/6 p-5  border-gray-300 bg-gray-100">
          <div className="flex flex-col justify-around items-center my-4 w-full h-full">
            <div className="flex flex-col justify-between items-center rounded-lg p-2 text-3xl text-gray-800  font-normal m-1">
              <span>{"À quel client voulez-vous ajouté le Log ? :"}</span>

              <Box className="w-1/2 my-5 rounded-3 border-gray-400">
                <FormControl fullWidth>
                  <InputLabel id="Client">Client</InputLabel>
                  <Select
                    id="Client"
                    value={
                      (listClient &&
                        listClient.find((c) => c.id === selectedClient.id)) ||
                      ""
                    }
                    label="Client"
                    onChange={handleClientChange}
                  >
                    {listClient &&
                      listClient.map((client) => {
                        return <MenuItem value={client}>{client.nom}</MenuItem>;
                      })}
                  </Select>
                </FormControl>
              </Box>
            </div>
            <div className="flex flex-col justify-around items-center my-2 w-full">
              <div className="flex flex-col justify-start items-start w-1/2 p-10">
                <span className="font-normal text-lg">Nom du Log</span>
                <TextField
                  className="w-full font-normal"
                  id="outlined-basic"
                  label="Nom"
                  value={nom}
                  variant="outlined"
                  inputRef={nomRef}
                  onFocus={() => onFocus("nom", nomRef)}
                  onChange={handleInputChange}
                />
              </div>
            </div>
            <div
              className="mx-6 flex justify-center items-center font-bold text-3xl text-white font-normal bg-gray-900 rounded-lg hover:bg-white hover:text-black hover:scale-110 w-80 h-48 "
              onClick={handleClick}
            >
              <span>Créer un Log</span>
            </div>
          </div>
        </div>
      </div>
      {keyboardVisible && (
              <div className="sticky bottom-0 left-0 w-full z-50">
                          <Keyboard
                onChange={onChange}
                inputName={"nom"}
                onKeyPress={(button) => {
                  if (button === "{enter}") {
                    setKeyboardVisible(false);
                  }
                }}
              />
              </div>
    
            )}
             <Dialog
        open={openError}
        aria-describedby="alert-dialog-slide-description"
      >
        <DialogTitle className="font-normal font-bold text-lg text-red-500 ">
          ERREUR
          <IconButton
            aria-label="close"
            onClick={() => setOpenError(false)}
            sx={{
              position: "absolute",
              right: 8,
              top: 8,
            }}
          >
            <div className="flex justify-center items-center w-full">
              <CancelIcon className="text-red-500 text-6xl hover:scale-105" />
            </div>
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <div className="flex flex-col p-2 justify-center items-center text-gray-400 font-normal font-bold ">
            <p className="text-justify  font-Cairo leading-normal text-3xl">
              {errorMessage}
            </p>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
