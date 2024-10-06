import React, { useState, useRef, useEffect } from "react";
import BackButton from "../components/BackButton";
import { useNavigate } from "react-router-dom";
import TextField from "@mui/material/TextField";
import Keyboard from "react-simple-keyboard";
import "react-simple-keyboard/build/css/index.css";
import CancelIcon from "@mui/icons-material/Cancel";
import { IconButton, Dialog, DialogContent, DialogTitle } from "@mui/material";

export default function CreerClient() {
  const ipcRenderer = window.require("electron").ipcRenderer;
  const [nom, setNom] = useState("");
  const [telephone, setTelephone] = useState("");
  const [email, setEmail] = useState("");
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [focus, setFocus] = useState();
  const [openError,setOpenError] = useState(false)
  const [errorMessage, setErrorMessage] = useState();
  // Create refs for the input fields
  const nomRef = useRef(null);
  const telephoneRef = useRef(null);
  const emailRef = useRef(null);

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

  const handleInputChange = (event) => {
    if (focus === "nom") setNom(event.target.value);
    else if (focus === "email") setEmail(event.target.value);
    else if (focus === "telephone") setTelephone(event.target.value);
  };

  const onChange = (inputValue) => {
    if (focus === "nom") setNom(inputValue);
    else if (focus === "email") setEmail(inputValue);
    else if (focus === "telephone") setTelephone(inputValue);
  };

  const handleClick = async () => {
    const result = await ipcRenderer.invoke("createClient", { nom, telephone, email });
    if (result) {
      back();
    }
  };

  const navigate = useNavigate();

  const back = () => {
    navigate("/optionpiece");
  };

  const onFocus = (name, elementRef) => {
    setKeyboardVisible(true);
    setFocus(name);

    // Scroll to the center of the screen
    elementRef.current.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  };

  return (
    <div className=" flex flex-col justify-center items-center w-screen">
      <div className=" box-border flex flex-col w-full h-full justify-center items-center bg-gray-200 px-5 py-10">
        <div className="box-border flex items-center w-5/6 h-1/6 border-gray-300 bg-gray-100 rounded-lg p-5 my-4">
          <div className="w-1/3"></div>
          <div className="w-1/3 text-3xl text-gray-800 uppercase font-normal flex justify-center items-center">
            {"Créer un Client "}
          </div>
          <BackButton back={back}></BackButton>
        </div>

        <div className=" box-border shadow-xl rounded-lg flex flex-col justify-center items-center w-5/6 h-5/6 p-5  border-gray-300 bg-gray-100">
          <div className="flex flex-col justify-center items-center my-4 w-full h-full">
            <div className="flex flex-col justify-around items-center my-2 w-full h-full">
              <div className="flex flex-col justify-start items-start w-1/2 p-10">
                <span className="font-normal text-lg">Nom</span>
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
              <div className="flex flex-col justify-start items-start w-1/2 p-10">
                <span className="font-normal text-lg">Téléphone</span>
                <TextField
                  className="w-full font-normal"
                  id="outlined-basic"
                  label="Téléphone"
                  value={telephone}
                  variant="outlined"
                  inputRef={telephoneRef}
                  onFocus={() => onFocus("telephone", telephoneRef)}
                  onChange={handleInputChange}
                />
              </div>
              <div className="flex flex-col justify-start items-start w-1/2 p-10">
                <span className="font-normal text-lg">Email</span>
                <TextField
                  className="w-full font-normal"
                  id="outlined-basic"
                  label="Email"
                  value={email}
                  variant="outlined"
                  inputRef={emailRef}
                  onFocus={() => onFocus("email", emailRef)}
                  onChange={handleInputChange}
                />
              </div>
            </div>
            <div
              className="mx-6 flex justify-center items-center font-bold text-3xl text-white font-normal bg-gray-900 rounded-lg hover:bg-white hover:text-black hover:scale-110 w-80 h-48 "
              onClick={handleClick}
            >
              <span>Créer un Client</span>
            </div>
            
          </div>
        </div>
      </div>
      {keyboardVisible && (
              <div className="sticky bottom-0 left-0 w-full z-50">
                          <Keyboard
                onChange={onChange}
                inputName={focus}
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
