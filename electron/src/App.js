import "./App.css";
import Home from "./pages/Home";
import History from "./pages/History";
import Analyse from "./pages/Analyse";
import AnalyseFailed from "./pages/AnalyseFailed";
import PagePiece from "./pages/PagePiece";
import { Route, Routes } from "react-router-dom";
import { HashRouter } from "react-router-dom";
import uiContext from "./Context/context";
import useUIState from "./Context/uiState";
import OptionPiece from "./pages/OptionPiece";
import CreerClient from "./pages/CreerClient";
import CreerLog from "./pages/CreerLog";
import { StyledEngineProvider } from "@mui/material/styles";
import LoadingPage from "./pages/LoadingPage";
import SettingMachine from "./pages/SettingMachine";
import { useEffect, useState } from "react";

function App() {
  const uiState = useUIState();
  
  return (
    <StyledEngineProvider injectFirst>
      <uiContext.Provider value={uiState}>
        <HashRouter>
          <Routes>
            <Route path="/" exact Component={Home} />
            <Route path="history" Component={History} />
            <Route path="analyse" Component={Analyse} />
            <Route path="analysefailed/:id" Component={AnalyseFailed} />
            <Route path="piece/:id" Component={PagePiece} />
            <Route path="optionpiece" Component={OptionPiece} />
            <Route path="creerclient" Component={CreerClient} />
            <Route path="creerlog" Component={CreerLog} />
            <Route path="loading" Component={LoadingPage} />
            <Route path="setting" Component={SettingMachine} />
          </Routes>
        </HashRouter>
      </uiContext.Provider>
    </StyledEngineProvider>
  );
}

export default App;
