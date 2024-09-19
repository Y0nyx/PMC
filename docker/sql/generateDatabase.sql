CREATE TABLE erreur_Soudure
(
  ID VARCHAR NOT NULL,
  Nom VARCHAR NOT NULL,
  Description VARCHAR NOT NULL,
  PRIMARY KEY (ID)
);

CREATE TABLE client
(
  ID VARCHAR NOT NULL,
  Nom VARCHAR NOT NULL,
  Email VARCHAR,
  Telephone VARCHAR,
  PRIMARY KEY (ID)
);

CREATE TABLE log
(
  ID VARCHAR NOT NULL,
  ID_client VARCHAR NOT NULL,
  Nom VARCHAR NOT NULL,
  PRIMARY KEY (ID),
  FOREIGN KEY (ID_client) REFERENCES Client(ID)
);

CREATE TABLE piece
(
  ID VARCHAR NOT NULL,
  Date TIMESTAMP NOT NULL,
  Photo VARCHAR NOT NULL,
  BoundingBox VARCHAR NOT NULL,
  Resultat INT NOT NULL,
  ID_client VARCHAR NOT NULL,
  ID_log VARCHAR NOT NULL,
  ID_type_piece VARCHAR NOT NULL,
  ID_erreur_Soudure VARCHAR NOT NULL,
  PRIMARY KEY (ID),
  FOREIGN KEY (ID_log) REFERENCES Log(ID),
  FOREIGN KEY (ID_erreur_Soudure) REFERENCES Erreur_Soudure(ID)
);


CREATE TABLE type_piece
(
  ID VARCHAR NOT NULL,
  Nom VARCHAR NOT NULL,
  Description VARCHAR NOT NULL,
  PRIMARY KEY (ID)
);

-- Inserting null erreur de soudurwe
INSERT INTO Erreur_Soudure (ID, Nom, Description) VALUES
('0', '', '');
-- Inserting the null client Client table
INSERT INTO Client (ID, Nom, Email, Telephone) VALUES
('0', '', '', '');

-- Inserting the null log
INSERT INTO Log (ID, ID_client,Nom) VALUES
('0', 0,'log1');
