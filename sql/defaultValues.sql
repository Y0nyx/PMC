-- Inserting null erreur de soudurwe
INSERT INTO Erreur_Soudure (ID, Nom, Description) VALUES
('0', '', '');
-- Inserting the null client Client table
INSERT INTO Client (ID, Nom, Email, Telephone) VALUES
('0', 'default', '', '');

-- Inserting the null log
INSERT INTO Log (ID, ID_client,Nom) VALUES
('0', 0,'default');

-- Insertion de données de test dans la table Type de piece
INSERT INTO type_piece (ID, Nom, Description) VALUES
('1', 'Ancrage 4X4', 'Ancrage 4X4'),
('2', 'Ancrage 10X8', 'Ancrage 10X8'),
('3', 'Ancrage 10X12', 'Ancrage 10X12');