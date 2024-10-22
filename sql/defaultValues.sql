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
('1', 'Piece 1', 'Description 1'),
('2', 'Piece 2', 'Description 2'),
('3', 'Piece 3', 'Description 3');