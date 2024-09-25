-- Insertion de données de test dans la table Erreur_Soudure
INSERT INTO Erreur_Soudure (ID, Nom, Description) VALUES
('1', 'Error1', 'Description 1'),
('2', 'Error2', 'Description 2'),
('3', 'Error3', 'Description 3');

-- Insertion de données de test dans la table Type de piece
INSERT INTO type_piece (ID, Nom, Description) VALUES
('1', 'Piece 1', 'Description 1'),
('2', 'Piece 2', 'Description 2'),
('3', 'Piece 3', 'Description 3');

-- Insertion de données de test dans la table Client
INSERT INTO Client (ID, Nom, Email, Telephone) VALUES
('1', 'Client1', 'client1@example.com', '1234567890'),
('2', 'Client2', 'client2@example.com', '9876543210'),
('3', 'Client3', 'client3@example.com', '5555555555');

-- Insertion de données de test dans la table Log
INSERT INTO Log (ID, ID_client,Nom) VALUES
('1', '1','log1'),
('2', '2','log2'),
('3', '3','log3');

-- Insertion de données de test dans la table pièce
INSERT INTO piece (ID, Date, Photo,BoundingBox, Resultat, ID_client, ID_log, ID_type_piece, ID_erreur_Soudure) VALUES
('1', '2024-02-08 11:15:00', '/imagesSoudure/piece1','/boundingBox/piece1', '1', '1', '1', '1', '1'),
('2', '2023-10-28 12:30:00', '/imagesSoudure/piece1','/boundingBox/piece1' ,'0', '1', '1', '1', '0'),
('3', '2024-07-21 08:30:00', '/imagesSoudure/piece1','/boundingBox/piece1', '1', '2', '1', '3', '3');
