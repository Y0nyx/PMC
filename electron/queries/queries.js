const { Client } = require("pg");
const path = require("path");
let appPath = global.appPath;
const config = require(path.join(appPath, "configElectron.js"));

// Create a new PostgreSQL client

async function generateDatabase(sql) {
  await makeQuery(sql);
}

async function deletePiece(selected) {
  let id;
  if (selected.length > 1) id = selected.map((item) => `'${item}'`).join(",");
  else id = `'${selected[0]}'`;
  let table = "piece";
  let query = `DELETE FROM ${table} WHERE id IN (${id});`;

  console.log(query);
  await makeQuery(query);
}

async function getAllimages() {
  let table = "piece";
  let query = `SELECT photo,boundingbox from ${table};`;

  let result = await makeQuery(query);

  return result;
}

async function createPiece(piece) {
  let table = "piece";
  let query = `INSERT INTO ${table} (id,date,photo,boundingbox,resultat,id_client,id_log,id_type_piece,id_erreur_soudure) VALUES ('${piece.id}','${piece.date}','${piece.url}','${piece.boundingbox}','${piece.resultat}','${piece.id_client}','${piece.id_log}','${piece.id_type_piece}','${piece.id_erreur_soudure}');`;
  await makeQuery(query);
}

async function fetchPieces(id_client, id_log) {
  let table = "piece";
  let table2 = "erreur_soudure";
  let table3 = "type_piece";
  let table4 = "client";
  let table5 = "log";

  let query = `SELECT 
  ${table}.id as id,
  ${table}.date as date,
  ${table}.photo as photo,
  ${table}.boundingbox as boundingbox,
  ${table}.resultat as resultat,
  ${table}.id_erreur_soudure,
  ${table2}.nom as nom_erreur_soudure,
  ${table2}.description as description_erreur_soudure, 
  ${table3}.nom as nom_type_piece,
  ${table3}.description as description_type_piece,
  ${table4}.nom as nom_client,
  ${table5}.id as id_log 
  FROM ${table} JOIN ${table2} ON ${table}.id_erreur_soudure = ${table2}.id 
  JOIN ${table3} ON ${table}.id_type_piece = ${table3}.id
  JOIN ${table4} ON ${table}.id_client = ${table4}.id 
  JOIN ${table5} ON ${table}.id_log = ${table5}.id  
  WHERE ${table}.id_client = '${id_client}' AND ${table}.id_log = '${id_log}';`;

  let result = await makeQuery(query);

  return result;
}

async function fetchPiece(id) {
  let table = "piece";
  let table2 = "erreur_soudure";
  let table3 = "type_piece";
  let table4 = "client";
  let table5 = "log";

  let query = `SELECT 
  ${table}.id as id,
  ${table}.date as date,
  ${table}.photo as photo,
  ${table}.boundingbox as boundingbox,
  ${table}.resultat as resultat,
  ${table}.id_erreur_soudure,
  ${table2}.nom as nom_erreur_soudure,
  ${table2}.description as description_erreur_soudure, 
  ${table3}.nom as nom_type_piece,
  ${table3}.description as description_type_piece,
  ${table4}.nom as nom_client,
  ${table5}.id as id_log 
  FROM ${table} 
  JOIN ${table2} ON ${table}.id_erreur_soudure = ${table2}.id 
  JOIN ${table3} ON ${table}.id_type_piece = ${table3}.id 
  JOIN ${table4} ON ${table}.id_client = ${table4}.id 
  JOIN ${table5} ON ${table}.id_log = ${table5}.id   
  WHERE ${table}.id = '${id}';`;

  let result = await makeQuery(query);

  return result[0];
}

async function fetchClients() {
  let table = "client";

  let query = `SELECT 
  *
  FROM ${table}`;

  let result = await makeQuery(query);

  return result;
}

async function fetchLogs(client_id) {
  let table = "log";
  let query = `SELECT 
  *
  FROM ${table} 
  WHERE ${table}.id_client = '${client_id}'`;

  let result = await makeQuery(query);

  return result;
}

async function fetchTypesPiece() {
  let table = "type_piece";
  let query = `SELECT 
  *
  FROM ${table} `;

  let result = await makeQuery(query);

  return result;
}

async function createClient(client) {
  let table = "client";
  let query = `INSERT INTO ${table} (id,nom,telephone,email) VALUES ('${client.id}','${client.nom}','${client.telephone}','${client.email}');`;
  await makeQuery(query);
}

async function createLog(log) {
  let table = "log";
  let query = `INSERT INTO ${table} (id,id_client,nom) VALUES ('${log.id}','${log.id_client}','${log.nom}');`;

  await makeQuery(query);
}

async function makeQuery(query) {
  try {
    const client = new Client({
      user: config.DB_USER,
      host: config.DB_HOST,
      database: config.DB_DATABASE,
      password: config.DB_PASSWORD,
      port: config.DB_PORT,
    });

    await client.connect();
    const result = await client.query(query);

    // Process the result
    const rows = result.rows;
    await client.end();
    return rows;
  } catch (error) {
    console.error("Error executing query", error);
    throw error; // Propagate the error to the caller
  }
}

async function resetData() {
  let query = `TRUNCATE TABLE erreur_Soudure,client,log,piece,type_piece`;
  await makeQuery(query);
}

module.exports = {
  generateDatabase,
  fetchPieces,
  fetchPiece,
  deletePiece,
  getAllimages,
  fetchClients,
  fetchLogs,
  fetchTypesPiece,
  createClient,
  createPiece,
  createLog,
  resetData,
};
