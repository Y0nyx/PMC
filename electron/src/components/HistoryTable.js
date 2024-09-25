import * as React from "react";
import PropTypes from "prop-types";
import { alpha } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TablePagination from "@mui/material/TablePagination";
import TableRow from "@mui/material/TableRow";
import TableSortLabel from "@mui/material/TableSortLabel";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";
import Checkbox from "@mui/material/Checkbox";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";
import DeleteIcon from "@mui/icons-material/Delete";
import FilterListIcon from "@mui/icons-material/FilterList";
import { visuallyHidden } from "@mui/utils";
import { useNavigate } from "react-router-dom";
function descendingComparator(a, b, orderBy) {
  if (b[orderBy] < a[orderBy]) {
    return -1;
  }
  if (b[orderBy] > a[orderBy]) {
    return 1;
  }
  return 0;
}

function getComparator(order, orderBy) {
  return order === "desc"
    ? (a, b) => descendingComparator(a, b, orderBy)
    : (a, b) => -descendingComparator(a, b, orderBy);
}

// Since 2020 all major browsers ensure sort stability with Array.prototype.sort().
// stableSort() brings sort stability to non-modern browsers (notably IE11). If you
// only support modern browsers you can replace stableSort(exampleArray, exampleComparator)
// with exampleArray.slice().sort(exampleComparator)
function stableSort(array, comparator) {
  const stabilizedThis = array.map((el, index) => [el, index]);
  stabilizedThis.sort((a, b) => {
    const order = comparator(a[0], b[0]);
    if (order !== 0) {
      return order;
    }
    return a[1] - b[1];
  });
  return stabilizedThis.map((el) => el[0]);
}

const headCells = [
  {
    id: "id",
    numeric: true,
    disablePadding: false,
    label: "ID",
  },
  {
    id: "photo",
    numeric: false,
    disablePadding: true,
    label: "Photo",
  },
  {
    id: "resultat",
    numeric: false,
    disablePadding: false,
    label: "resultat",
  },
  {
    id: "erreur",
    numeric: false,
    disablePadding: false,
    label: "Erreur",
  },
  {
    id: "date",
    numeric: false,
    disablePadding: false,
    label: "Date",
  },
  {
    id: "heure",
    numeric: false,
    disablePadding: false,
    label: "Heure",
  },
  {
    id: "info",
    numeric: false,
    disablePadding: false,
    label: "info",
  },
];

function EnhancedTableHead(props) {
  const {
    onSelectAllClick,
    order,
    orderBy,
    numSelected,
    rowCount,
    onRequestSort,
    onDelete,
  } = props;
  const createSortHandler = (property) => (event) => {
    onRequestSort(event, property);
  };

  return (
    <TableHead>
      <TableRow>
        <TableCell padding="checkbox">
          <Checkbox
            color="primary"
            indeterminate={numSelected > 0 && numSelected < rowCount}
            checked={rowCount > 0 && numSelected === rowCount}
            onChange={onSelectAllClick}
            inputProps={{
              "aria-label": "select all desserts",
            }}
          />
        </TableCell>
        {headCells.map((headCell) => (
          <TableCell
            className="font-extrabold text-lg font-normal"
            key={headCell.id}
            align="left"
            padding={headCell.disablePadding ? "none" : "normal"}
            sortDirection={orderBy === headCell.id ? order : false}
          >
            <TableSortLabel
              active={orderBy === headCell.id}
              direction={orderBy === headCell.id ? order : "asc"}
              onClick={createSortHandler(headCell.id)}
            >
              {headCell.label}
              {orderBy === headCell.id ? (
                <Box component="span" sx={visuallyHidden}>
                  {order === "desc" ? "sorted descending" : "sorted ascending"}
                </Box>
              ) : null}
            </TableSortLabel>
          </TableCell>
        ))}
      </TableRow>
    </TableHead>
  );
}

EnhancedTableHead.propTypes = {
  numSelected: PropTypes.number.isRequired,
  onRequestSort: PropTypes.func.isRequired,
  onSelectAllClick: PropTypes.func.isRequired,
  order: PropTypes.oneOf(["asc", "desc"]).isRequired,
  orderBy: PropTypes.string.isRequired,
  rowCount: PropTypes.number.isRequired,
};

function EnhancedTableToolbar(props) {
  const { numSelected, onDelete } = props;

  return (
    <Toolbar
      sx={{
        pl: { sm: 2 },
        pr: { xs: 1, sm: 1 },
        ...(numSelected > 0 && {
          bgcolor: (theme) =>
            alpha(
              theme.palette.primary.main,
              theme.palette.action.activatedOpacity
            ),
        }),
      }}
    >
      {numSelected > 0 ? (
        <Typography
          sx={{ flex: "1 1 100%" }}
          color="inherit"
          variant="subtitle1"
          component="div"
          className="font-bold text-xl font-normal"
        >
          {numSelected} selected
        </Typography>
      ) : (
        <Typography
          sx={{ flex: "1 1 100%" }}
          variant="h6"
          id="tableTitle"
          component="div"
          className="font-bold text-xl font-normal"
        >
          Historique des pièces
        </Typography>
      )}

      {numSelected > 0 ? (
        <Tooltip title="Delete">
          <IconButton>
            <DeleteIcon onClick={() => onDelete()} />
          </IconButton>
        </Tooltip>
      ) : (
        <Tooltip title="Filter list">
          <IconButton>
            <FilterListIcon />
          </IconButton>
        </Tooltip>
      )}
    </Toolbar>
  );
}

EnhancedTableToolbar.propTypes = {
  numSelected: PropTypes.number.isRequired,
};

export default function EnhancedTable({ rows_ }) {
  const [rows, setRows] = React.useState(rows_);
  const [order, setOrder] = React.useState("asc");
  const [orderBy, setOrderBy] = React.useState("id");
  const [selected, setSelected] = React.useState([]);
  const [page, setPage] = React.useState(0);
  const [dense, setDense] = React.useState(false);
  const [rowsPerPage, setRowsPerPage] = React.useState(5);

  const navigate = useNavigate();
  const ipcRenderer = window.require("electron").ipcRenderer;

  const handleRequestSort = (event, property) => {
    const isAsc = orderBy === property && order === "asc";
    setOrder(isAsc ? "desc" : "asc");
    setOrderBy(property);
  };
  const [visibleRows, setVisibleRows] = React.useState([]);
  const [emptyRows, setEmptyRows] = React.useState([]);

  React.useEffect(() => {
    // Avoid a layout jump when reaching the last page with empty rows.

    setEmptyRows(
      page > 0 ? Math.max(0, (1 + page) * rowsPerPage - rows.length) : 0
    );
    setVisibleRows(
      stableSort(rows, getComparator(order, orderBy)).slice(
        page * rowsPerPage,
        page * rowsPerPage + rowsPerPage
      )
    );
  }, [rows, order, orderBy, page, rowsPerPage]);
  const handleSelectAllClick = (event) => {
    if (event.target.checked) {
      const newSelected = rows.map((n) => n.id);
      setSelected(newSelected);
      console.log(newSelected);
      return;
    }
    setSelected([]);
  };

  const handleClick = (event, id) => {
    console.log(rows);
    const selectedIndex = selected.indexOf(id);
    let newSelected = [];

    if (selectedIndex === -1) {
      newSelected = newSelected.concat(selected, id);
    } else if (selectedIndex === 0) {
      newSelected = newSelected.concat(selected.slice(1));
    } else if (selectedIndex === selected.length - 1) {
      newSelected = newSelected.concat(selected.slice(0, -1));
    } else if (selectedIndex > 0) {
      newSelected = newSelected.concat(
        selected.slice(0, selectedIndex),
        selected.slice(selectedIndex + 1)
      );
    }

    setSelected(newSelected);
  };

  const navigatePiece = (event, id) => {
    navigate("/piece/" + id);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleChangeDense = (event) => {
    setDense(event.target.checked);
  };

  const handleDelete = () => {
    setRows(rows.filter((element) => !selected.includes(element.id)));
    ipcRenderer.send("deletePiece", [selected]);
    setSelected([]);
  };
  const isSelected = (name) => selected.indexOf(name) !== -1;

  return (
    <Box sx={{ width: "90%" }}>
      <Paper sx={{ width: "100%", mb: 2 }}>
        <EnhancedTableToolbar
          numSelected={selected.length}
          onDelete={handleDelete}
        />
        <TableContainer>
          <Table
            stickyHeader
            aria-labelledby="tableTitle"
            size={dense ? "small" : "medium"}
          >
            <EnhancedTableHead
              numSelected={selected.length}
              order={order}
              orderBy={orderBy}
              onSelectAllClick={handleSelectAllClick}
              onRequestSort={handleRequestSort}
              rowCount={rows.length}
            />
            <TableBody className="bg-gray-100">
              {visibleRows.map((row, index) => {
                const isItemSelected = isSelected(row.id);
                const labelId = `enhanced-table-checkbox-${index}`;
                let imageIndex;
                if (row.result == "succès") index = 0;
                else {
                  imageIndex = row.images.findIndex(
                    (image) => image.boundingBox != undefined
                  );
                }

                return (
                  <TableRow
                    hover
                    role="checkbox"
                    aria-checked={isItemSelected}
                    tabIndex={-1}
                    key={row.id}
                    selected={isItemSelected}
                    sx={{ cursor: "pointer" }}
                    className="h-32"
                  >
                    <TableCell
                      className=""
                      onClick={(event) => handleClick(event, row.id)}
                      padding="checkbox"
                    >
                      <Checkbox
                        color="primary"
                        checked={isItemSelected}
                        inputProps={{
                          "aria-labelledby": labelId,
                        }}
                      />
                    </TableCell>
                    <TableCell
                      className="font-bold font-normal text-lg"
                      onClick={(event) => handleClick(event, row.id)}
                      component="th"
                      id={labelId}
                      scope="row"
                      padding="none"
                    >
                      {row.id}
                    </TableCell>
                    <TableCell
                      onClick={(event) => handleClick(event, row.id)}
                      component="th"
                      id={labelId}
                      scope="row"
                      className="py-2"
                    >
                      {row.result == "succès" ? (
                        <img
                          className="w-48 h-48 rounded-lg"
                          src={row.images[0].url}
                        ></img>
                      ) : (
                        <div className="relative w-48 h-48 rounded-lg">
                          <img
                            src={row.images[imageIndex].url}
                            className="w-full h-full rounded-lg"
                          />

                          {row.images[imageIndex].boundingBox &&
                            row.images[imageIndex].boundingBox.box.map(
                              (box) => {
                                return (
                                  <div
                                    style={{
                                      top: `${box.yCenter * 100}%`,
                                      left: `${box.xCenter * 100}%`,
                                      width: `${box.width * 100}%`,
                                      height: `${box.height * 100}%`,
                                    }}
                                    className="absolute  bg-opacity-75 border-4 border-solid border-red-600 rounded"
                                  ></div>
                                );
                              }
                            )}
                        </div>
                      )}
                    </TableCell>
                    <TableCell
                      className="font-bold  font-normal text-lg"
                      onClick={(event) => handleClick(event, row.id)}
                      align="left"
                    >
                      {row.result}
                    </TableCell>
                    <TableCell
                      className="font-bold  font-normal text-lg"
                      onClick={(event) => handleClick(event, row.id)}
                      align="left"
                    >
                      {row.errorType}
                    </TableCell>
                    <TableCell
                      className="font-bold  font-normal text-lg"
                      onClick={(event) => handleClick(event, row.id)}
                      align="left"
                    >
                      {row.date}
                    </TableCell>
                    <TableCell
                      className="font-bold  font-normal text-lg"
                      onClick={(event) => handleClick(event, row.id)}
                      align="left"
                    >
                      {row.hour}
                    </TableCell>
                    <TableCell align="left" className="">
                      {
                        <div
                          on
                          onClick={(event) => navigatePiece(event, row.id)}
                          className=" text-lg font-normal font-bold hover:scale-110 hover:text-black hover:bg-white rounded-lg  m-1  border-2 border-black bg-black text-white h-20 w-32 flex justify-center items-center"
                        >
                          Voir
                        </div>
                      }
                    </TableCell>
                  </TableRow>
                );
              })}
              <TableRow
                style={{
                  height: (dense ? 33 : 53) * emptyRows,
                }}
              >
                <TableCell colSpan={6} />
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={rows.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>
      <FormControlLabel
        control={<Switch checked={dense} onChange={handleChangeDense} />}
        label="Dense padding"
      />
    </Box>
  );
}
