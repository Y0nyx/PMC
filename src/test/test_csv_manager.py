from . import ajuste_import
ajuste_import()

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pipeline.CsvManager import CsvManager
from pipeline.CsvResultRow import CsvResultRow
from common.constants import CSV_FILE_NAME, CSV_FIELDS

@pytest.fixture
def csv_manager():
    return CsvManager()

@pytest.mark.short
def test_add_new_row(csv_manager):
    row = MagicMock(spec=CsvResultRow)
    row.get_writable_row.return_value = [1, 'model_ref', 'seg_model', 'sup_model', 'path', 'results', 'threshold', 'seg_path', 'sub_path', 'unsup_pred_path', 'threshold', 'algo', 'defect_res', 'defect_bb', 'sup_res', 'sup_threshold', 'sup_bb', 'manual_verif', 'date', 'time']
    row.set_id = MagicMock()
    
    with patch('builtins.open', mock_open(read_data="id,un_sup_model_ref,seg_model_ref,sup_model_ref,capture_img_path,seg_results,seg_threshold,seg_img_path,sub_img_path,unsup_pred_img_path,unsup_defect_threshold,unsup_threshold_algo,unsup_defect_res,unsup_defect_bb,sup_defect_res,sup_defect_threshold,sup_defect_bb,manual_verification_result,date,time\n")) as mock_file:
        csv_manager.add_new_row(row)
        mock_file.assert_called_with(CSV_FILE_NAME, 'a', newline='')
        row.set_id.assert_called_once_with(1)

@pytest.mark.short
def test_check_headers_creation(csv_manager):
    with patch('os.path.isfile', return_value=False), patch('builtins.open', mock_open()) as mock_file:
        with patch('builtins.print') as mock_print:
            csv_manager.check_headers()
            mock_file.assert_called_once_with(CSV_FILE_NAME, 'w', newline='')
            mock_print.assert_called_with(f"{CSV_FILE_NAME} has been created with headers.")
