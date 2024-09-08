from . import ajuste_import
ajuste_import()

import pytest
from pipeline.CsvResultRow import CsvResultRow

@pytest.mark.short
def test_check_completion():
    # Test où tous les champs sont remplis
    row = CsvResultRow(
        id=1, un_sup_model_ref='model1', seg_model_ref='model2', sup_model_ref='model3',
        capture_img_path='/path/to/image.jpg', seg_results='results', seg_threshold=0.5,
        seg_img_path='/path/to/seg_img.jpg', sub_img_path='/path/to/sub_img.jpg',
        unsup_pred_img_path='/path/to/unsup_pred_img.jpg', unsup_defect_threshold=0.1,
        unsup_threshold_algo='algo', unsup_defect_res='defect_res', unsup_defect_bb='bb',
        sup_defect_res='sup_defect_res', sup_defect_threshold=0.2, sup_defect_bb='sup_defect_bb',
        manual_verification_result='result', date='2024-09-07', time='12:00:00'
    )
    is_complete, missing_fields = row.check_completion()
    assert is_complete
    assert missing_fields is None

    # Test où certains champs sont manquants
    row = CsvResultRow(id=1)
    is_complete, missing_fields = row.check_completion()
    assert not is_complete
    assert 'un_sup_model_ref' in missing_fields

@pytest.mark.short
def test_check_completion():
    # Test où tous les champs sont remplis
    row = CsvResultRow(
        id=1, un_sup_model_ref='model1', seg_model_ref='model2', sup_model_ref='model3',
        capture_img_path='/path/to/image.jpg', seg_results='results', seg_threshold=0.5,
        seg_img_path='/path/to/seg_img.jpg', sub_img_path='/path/to/sub_img.jpg',
        unsup_pred_img_path='/path/to/unsup_pred_img.jpg', unsup_defect_threshold=0.1,
        unsup_threshold_algo='algo', unsup_defect_res='defect_res', unsup_defect_bb='bb',
        sup_defect_res='sup_defect_res', sup_defect_threshold=0.2, sup_defect_bb='sup_defect_bb',
        manual_verification_result='result', date='2024-09-07', time='12:00:00'
    )
    is_complete, missing_fields = row.check_completion()
    assert is_complete
    assert missing_fields is None

    # Test où certains champs sont manquants
    row.set_id(2)
    assert row.id == 2

@pytest.mark.short
def test_get_writable_row():
    row = CsvResultRow(
        id=1, un_sup_model_ref='model1', seg_model_ref='model2', sup_model_ref='model3',
        capture_img_path='/path/to/image.jpg', seg_results='results', seg_threshold=0.5,
        seg_img_path='/path/to/seg_img.jpg', sub_img_path='/path/to/sub_img.jpg',
        unsup_pred_img_path='/path/to/unsup_pred_img.jpg', unsup_defect_threshold=0.1,
        unsup_threshold_algo='algo', unsup_defect_res='defect_res', unsup_defect_bb='bb',
        sup_defect_res='sup_defect_res', sup_defect_threshold=0.2, sup_defect_bb='sup_defect_bb',
        manual_verification_result='result', date='2024-09-07', time='12:00:00'
    )
    row_data = row.get_writable_row()
    expected_data = [
        1, 'model1', 'model2', 'model3', '/path/to/image.jpg', 'results', 0.5,
        '/path/to/seg_img.jpg', '/path/to/sub_img.jpg', '/path/to/unsup_pred_img.jpg', 0.1,
        'algo', 'defect_res', 'bb', 'sup_defect_res', 0.2, 'sup_defect_bb', 'result', '2024-09-07', '12:00:00'
    ]
    assert row_data == expected_data
