from typing import Optional, List, Tuple

class CsvResultRow:
    def __init__(
        self, 
        id: Optional[int] = None, 
        un_sup_model_ref: Optional[str] = None, 
        seg_model_ref: Optional[str] = None, 
        sup_model_ref: Optional[str] = None,
        capture_img_path: Optional[str] = None, 
        seg_results: Optional[str] = None, 
        seg_threshold: Optional[float] = None, 
        seg_img_path: Optional[str] = None,
        sub_img_path: Optional[str] = None, 
        unsup_pred_img_path: Optional[str] = None, 
        unsup_defect_threshold: Optional[float] = None,
        unsup_threshold_algo: Optional[str] = None, 
        unsup_defect_res: Optional[str] = None, 
        unsup_defect_bb: Optional[str] = None,
        sup_defect_res: Optional[str] = None, 
        sup_defect_threshold: Optional[float] = None, 
        sup_defect_bb: Optional[str] = None,
        manual_verification_result: Optional[str] = None, 
        date: Optional[str] = None, 
        time: Optional[str] = None
    ) -> None:
        self.id = id
        self.un_sup_model_ref = un_sup_model_ref
        self.seg_model_ref = seg_model_ref
        self.sup_model_ref = sup_model_ref
        self.capture_img_path = capture_img_path
        self.seg_results = seg_results
        self.seg_threshold = seg_threshold
        self.seg_img_path = seg_img_path
        self.sub_img_path = sub_img_path
        self.unsup_pred_img_path = unsup_pred_img_path
        self.unsup_defect_threshold = unsup_defect_threshold
        self.unsup_threshold_algo = unsup_threshold_algo
        self.unsup_defect_res = unsup_defect_res
        self.unsup_defect_bb = unsup_defect_bb
        self.sup_defect_res = sup_defect_res
        self.sup_defect_threshold = sup_defect_threshold
        self.sup_defect_bb = sup_defect_bb
        self.manual_verification_result = manual_verification_result
        self.date = date
        self.time = time

    def check_completion(self) -> Tuple[bool, Optional[List[str]]]:
        required_fields = [
            'id', 'un_sup_model_ref', 'seg_model_ref', 'sup_model_ref',
            'capture_img_path', 'seg_results', 'seg_threshold', 'seg_img_path',
            'sub_img_path', 'unsup_pred_img_path', 'unsup_defect_threshold',
            'unsup_threshold_algo', 'unsup_defect_res', 'unsup_defect_bb',
            'sup_defect_res', 'sup_defect_threshold', 'sup_defect_bb',
            'manual_verification_result', 'date', 'time'
        ]

        missing_fields = [field for field in required_fields if getattr(self, field) is None]
        return (len(missing_fields) == 0, missing_fields if missing_fields else None)

    def set_id(self, id: int) -> None:
        self.id = id

    def get_writable_row(self) -> List[Optional[object]]:
        return [
            self.id,
            self.un_sup_model_ref,
            self.seg_model_ref,
            self.sup_model_ref,
            self.capture_img_path,
            self.seg_results,
            self.seg_threshold,
            self.seg_img_path,
            self.sub_img_path,
            self.unsup_pred_img_path,
            self.unsup_defect_threshold,
            self.unsup_threshold_algo,
            self.unsup_defect_res,
            self.unsup_defect_bb,
            self.sup_defect_res,
            self.sup_defect_threshold,
            self.sup_defect_bb,
            self.manual_verification_result,
            self.date,
            self.time
        ]
