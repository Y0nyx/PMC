class CsvResultRow:
    def __init__(self, id=None, un_sup_model_ref=None, seg_model_ref=None, sup_model_ref=None,
                 capture_img_path=None, seg_results=None, seg_threshold=None, seg_img_path=None,
                 sub_img_path=None, unsup_pred_img_path=None, unsup_defect_threshold=None,
                 unsup_threshold_algo=None, unsup_defect_res=None, unsup_defect_bb=None,
                 sup_defect_res=None, sup_defect_threshold=None, sup_defect_bb=None,
                 manual_verification_result=None, date=None, time=None):
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

    def check_completion(self):
        required_fields = ['id', 'un_sup_model_ref', 'seg_model_ref', 'sup_model_ref',
                           'capture_img_path', 'seg_results', 'seg_threshold', 'seg_img_path',
                           'sub_img_path', 'unsup_pred_img_path', 'unsup_defect_threshold',
                           'unsup_threshold_algo', 'unsup_defect_res', 'unsup_defect_bb',
                           'sup_defect_res', 'sup_defect_threshold', 'sup_defect_bb',
                           'manual_verification_result', 'date', 'time']

        missing_fields = [field for field in required_fields if getattr(self, field) is None]
        if missing_fields:
            return False, missing_fields
        else:
            return True, None
    
    def set_id(self, id): self.id = id

    def get_writable_row(self):
        # Assuming the order of fields in your CSV matches the order of attributes in this list
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

