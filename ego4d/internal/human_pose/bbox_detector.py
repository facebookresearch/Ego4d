from mmdet.apis import inference_detector, init_detector

from mmpose.apis import process_mmdet_results


##------------------------------------------------------------------------------------
class DetectorModel:
    def __init__(
        self, detector_config=None, detector_checkpoint=None, min_bbox_score=0.7
    ):
        self.detector_config = detector_config
        self.detector_checkpoint = detector_checkpoint
        self.detector = init_detector(
            self.detector_config, self.detector_checkpoint, device="cuda:0".lower()
        )
        self.min_bbox_score = min_bbox_score

    ## iou_threshold: the threshold to decide whether to use the offshelf bbox or not
    def get_bboxes(self, image_name, bboxes, iou_threshold=0.3):
        det_results = inference_detector(self.detector, image_name)
        person_results = process_mmdet_results(
            det_results, 1
        )  # keep the person class bounding boxes.
        person_results = [
            bbox for bbox in person_results if bbox["bbox"][4] > self.min_bbox_score
        ]

        refined_bboxes = bboxes.copy()
        is_offshelf_valid = [True] * len(person_results)

        ## go through over the aria bboxes
        for i, bbox in enumerate(refined_bboxes):
            max_iou = 0
            max_iou_offshelf_bbox = None
            max_iou_index = -1

            for j, offshelf_bbox in enumerate(person_results):
                if is_offshelf_valid[j] == True:
                    iou = self.bb_intersection_over_union(
                        boxA=bbox["bbox"], boxB=offshelf_bbox["bbox"]
                    )

                    if iou > max_iou:
                        max_iou = iou
                        max_iou_offshelf_bbox = offshelf_bbox["bbox"]
                        max_iou_index = j
            if max_iou > iou_threshold:
                refined_bboxes[i]["bbox"] = max_iou_offshelf_bbox
                is_offshelf_valid[max_iou_index] = False

        return refined_bboxes

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
