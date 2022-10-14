import numpy as np
import cv2
from shapely.geometry import Polygon


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DetectionIoUEvaluator:
    def __init__(self, is_output_polygon=False, iou_constraint=0.5, area_precision_constraint=0.5):
        self.is_output_polygon = is_output_polygon
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        ## init
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        # metrics
        recall = 0
        precision = 0
        hmean = 0

        # num of 'pred' & 'gt' matching
        detMatch = 0
        pairs = []
        iouMat = np.empty([1, 1])

        # num of cared polys
        num_gtCare = 0
        num_detCare = 0

        # list of polys
        gtPolys = []
        detPolys = []

        # idx of dontcare polys
        gtPolys_dontcare = []
        detPolys_dontcare = []

        # log string
        evaluationLog = ""

        ## gt
        for i in range(len(gt)):
            poly = gt[i]['polys']
            dontcare = gt[i]['dontcare']
            if not Polygon(poly).is_valid or not Polygon(poly).is_simple:
                continue
            gtPolys.append(poly)
            if dontcare:
                gtPolys_dontcare.append(len(gtPolys) - 1)

        evaluationLog += f"GT polygons: {str(len(gtPolys))}" + \
        (f" ({len(gtPolys_dontcare)} don't care)\n" if len(gtPolys_dontcare) > 0 else "\n")

        ## pred
        for i in range(len(pred)):
            poly = pred[i]
            if not Polygon(poly).is_valid or not Polygon(poly).is_simple:
                continue
            detPolys.append(poly)

            # For dontcare gt
            if len(gtPolys_dontcare) > 0:
                for idx in gtPolys_dontcare:
                    dontcare_poly = gtPolys[idx]
                    intersected_area = get_intersection(dontcare_poly, poly)
                    poly_area = Polygon(poly).area
                    precision = 0 if poly_area == 0 else intersected_area / poly_area
                    # If precision enough, append as dontcare det.
                    if (precision > self.area_precision_constraint):
                        detPolys_dontcare.append(len(detPolys) - 1)
                        break

        evaluationLog += f"DET polygons: {len(detPolys)}" + \
        (f" ({len(detPolys_dontcare)} don't care)\n" if len(detPolys_dontcare) > 0 else "\n")

        ## calc
        if len(gtPolys) > 0 and len(detPolys) > 0:
            # visit arrays
            iouMat = np.empty([len(gtPolys), len(detPolys)])
            gtRectMat = np.zeros(len(gtPolys), np.int8)
            detRectMat = np.zeros(len(detPolys), np.int8)

            # IoU
            if self.is_output_polygon:
                # polygon
                for gt_idx in range(len(gtPolys)):
                    for det_idx in range(len(detPolys)):
                        pG = gtPolys[gt_idx]
                        pD = detPolys[det_idx]
                        iouMat[gt_idx, det_idx] = get_intersection_over_union(pD, pG)
            else:
                # rectangle
                for gt_idx in range(len(gtPolys)):
                    for det_idx in range(len(detPolys)):
                        pG = np.float32(gtPolys[gt_idx])
                        pD = np.float32(detPolys[det_idx])
                        iouMat[gt_idx, det_idx] = self.iou_rotate(pD, pG)

            for gt_idx in range(len(gtPolys)):
                for det_idx in range(len(detPolys)):
                    # If IoU == 0 and care
                    if gtRectMat[gt_idx] == 0 and detRectMat[det_idx] == 0 \
                    and (gt_idx not in gtPolys_dontcare) and (det_idx not in detPolys_dontcare):
                        # If IoU enough
                        if iouMat[gt_idx, det_idx] > self.iou_constraint:
                            # Mark the visit arrays
                            gtRectMat[gt_idx] = 1
                            detRectMat[det_idx] = 1
                            detMatch += 1
                            pairs.append({'gt': gt_idx, 'det': det_idx})
                            evaluationLog += f"Match GT #{gt_idx} with Det #{det_idx}\n"

        ## summary
        num_gtCare += (len(gtPolys) - len(gtPolys_dontcare))
        num_detCare += (len(detPolys) - len(detPolys_dontcare))

        if num_gtCare == 0:
            recall = 1.0
            precision = 0.0 if num_detCare > 0 else 1.0
        else:
            recall = float(detMatch) / num_gtCare
            precision = 0 if num_detCare == 0 else float(
                detMatch) / num_detCare
        hmean = 0 if (precision + recall) == 0 else \
                2.0 * precision * recall / (precision + recall)

        metric = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPolys) > 100 else iouMat.tolist(),
            'gtPolys': gtPolys,
            'detPolys': detPolys,
            'gtCareNum': num_gtCare,
            'detCareNum': num_detCare,
            'gtDontCare': gtPolys_dontcare,
            'detDontCare': detPolys_dontcare,
            'detMatched': detMatch,
            'evaluationLog': evaluationLog
        }
        return metric

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCareNum']
            numGlobalCareDet += result['detCareNum']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                    methodRecall + methodPrecision)

        methodMetrics = {'precision': methodPrecision,
                         'recall': methodRecall, 'hmean': methodHmean}

        return methodMetrics

    @staticmethod
    def iou_rotate(box_a, box_b, method='union'):
        rect_a = cv2.minAreaRect(box_a)
        rect_b = cv2.minAreaRect(box_b)
        r1 = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        if r1[0] == 0:
            return 0
        else:
            inter_area = cv2.contourArea(r1[1])
            area_a = cv2.contourArea(box_a)
            area_b = cv2.contourArea(box_b)
            union_area = area_a + area_b - inter_area
            if union_area == 0 or inter_area == 0:
                return 0
            if method == 'union':
                iou = inter_area / union_area
            elif method == 'intersection':
                iou = inter_area / min(area_a, area_b)
            else:
                raise NotImplementedError
            return iou


class QuadMetric:
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator(is_output_polygon=is_output_polygon)

    def measure(self, batch, output, box_thresh=0.7):
        '''
        batch: (image, polygons, ignore_tags)
            image: tensor of shape (N, C, H, W).
            polys: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            dontcare: tensor of shape (N, K), indicates whether a region is ignorable or not.
        output: (polygons, ...)
        '''
        gt_polys = batch['polys'].asnumpy().astype(np.float32)
        gt_dontcare = batch['dontcare'].asnumpy()
        pred_polys = np.array(output[0])
        pred_scores = np.array(output[1])

        # Loop i for every batch
        for i in range(len(gt_polys)):
            gt = [{'polys': gt_polys[i][j], 'dontcare': gt_dontcare[i][j]}
                  for j in range(len(gt_polys[i]))]
            if self.is_output_polygon:
                pred = [pred_polys[i][j] for j in range(len(pred_polys[i]))]
            else:
                pred = [pred_polys[i][j, :, :].astype(np.int32)
                        for j in range(pred_polys[i].shape[0]) if pred_scores[i][j] >= box_thresh]
        return self.evaluator.evaluate_image(gt, pred)


    def validate_measure(self, batch, output):
        return self.measure(batch, output, box_thresh=0.55)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics for image_metrics in raw_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }