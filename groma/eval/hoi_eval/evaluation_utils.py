"""Evaluation utilities for calculating HOI detection metrics."""


def calculate_single_image_metrics(predictions, gt_hois, image_width, image_height):
    """Calculate evaluation metrics for a single image"""
    if not predictions or not gt_hois:
        return {'matches': [], 'precision': 0.0, 'recall': 0.0, 'total_predictions': len(predictions), 'total_gt': len(gt_hois)}

    matches = []
    used_gt = set()

    # Sort predictions by confidence score (descending)
    sorted_predictions = sorted(enumerate(predictions), key=lambda x: x[1][1], reverse=True)

    for pred_idx, pred in sorted_predictions:
        hoi_id, score, person_x1, person_y1, person_x2, person_y2, object_x1, object_y1, object_x2, object_y2 = pred

        print(f"\n📋 Evaluating Prediction {pred_idx}: HOI {hoi_id}, Score {score:.4f}")
        print(f"  Pred Person bbox: [{person_x1:.1f}, {person_y1:.1f}, {person_x2:.1f}, {person_y2:.1f}]")
        print(f"  Pred Object bbox: [{object_x1:.1f}, {object_y1:.1f}, {object_x2:.1f}, {object_y2:.1f}]")

        best_match = None
        best_iou = 0.0
        best_person_iou = 0.0
        best_object_iou = 0.0
        all_matches = []  # Track all potential matches for debugging

        for gt_idx, gt_hoi in enumerate(gt_hois):
            if gt_idx in used_gt:
                continue

            if gt_hoi['hoi_id'] != hoi_id:
                continue

            # Calculate IoU for both person and object boxes
            person_iou = calculate_bbox_iou(
                [person_x1, person_y1, person_x2, person_y2],
                gt_hoi['person_bbox']
            )
            object_iou = calculate_bbox_iou(
                [object_x1, object_y1, object_x2, object_y2],
                gt_hoi['object_bbox']
            )

            # Debug: Print detailed IoU information
            print(f"  GT {gt_idx}: Person IoU = {person_iou:.4f}, Object IoU = {object_iou:.4f}")
            print(f"    Pred Person bbox: [{person_x1:.1f}, {person_y1:.1f}, {person_x2:.1f}, {person_y2:.1f}]")
            print(f"    GT Person bbox:   {gt_hoi['person_bbox']}")
            print(f"    Pred Object bbox: [{object_x1:.1f}, {object_y1:.1f}, {object_x2:.1f}, {object_y2:.1f}]")
            print(f"    GT Object bbox:   {gt_hoi['object_bbox']}")

            # Use minimum IoU as in official evaluation
            min_iou = min(person_iou, object_iou)

            # Store all match information for debugging
            all_matches.append({
                'gt_idx': gt_idx,
                'person_iou': person_iou,
                'object_iou': object_iou,
                'min_iou': min_iou,
                'used': False
            })

            # Track the best match regardless of threshold (for debugging)
            if min_iou > best_iou:
                best_iou = min_iou
                best_match = gt_idx
                best_person_iou = person_iou
                best_object_iou = object_iou

        # Print summary of all GT comparisons
        print(f"  📊 Summary - Compared against {len(all_matches)} GT annotations:")
        for match_info in all_matches:
            status = "🏆 BEST" if match_info['gt_idx'] == best_match else "   "
            print(f"    {status} GT{match_info['gt_idx']}: Person={match_info['person_iou']:.4f}, Object={match_info['object_iou']:.4f}, Min={match_info['min_iou']:.4f}")

        # Apply threshold check: match is valid only if best IoU >= 0.5
        if best_match is not None and best_iou >= 0.5:
            used_gt.add(best_match)
            matches.append({
                'prediction_idx': pred_idx,
                'gt_idx': best_match,
                'hoi_id': hoi_id,
                'score': score,
                'iou': best_iou,
                'person_iou': best_person_iou,
                'object_iou': best_object_iou,
                'match': True
            })
            print(f"✅ MATCH: Pred {pred_idx} -> GT {best_match}, Min IoU: {best_iou:.4f} (Person: {best_person_iou:.4f}, Object: {best_object_iou:.4f})")
        else:
            # Store the best match info even for failed matches (for debugging)
            matches.append({
                'prediction_idx': pred_idx,
                'gt_idx': best_match,
                'hoi_id': hoi_id,
                'score': score,
                'iou': best_iou if best_match is not None else 0.0,
                'person_iou': best_person_iou if best_match is not None else 0.0,
                'object_iou': best_object_iou if best_match is not None else 0.0,
                'match': False
            })
            if best_match is not None:
                print(f"❌ NO MATCH: Pred {pred_idx} -> GT {best_match}, Min IoU: {best_iou:.4f} < 0.5 threshold (Person: {best_person_iou:.4f}, Object: {best_object_iou:.4f})")
            else:
                print(f"❌ NO MATCH: Pred {pred_idx}, no valid ground truth found")

    # Calculate precision and recall
    true_positives = sum(1 for m in matches if m['match'])
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(gt_hois) if gt_hois else 0.0

    return {
        'matches': matches,
        'precision': precision,
        'recall': recall,
        'true_positives': true_positives,
        'total_predictions': len(predictions),
        'total_gt': len(gt_hois)
    }


def calculate_bbox_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0