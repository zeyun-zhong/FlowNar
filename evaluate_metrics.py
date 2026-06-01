import json
from collections import defaultdict
import glob
import os
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


class VideoEvalCap:
    def __init__(
            self, uids, gts, res,
            metrics=("Meteor", "Rouge", "Cider"),
    ):
        self.evalImgs = []
        self.eval = {}
        self.eval_sample = recursive_defaultdict()
        self.imgToEval = {}
        self.uids = uids
        self.gts = gts
        self.res = res
        self.metrics = metrics

    def init_scorers(self):
        metric_dict = {
            "Bleu": (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            "Meteor": (Meteor(), "METEOR"),
            "Rouge": (Rouge(), "ROUGE_L"),
            "Cider": (Cider(), "CIDEr"),
        }
        return [metric_dict[key] for key in self.metrics]

    def evaluate(self):
        gts = self.gts
        res = self.res

        # Set up scorers
        scorers = self.init_scorers()

        # Compute scores
        for scorer, method in scorers:
            print(f'Computing {scorer.method()} score...')
            score, scores = scorer.compute_score(gts, res)

            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self._set_eval(sc, m)
                    self._set_eval_sample(scs, m)
                    print(f"{m}: {sc:.4f}")
            else:
                self._set_eval(score, method)
                self._set_eval_sample(scores, method)
                print(f"{method}: {score:.4f}" if isinstance(score, float) else f"{method}: {score}")

    def _set_eval(self, score, method):
        self.eval[method] = score

    def _set_eval_sample(self, scores, method):
        for uid, score in zip(self.uids, scores):
            self.eval_sample[uid][method] = score


class RecursiveNarrationEvaluator:
    def __init__(self, gts, preds,
                 iou_thresh=0.5,
                 narration_metrics=("Meteor", "Rouge", "Cider")
                 ):
        self.gts = gts
        self.preds = preds
        self.init_metrics()
        self.narration_metrics = narration_metrics
        self.iou_thresh = iou_thresh

    def init_metrics(self):
        self.t_precision = {}
        self.t_recall = {}
        self.t_f1 = {}

    def evaluate(self, dataset:str="ego4d"):
        # Find matches
        if dataset == "ego4d":
            matched_pairs = self.match_gts_preds()
        else:
            matched_pairs = self.match_gts_preds_egoexo4d()

        # Evaluate narration quality of matched pairs
        metrics, metrics_sample = self.evaluate_narrations(matched_pairs)

        metrics['t_precision'] = np.mean(list(self.t_precision.values()))
        metrics['t_recall'] = np.mean(list(self.t_recall.values()))
        metrics['t_f1'] = np.mean(list(self.t_f1.values()))

        # Deal with video-level metrics
        def inner_defaultdict_factory():
            """Returns a defaultdict that defaults to creating lists."""
            return defaultdict(list)
        grouped_by_video = defaultdict(inner_defaultdict_factory)
        for original_key, ms in metrics_sample.items():
            # Find the last occurrence of '_match_' to handle potential underscores in video_id
            match_separator_index = original_key.rfind('_match_')

            video_id = original_key[:match_separator_index]

            for m_name, m_value in ms.items():
                grouped_by_video[video_id][m_name].append(m_value)
                pass

        for video_id, video_metrics in grouped_by_video.items():
            tmp = {}
            # first get mean over all segments
            for m_name, m_values in video_metrics.items():
                tmp[m_name] = np.mean(m_values)

            # Then we add the temporal alignment metrics
            tmp['t_precision'] = self.t_precision[video_id]
            tmp['t_recall'] = self.t_recall[video_id]
            tmp['t_f1'] = self.t_f1[video_id]
            grouped_by_video[video_id] = tmp

        return metrics, grouped_by_video

    def convert_to_segments(self, interval=None, entries=None, start=None):
        """Convert point annotations to fixed-duration segments"""
        if start is None:
            start = float(interval.split(',')[0])
        segments = []
        times = [e['time'] for e in entries]
        times = [start] + times
        segments = [(s, e) if e > s else (e, s) for s, e in zip(times[:-1], times[1:])]
        texts = [e['text'] for e in entries]
        return segments, texts

    def _pairwise_iou(self, gt_segments, pred_segments):
        """Vectorized IoU computation between all GT and prediction segments"""
        gt_starts = gt_segments[:, 0]
        gt_ends = gt_segments[:, 1]
        pred_starts = pred_segments[:, 0]
        pred_ends = pred_segments[:, 1]

        # Compute intersections
        intersections = np.maximum(
            0,
            np.minimum(gt_ends[:, None], pred_ends) -
            np.maximum(gt_starts[:, None], pred_starts)
        )

        # Compute unions
        gt_durations = gt_ends - gt_starts
        pred_durations = pred_ends - pred_starts
        unions = gt_durations[:, None] + pred_durations - intersections

        return intersections / np.clip(unions, 1e-8, None)

    def _pairwise_giou(self, gt_segments, pred_segments):
        """
        Vectorized Temporal Generalized IoU (GIoU) computation between
        all GT and prediction segments.
        """
        # Extract starts and ends
        gt_starts = gt_segments[:, 0]
        gt_ends = gt_segments[:, 1]
        pred_starts = pred_segments[:, 0]
        pred_ends = pred_segments[:, 1]

        # Calculate durations
        gt_durations = gt_ends - gt_starts
        pred_durations = pred_ends - pred_starts

        # --- Standard IoU Calculation ---
        # Compute intersections (I) using broadcasting
        intersections = np.maximum(
            0.0,  # Use float 0.0 for consistency
            np.minimum(gt_ends[:, None], pred_ends) -
            np.maximum(gt_starts[:, None], pred_starts)
        )

        # Compute unions (U) using broadcasting
        unions = gt_durations[:, None] + pred_durations - intersections

        # Compute IoU = I / U
        # Add a small epsilon to avoid division by zero
        unions_clipped = np.clip(unions, 1e-8, None)
        ious = intersections / unions_clipped

        # --- GIoU Specific Calculation ---
        # Compute the start and end of the smallest enclosing segment (C)
        # using broadcasting
        enclose_starts = np.minimum(gt_starts[:, None], pred_starts)
        enclose_ends = np.maximum(gt_ends[:, None], pred_ends)

        # Compute the duration of the enclosing segment (C)
        enclose_durations = enclose_ends - enclose_starts

        # Compute GIoU = IoU - (|C| - |U|) / |C|
        # Add a small epsilon to the denominator to avoid division by zero
        enclose_durations_clipped = np.clip(enclose_durations, 1e-8, None)
        gious = ious - (enclose_durations - unions) / enclose_durations_clipped

        return gious

    def match_gts_preds(self):
        matched_pairs = {}
        for video_uid in self.preds:
            if video_uid not in self.gts:
                continue
            video_preds = self.preds[video_uid]
            video_gts = self.gts[video_uid]
            for interval in video_preds:
                if interval not in video_gts:
                    continue
                # Collect all GT narration entries for this interval across all UUIDs
                gt_times = []
                gt_texts = []
                for uuid in video_gts[interval]:
                    gt_narrations = video_gts[interval][uuid]['narration']
                    segments, texts = self.convert_to_segments(interval, gt_narrations)
                    gt_times.extend(segments)
                    gt_texts.extend(texts)

                if not gt_times:  # empty list
                    continue

                # Extract GT data
                gt_times = np.array(gt_times)
                gt_texts = np.array(gt_texts, dtype=object)

                # Extract prediction data
                pred_entries = video_preds[interval]
                pred_times, pred_texts = self.convert_to_segments(interval, pred_entries)
                pred_times = np.array(pred_times)
                pred_texts = np.array(pred_texts, dtype=object)

                # Calculate temporal alignment metrics
                aligned_indices = self.evaluate_temporal_alighment(pred_times, gt_times, video_uid=f'{video_uid}_{interval}')

                # Create matches with metrics
                for idx, (gt_idx, pred_idx) in enumerate(aligned_indices):
                    key = f"{video_uid}_{interval}_match_{idx + 1}"
                    matched_pairs[key] = {
                        'gt': gt_texts[gt_idx],
                        'pred': pred_texts[pred_idx],
                    }
        return matched_pairs

    def match_gts_preds_egoexo4d(self):
        matched_pairs = {}
        for video_uid in self.preds:
            if video_uid not in self.gts:
                continue
            video_preds = self.preds[video_uid]
            video_gts = self.gts[video_uid]
            # Collect all GT narration entries for this interval across all UUIDs
            gt_times = []
            gt_texts = []
            for gt_narrations in video_gts:
                segments, texts = self.convert_to_segments(entries=gt_narrations, start=0)
                gt_times.extend(segments)
                gt_texts.extend(texts)

            if not gt_times:  # empty list
                continue

            # Extract GT data
            gt_times = np.array(gt_times)
            gt_texts = np.array(gt_texts, dtype=object)

            # Extract prediction data
            pred_entries = video_preds
            pred_times, pred_texts = self.convert_to_segments(entries=pred_entries, start=0)
            pred_times = np.array(pred_times)
            pred_texts = np.array(pred_texts, dtype=object)

            # Calculate temporal alignment metrics
            aligned_indices = self.evaluate_temporal_alighment(pred_times, gt_times, video_uid)

            # Create matches with metrics
            for idx, (gt_idx, pred_idx) in enumerate(aligned_indices):
                key = f"{video_uid}_match_{idx + 1}"
                matched_pairs[key] = {
                    'gt': gt_texts[gt_idx],
                    'pred': pred_texts[pred_idx],
                }
        return matched_pairs


    def evaluate_temporal_alighment(self, pred_times, gt_times, video_uid):
        iou_matrix = self._pairwise_iou(gt_times, pred_times)

        # 1. Find all valid matches
        matches = iou_matrix >= self.iou_thresh

        # 2. Prediction-centric precision calculation
        tp_p = np.sum(np.any(matches, axis=0))  # Predictions with ≥1 match
        fp = len(pred_times) - tp_p

        # 3. GT-centric recall calculation
        tp_gt = np.sum(np.any(matches, axis=1))  # GTs covered by ≥1 prediction
        fn = len(gt_times) - tp_gt

        # 4. Compute metrics
        precision = tp_p / (tp_p + fp) if (tp_p + fp) else 0
        recall = tp_gt / (tp_gt + fn) if (tp_gt + fn) else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        self.t_precision[video_uid] = precision
        self.t_recall[video_uid] = recall
        self.t_f1[video_uid] = f1

        # 5. Return indices of matches
        # Previously, we used np.argwhere(matches) to get indices of matches
        giou_matrix = self._pairwise_giou(gt_times, pred_times)
        # indices = np.argwhere(matches)
        pred_indices = giou_matrix.argmax(axis=1)
        indices = np.stack((np.arange(giou_matrix.shape[0]), pred_indices), axis=1)
        return indices

    def evaluate_narrations(self, matched_pairs):
        gts = {key: [val['gt']] for key, val in matched_pairs.items()}
        preds = {key: [val['pred']] for key, val in matched_pairs.items()}
        keys = list(matched_pairs.keys())
        narration_evaluator = VideoEvalCap(keys, gts, preds, metrics=self.narration_metrics)
        narration_evaluator.evaluate()
        return narration_evaluator.eval, narration_evaluator.eval_sample


def preprocess_predicted_narrations(fps, timestep, narration, start_time=0):
    # Convert frame numbers to seconds
    sec = (float(timestep) + 1) / fps + start_time

    # ' You walk around the workshop.<|eot_id|>' -> 'You walk around the workshop.'
    narration = narration.strip().replace("<|eot_id|>", "")
    return (sec, narration)


def calculate_metrics_for_streaming_narrations_ego4d(pred_path, gts, fps=2):
    sample_names = [f"{vid}_{key}" for vid, annos in gts.items() for key in annos.keys()]

    all_narration_files = sorted(glob.glob(os.path.join("outputs", pred_path, '*.json')))
    all_narration_files = [f for f in all_narration_files if "predict" not in f]

    assert len(all_narration_files) == len(sample_names)

    # Construct preds
    preds = recursive_defaultdict()
    for i, sample_name in enumerate(sample_names):
        narrations = json.load(open(all_narration_files[i]))
        vid, interval = sample_name.split("_")
        start_time = float(interval.split(",")[0])
        narrations = [preprocess_predicted_narrations(fps, key, val, start_time) for key, val in narrations.items()]
        preds[vid][interval] = [{"text": text, "time": time} for time, text in narrations]

    evaluator = RecursiveNarrationEvaluator(gts, preds)
    metrics, metrics_sample = evaluator.evaluate()

    sorted_metric_sample = compare_gts_preds(gts, preds, metrics_sample, is_ego4d=True)

    print(metrics)
    return metrics


def calculate_metrics_for_streaming_narrations_egoexo4d(pred_path, gts, fps=2):
    # We only select narrations that have ground truth narrations
    sample_names = []
    gts_selected = defaultdict(list)
    for key, annos in gts.items():

        start, end = float("inf"), 0
        for annotation in annos:
            narrations = annotation['descriptions']
            if not narrations:
                continue
            start = min(start, narrations[0]['timestamp'])
            end = max(end, narrations[-1]['timestamp'])
            to_append = [{'time': el['timestamp'], 'text': el['text']} for el in narrations]
            gts_selected[key].append(to_append)

        if end == 0:
            continue
        sample_names.append((key, start, end))

    all_narration_files = sorted(glob.glob(os.path.join("outputs", pred_path, '*.json')))
    all_narration_files = [f for f in all_narration_files if "predict" not in f]

    assert len(all_narration_files) == len(sample_names)

    # Construct preds
    preds = recursive_defaultdict()
    for i, (sample_name, start, end) in enumerate(sample_names):
        narrations = json.load(open(all_narration_files[i]))
        narrations = [preprocess_predicted_narrations(fps, key, val, start) for key, val in narrations.items()]
        preds[sample_name] = [{"text": text, "time": time} for time, text in narrations]
        pass

    evaluator = RecursiveNarrationEvaluator(gts_selected, preds)
    metrics, metrics_sample = evaluator.evaluate(dataset="egoexo4d")

    # sorted_metric_sample = compare_gts_preds(gts, preds, metrics_sample)

    print(metrics)
    return metrics


def calculate_metrics_for_streaming_narrations_ek100(pred_path, gts, fps=2):
    # We only select narrations that have ground truth narrations
    sample_names = [(key, el[0]['time'], el[-1]['time']) for key, el in gts.items()]
    gts_selected = defaultdict(list)
    for key, annos in gts.items():
        gts_selected[key].append(annos)

    all_narration_files = sorted(glob.glob(os.path.join("outputs", pred_path, '*.json')))
    all_narration_files = [f for f in all_narration_files if "predict" not in f]

    assert len(all_narration_files) == len(sample_names)

    # Construct preds
    preds = recursive_defaultdict()
    for i, (sample_name, start, end) in enumerate(sample_names):
        narrations = json.load(open(all_narration_files[i]))
        narrations = [preprocess_predicted_narrations(fps, key, val, start) for key, val in narrations.items()]
        preds[sample_name] = [{"text": text, "time": time} for time, text in narrations]
        pass

    evaluator = RecursiveNarrationEvaluator(gts_selected, preds)
    metrics, metrics_sample = evaluator.evaluate(dataset="ek100")

    sorted_metric_sample = compare_gts_preds(gts, preds, metrics_sample)
    print(metrics)
    return metrics


def calculate_metrics_for_streaming_narrations(pred_path, fps=2, data_root=""):
    if "egoexo4d" in pred_path:
        gts = json.load(open(os.path.join(data_root, "egoexo4d/annotations/refined_egocentric_atomic_descriptions_gpt_val.json")))
        return calculate_metrics_for_streaming_narrations_egoexo4d(pred_path, gts, fps)
    if "ek100" in pred_path:
        gts = json.load(open(os.path.join(data_root, "epickitchens100/annotations/refined_val_narrations.json")))
        return calculate_metrics_for_streaming_narrations_ek100(pred_path, gts, fps)
    gts = json.load(open(os.path.join(data_root, "ego4d/v2/annotations/refined_narration_stream_val.json")))
    return calculate_metrics_for_streaming_narrations_ego4d(pred_path, gts, fps)


def compare_gts_preds(gts, preds, metrics_sample, is_ego4d=False):
    for video_id, video_metrics in metrics_sample.items():
        if is_ego4d:
            key = video_id.split("_")[0]
            video_metrics['gt'] = gts[key][video_id.split("_")[1]]
            video_metrics['pred'] = preds[key][video_id.split("_")[1]]
        else:
            video_metrics['gt'] = gts[video_id]
            video_metrics['pred'] = preds[video_id]

    sorted_metrics_sample = sorted(metrics_sample.items(), key=lambda x: x[1]['t_f1'], reverse=True)
    return sorted_metrics_sample

