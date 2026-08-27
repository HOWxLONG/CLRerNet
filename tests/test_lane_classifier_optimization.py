import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from libs.lane_classifier.crop import CLASSES, lane_strip_crop_with_mode
from libs.lane_classifier.dataset import LaneStripDataset
from libs.lane_classifier.eval import (
    empty_end_to_end_counts,
    end_to_end_metrics,
    update_end_to_end_counts,
)
from libs.lane_classifier.infer import classify_lane
from libs.lane_classifier.model import build_model, load_classifier_checkpoint


class LaneClassifierOptimizationTest(unittest.TestCase):
    def test_models_forward_and_checkpoint_compatibility(self):
        inputs = torch.rand(2, 3, 288, 128)
        for model_type in ('legacy', 'sequence_fusion'):
            model = build_model(model_type=model_type)
            self.assertEqual(tuple(model(inputs).shape), (2, len(CLASSES)))

        with tempfile.TemporaryDirectory() as directory:
            legacy = build_model(model_type='legacy')
            old_path = Path(directory) / 'old.pth'
            torch.save({'state_dict': legacy.state_dict(), 'classes': CLASSES}, old_path)
            loaded, meta = load_classifier_checkpoint(old_path)
            self.assertEqual(meta['model_type'], 'legacy')
            self.assertEqual(tuple(loaded(inputs).shape), (2, len(CLASSES)))

            sequence = build_model(model_type='sequence_fusion')
            new_path = Path(directory) / 'new.pth'
            torch.save(
                {
                    'state_dict': sequence.state_dict(),
                    'classes': CLASSES,
                    'model_type': 'sequence_fusion',
                    'crop_mode': 'scaled',
                    'temperature': 1.7,
                },
                new_path,
            )
            _, meta = load_classifier_checkpoint(new_path)
            self.assertEqual(meta['crop_mode'], 'scaled')
            self.assertAlmostEqual(meta['temperature'], 1.7)

    def test_crop_modes_and_calibrated_probabilities(self):
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        points = [[280, 350], [300, 250], [315, 150], [320, 60]]
        for mode, expected_width in (('fixed', 128), ('scaled', 64), ('normalized', 64)):
            crop, effective_width = lane_strip_crop_with_mode(
                image,
                points,
                crop_mode=mode,
                strip_width=64 if mode == 'normalized' else 128,
                normalized_size=(1024, 544),
            )
            self.assertEqual(crop.shape, (288, 128, 3))
            self.assertEqual(effective_width, expected_width)

        class ConstantClassifier(torch.nn.Module):
            def forward(self, tensor):
                return torch.tensor([[0.0, 1.0, 2.0]], device=tensor.device).repeat(len(tensor), 1)

        settings = {
            'crop_size': (288, 128),
            'strip_width': 128,
            'crop_mode': 'fixed',
            'strip_reference_width': 2560,
            'strip_min_width': 64,
            'strip_max_width': 192,
            'normalized_size': (1024, 544),
            'top_crop_ratio': 0.08,
        }
        label, score, probs, _, _ = classify_lane(
            ConstantClassifier(), image, points, settings, 'cpu', temperature=2.0
        )
        self.assertEqual(label, 'joint')
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertAlmostEqual(score, probs['joint'])

    def test_prediction_matched_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            predictions = Path(directory) / 'predictions'
            root.mkdir()
            predictions.mkdir()
            image_path = root / 'sample.jpg'
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.line(image, (280, 350), (320, 60), (255, 255, 255), 8)
            cv2.imwrite(str(image_path), image)
            points = [[280, 350], [295, 250], [310, 150], [320, 60]]
            annotation = {
                'imageHeight': 360,
                'imageWidth': 640,
                'shapes': [{'label': 'solid', 'points': points}],
            }
            (root / 'sample.json').write_text(json.dumps(annotation), encoding='utf-8')
            (root / 'split.txt').write_text('sample.jpg\n', encoding='utf-8')
            prediction = {'image': 'sample.jpg', 'lanes': [{'points': points, 'lane_score': 0.9}]}
            (predictions / 'sample.json').write_text(json.dumps(prediction), encoding='utf-8')

            dataset = LaneStripDataset(
                data_root=root,
                split_file=root / 'split.txt',
                prediction_dir=predictions,
                predicted_crop_prob=1.0,
                require_prediction=True,
                crop_mode='scaled',
            )
            tensor, target, meta = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 288, 128))
            self.assertEqual(int(target), 0)
            self.assertEqual(meta['crop_source'], 'prediction')
            self.assertGreaterEqual(meta['prediction_iou'], 0.99)
            self.assertEqual(meta['effective_strip_width'], 64)

    def test_end_to_end_class_counts(self):
        gt = [
            {'label': 'solid'},
            {'label': 'dashed'},
            {'label': 'joint'},
        ]
        pred = [
            {'label': 'solid'},
            {'label': 'solid'},
            {'label': 'dashed'},
        ]
        matches = [
            {'gt_index': 0, 'pred_index': 0, 'iou': 0.8},
            {'gt_index': 1, 'pred_index': 1, 'iou': 0.7},
        ]
        counts = empty_end_to_end_counts()
        update_end_to_end_counts(counts, gt, pred, matches, {0, 1}, {0, 1})
        metrics = end_to_end_metrics(counts)
        self.assertEqual(metrics['per_class']['solid']['tp'], 1)
        self.assertEqual(metrics['per_class']['solid']['fp'], 1)
        self.assertEqual(metrics['per_class']['dashed']['fp'], 1)
        self.assertEqual(metrics['per_class']['dashed']['fn'], 1)
        self.assertEqual(metrics['per_class']['joint']['fn'], 1)


if __name__ == '__main__':
    unittest.main()
