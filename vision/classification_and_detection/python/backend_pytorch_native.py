# pylint: disable=unused-argument,missing-docstring
import sys
import torch
import torchvision
import torchvision.ops
import backend

sys.path.insert(0, '/expanse/lustre/projects/ddp324/blee24/training/retired_benchmarks/retinanet/ssd')

NMS_THRESH = 0.5
SCORE_THRESH = 0.05
DETECTIONS_PER_IMG = 300
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

class BackendPytorchNative(backend.Backend):
    def __init__(self):
        super(BackendPytorchNative, self).__init__()
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mean = None
        self.std = None
        print("ACTUAL DEVICE IN USE:", self.device)

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        torch.ops.load_library(torchvision.ops.__file__.replace('ops/__init__.py', '_C.so'))
        from model.retinanet import retinanet_from_backbone
        self.model = retinanet_from_backbone('resnext50_32x4d', 264, image_size=[800, 800])
        weights = torch.load('/expanse/lustre/projects/ddp324/blee24/resnext50_32x4d_fpn_weights.pth', map_location='cpu')
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.mean = torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(IMAGE_STD).view(1, 3, 1, 1).to(self.device)
        self.inputs = inputs if inputs else ["image"]
        self.outputs = outputs if outputs else ["boxes", "labels", "scores"]
        # Precompute anchor and label indices
        sample = torch.randn(1, 3, 800, 800).to(self.device)
        from model.image_list import ImageList
        images_list = ImageList(sample, [(800, 800)])
        with torch.no_grad():
            features = self.model.backbone(sample)
            features_list = list(features.values())
            head_outputs = self.model.head(features_list)
        num_anchors = head_outputs["cls_logits"].shape[1]
        num_classes = head_outputs["cls_logits"].shape[2]
        self.anchor_idx = torch.arange(num_anchors, device=self.device).repeat_interleave(num_classes)
        self.labels_idx = torch.arange(num_classes, device=self.device).repeat(num_anchors)
        return self

    def apply_nms_batch(self, raw_outputs):
        results = []
        for output in raw_outputs:
            boxes = output["boxes"].to(self.device)
            scores = output["scores"].to(self.device)
            labels = output["labels"].to(self.device)
            keep_mask = scores > SCORE_THRESH
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]
            if len(boxes) == 0:
                results.append({"boxes": boxes, "scores": scores, "labels": labels})
                continue
            keep = torchvision.ops.batched_nms(boxes, scores, labels, NMS_THRESH)
            keep = keep[:DETECTIONS_PER_IMG]
            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            })
        return results

    def predict(self, feed):
        from model.image_list import ImageList
        key = [key for key in feed.keys()][0]
        batch = torch.from_numpy(feed[key]).float().to(self.device)
        batch = (batch - self.mean) / self.std
        n = batch.shape[0]
        images_list = ImageList(batch, [(800, 800)] * n)
        with torch.no_grad():
            features = self.model.backbone(batch)
            features_list = list(features.values())
            head_outputs = self.model.head(features_list)
            anchors = self.model.anchor_generator(images_list, features_list)
            cls_logits = head_outputs['cls_logits']
            bbox_reg = head_outputs['bbox_regression']
            anchors_cat = anchors[0]  # anchors for single image (same for all images)
            boxes_all = torch.cat([self.model.box_coder.decode(bbox_reg[i], [anchors_cat]) for i in range(n)]).reshape(n, -1, 4)
            scores_all = torch.sigmoid(cls_logits)
            scores_flat = scores_all.reshape(n, -1)
            labels_flat = torch.arange(cls_logits.shape[-1], device=self.device).repeat(cls_logits.shape[1]).unsqueeze(0).expand(n, -1)
            anchor_flat = torch.arange(cls_logits.shape[1], device=self.device).repeat_interleave(cls_logits.shape[-1]).unsqueeze(0).expand(n, -1)
            raw_output = []
            for i in range(n):
                keep_mask = scores_flat[i] > self.model.score_thresh
                scores_i = scores_flat[i][keep_mask]
                labels_i = labels_flat[i][keep_mask]
                anchor_i = anchor_flat[i][keep_mask]
                boxes_i = boxes_all[i][anchor_i]
                boxes_i = torch.clamp(boxes_i, min=0, max=800)
                num_topk = min(self.model.topk_candidates, scores_i.size(0))
                scores_i, topk_idx = scores_i.topk(num_topk)
                boxes_i = boxes_i[topk_idx]
                labels_i = labels_i[topk_idx]
                raw_output.append({'boxes': boxes_i, 'scores': scores_i, 'labels': labels_i})
        result = self.apply_nms_batch(raw_output)
        del boxes_all, cls_logits, bbox_reg, head_outputs, features, features_list, anchors, scores_all, scores_flat, labels_flat, anchor_flat, raw_output, batch
        torch.cuda.empty_cache()
        return result
