# RetinaNet Optimization - Additional Modified Files

These files are from the MLCommons training repository and were modified as part of the RetinaNet inference optimization work.

## Files

### retinanet.py
**Original location:** `training/retired_benchmarks/retinanet/ssd/model/retinanet.py`

**Change:** Removed `batched_nms()` call from `postprocess_detections()`, replaced with `keep = torch.arange(len(image_boxes))`. This removes NMS from the TorchScript forward pass so it can be applied separately as GPU-batched NMS in the inference backend.

### transform.py
**Original location:** `training/retired_benchmarks/retinanet/ssd/model/transform.py`

**Change:** Fixed type annotations (`Tuple[int, int]` → `List[int]`) and added `assert image_size is not None` for TorchScript compatibility when re-exporting the model.

## How to Apply
1. Clone the MLCommons training repo: `https://github.com/mlcommons/training`
2. Navigate to `retired_benchmarks/retinanet/ssd/model/`
3. Replace `retinanet.py` and `transform.py` with the files in this folder
4. Re-export the model using the modified source
