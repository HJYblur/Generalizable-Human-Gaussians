# RenderPeople -> GHG Cross-Domain Eval

Purpose: evaluate pretrained GHG on SHERF RenderPeople without training/fine-tuning.

Key facts:
- GHG protocol bins: 16 (`*_000..*_015`)
- Pretrained input/source views: `[0,6,11]`
- Metric target views: `[3,8,13]`
- Position/visibility maps are from aligned SMPL-X OBJ meshes.
- Original RenderPeople scanned meshes are not required.
- SHERF provides SMPL params; run official SMPL->SMPL-X transfer externally.

## Commands
(uses scripts added in this patch)

```bash
python tools/prepare_sherf_renderpeople_for_ghg.py --raw-root datasets/RenderPeople/raw_sherf/20230228 --out-root datasets/RenderPeople --phase val --pose-id 0 --out-res 1024 --mapping-mode index
python tools/export_sherf_smpl_obj.py --raw-root datasets/RenderPeople/raw_sherf/20230228 --dataset-root datasets/RenderPeople --out-root datasets/RenderPeople/transfer/smpl_obj --smpl-model-root datasets/RenderPeople/models --pose-id 0
# run official SMPL-X transfer via wrapper script (uses your official transfer command)
python tools/transfer_smpl_obj_to_smplx_obj.py --smpl-obj-root datasets/RenderPeople/transfer/smpl_obj --out-root datasets/RenderPeople/transfer/smplx_fit --command-template "python /path/to/official_transfer.py --input {src} --output {dst}"
python tools/export_transferred_smplx_obj.py --fit-root datasets/RenderPeople/transfer/smplx_fit --out-root datasets/RenderPeople/val/smplx_obj
python process_dataset/render_position_map.py --dataset-root datasets/RenderPeople --phase val --resolution 1024 --smplx-uv-obj datasets/RenderPeople/smplx_uv.obj
python process_dataset/render_visibility_map.py --dataset-root datasets/RenderPeople --phase val --resolution 1024 --smplx-uv-obj datasets/RenderPeople/smplx_uv.obj
python tools/check_renderpeople_ghg_dataset.py --data-root datasets/RenderPeople/val --subject rp000 --sample rp000_000 --required-input-views 0,6,11 --required-target-views 3,8,13
CUDA_VISIBLE_DEVICES=0 python eval.py --test_data_root datasets/RenderPeople/val --regressor_path weights/model_gaussian.pth --inpaintor_path weights/model_inpaint.pth --novel_view_nums 4 --bg_color black --exp_name GHG_RenderPeople
python metrics/compute_metrics_custom.py --target-root datasets/RenderPeople/val/img --mask-root datasets/RenderPeople/val/mask --pred-root outputs/eval/GHG_RenderPeople --output-root outputs/metrics/GHG_RenderPeople --eval-target-views 3,8,13
```
