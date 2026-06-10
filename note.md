## Steps to eval NLF-GS on GHG

1. Run `convert_processed_thuman_to_ghg` in NLF-GS repo.
2. Generate position and visibility maps:
```
python process_dataset/render_position_map.py --phase val

python process_dataset/render_visibility_map.py --phase val
```
3. Run eval:
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --test_data_root datasets/THuman/val \
  --regressor_path weights/model_gaussian.pth \
  --inpaintor_path weights/model_inpaint.pth \
  --novel_view_nums 1 \
  --bg_color black
```