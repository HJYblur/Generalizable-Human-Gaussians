import argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image

def chk(p,miss):
    if not p.exists(): miss.append(str(p))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--subject',required=True); ap.add_argument('--sample',required=True)
    ap.add_argument('--resolution',type=int,default=1024); ap.add_argument('--required-input-views',default='0,6,11'); ap.add_argument('--required-target-views',default='3,8,13'); a=ap.parse_args()
    r=Path(a.data_root); s=a.subject; sm=a.sample; miss=[]
    base=[r/f'img/{sm}/0.jpg',r/f'mask/{sm}/0.png',r/f'parm/{sm}/0_intrinsic.npy',r/f'parm/{sm}/0_extrinsic.npy',r/f'smplx_obj/{s}.obj',r/f'position_map_uv_space/{s}_{a.resolution}.npy']
    base += [r/f'position_map_uv_space_outer_shell_{i}/{s}_{a.resolution}.npy' for i in range(1,5)]
    base += [r/f'visibility_map_uv_space/{sm}.npy']+[r/f'visibility_map_uv_space_outer_shell_{i}/{sm}.npy' for i in range(1,5)]
    for p in base: chk(p,miss)
    for v in [int(x) for x in a.required_input_views.split(',') if x]:
        n=f'{s}_{v:03d}'
        for p in [r/f'img/{n}/0.jpg',r/f'mask/{n}/0.png',r/f'parm/{n}/0_intrinsic.npy',r/f'parm/{n}/0_extrinsic.npy',r/f'visibility_map_uv_space/{n}.npy']+[r/f'visibility_map_uv_space_outer_shell_{i}/{n}.npy' for i in range(1,5)]: chk(p,miss)
    for v in [int(x) for x in a.required_target_views.split(',') if x]:
        n=f'{s}_{v:03d}'
        for p in [r/f'img/{n}/0.jpg',r/f'mask/{n}/0.png',r/f'parm/{n}/0_intrinsic.npy',r/f'parm/{n}/0_extrinsic.npy']: chk(p,miss)
    if miss:
        print('Missing critical files:'); [print(m) for m in miss]; sys.exit(1)
    img=Image.open(r/f'img/{sm}/0.jpg'); mask=Image.open(r/f'mask/{sm}/0.png')
    K=np.load(r/f'parm/{sm}/0_intrinsic.npy'); E=np.load(r/f'parm/{sm}/0_extrinsic.npy'); P=np.load(r/f'position_map_uv_space/{s}_{a.resolution}.npy'); V=np.load(r/f'visibility_map_uv_space/{sm}.npy')
    print('image',img.size,'mask',mask.size,'K',K.shape,'E',E.shape,'position',P.shape,'visibility',V.shape)

if __name__=='__main__': main()
