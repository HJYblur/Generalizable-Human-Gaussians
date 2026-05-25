import argparse, json, math, re
from pathlib import Path
import cv2, numpy as np
from PIL import Image
NUM_SUBVIEWS=5

def angle_to_rp_cam(angle_deg, rp_num_views=36):
    step=360.0/rp_num_views
    return int(math.floor((angle_deg%360.0)/step+0.5))%rp_num_views

def ghg_bin_subview_to_rp_cam(ghg_bin, subview_id, ghg_num_bins=16, rp_num_views=36, num_subviews=5):
    ghg_step=360.0/ghg_num_bins
    local_offset=subview_id/(num_subviews-1)*ghg_step
    angle=ghg_bin*ghg_step+local_offset
    return angle_to_rp_cam(angle,rp_num_views)



def _seq_index(name: str):
    m=re.search(r"seq_(\d+)", name)
    return int(m.group(1)) if m else None

def _filter_first_100(raws):
    kept=[]
    for r in raws:
        idx=_seq_index(r)
        if idx is not None and idx < 100:
            kept.append(r)
    return kept

def main():
    ap=argparse.ArgumentParser();
    ap.add_argument('--raw-root',required=True); ap.add_argument('--out-root',required=True)
    ap.add_argument('--phase',default='val'); ap.add_argument('--pose-id',type=int,default=0)
    ap.add_argument('--out-res',type=int,default=1024); ap.add_argument('--num-ghg-bins',type=int,default=16)
    ap.add_argument('--num-rp-views',type=int,default=36); ap.add_argument('--mapping-mode',choices=['index'],default='index')
    ap.add_argument('--max-subjects',type=int); ap.add_argument('--max-seq-id',type=int,default=99)
    ap.add_argument('--overwrite',action='store_true'); args=ap.parse_args()
    raw_root=Path(args.raw_root); out_root=Path(args.out_root); data_root=out_root/args.phase
    for d in [data_root/'img',data_root/'mask',data_root/'parm']: d.mkdir(parents=True,exist_ok=True)
    human_list=raw_root/'human_list.txt'
    if not human_list.exists(): raise FileNotFoundError(human_list)
    raws=[x.strip() for x in human_list.read_text().splitlines() if x.strip()]
    raws=_filter_first_100(raws)
    raws=[r for r in raws if _seq_index(r) is not None and _seq_index(r)<=args.max_seq_id]
    if args.max_subjects: raws=raws[:args.max_subjects]
    print(f'[prepare] using {len(raws)} subjects with seq <= {args.max_seq_id}')
    mapping={}; splits=[]
    for i,raw in enumerate(raws):
        sid=f'rp{i:03d}'; mapping[sid]=raw; splits.append(sid); print(f'[prepare] {sid} <- {raw}')
        root=raw_root/raw; cam_path=root/'cameras.json'
        if not cam_path.exists(): raise FileNotFoundError(cam_path)
        cams=json.loads(cam_path.read_text()); cams=cams.get('cameras',cams)
        for g in range(args.num_ghg_bins):
            sample=f'{sid}_{g:03d}'; idir=data_root/'img'/sample; mdir=data_root/'mask'/sample; pdir=data_root/'parm'/sample
            if idir.exists() and not args.overwrite: raise FileExistsError(f'{idir} exists, use --overwrite')
            idir.mkdir(parents=True,exist_ok=True); mdir.mkdir(parents=True,exist_ok=True); pdir.mkdir(parents=True,exist_ok=True)
            for sv in range(NUM_SUBVIEWS):
                cid=ghg_bin_subview_to_rp_cam(g,sv,args.num_ghg_bins,args.num_rp_views,NUM_SUBVIEWS); ckey=f'camera{cid:04d}'
                entry=cams[ckey]; K=np.array(entry.get('K',entry.get('intrinsic')),dtype=np.float32).reshape(3,3)
                R=np.array(entry.get('R',entry.get('rotation')),dtype=np.float32).reshape(3,3)
                T=np.array(entry.get('T',entry.get('translation')),dtype=np.float32).reshape(3)
                extr=np.concatenate([R,T[:,None]],1).astype(np.float32)
                ip=root/'img'/ckey/f'{args.pose_id:04d}.jpg'; mp=root/'mask'/ckey/f'{args.pose_id:04d}.png'
                if not ip.exists(): raise FileNotFoundError(ip)
                if not mp.exists(): raise FileNotFoundError(mp)
                rgb=np.array(Image.open(ip).convert('RGB')); mask=np.array(Image.open(mp)); mask=((mask>0).astype(np.uint8)*255)
                if mask.ndim==3: mask=mask[...,0]
                mask=np.stack([mask,mask,mask],-1)
                oh,ow=rgb.shape[:2]; rgb=cv2.resize(rgb,(args.out_res,args.out_res),interpolation=cv2.INTER_AREA)
                mask=cv2.resize(mask,(args.out_res,args.out_res),interpolation=cv2.INTER_NEAREST)
                K=K.copy(); K[0,:]*=args.out_res/ow; K[1,:]*=args.out_res/oh
                Image.fromarray(rgb).save(idir/f'{sv}.jpg'); Image.fromarray(mask).save(mdir/f'{sv}.png')
                np.save(pdir/f'{sv}_intrinsic.npy',K.astype(np.float32)); np.save(pdir/f'{sv}_extrinsic.npy',extr)
    (out_root/f'split_{args.phase}.txt').write_text('\n'.join(splits)+'\n'); (out_root/'subject_mapping.json').write_text(json.dumps(mapping,indent=2))
    print(f'[prepare] done {len(splits)} subjects')
if __name__=='__main__': main()
