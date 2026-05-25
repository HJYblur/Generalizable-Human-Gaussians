import argparse, os, glob
import imageio.v2 as imageio
import skimage.metrics
import numpy as np
import torch
from lpips import LPIPS

def mse(a,b): return np.mean((a-b)**2)

def parse_views(s): return [int(x) for x in s.split(',') if x!='']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target-root',required=True); ap.add_argument('--mask-root',required=True)
    ap.add_argument('--pred-root',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--eval-target-views',default='3,8,13'); args=ap.parse_args()
    os.makedirs(args.output_root,exist_ok=True)
    ours=os.path.join(args.output_root,'pred'); gt=os.path.join(args.output_root,'gt'); os.makedirs(ours,exist_ok=True); os.makedirs(gt,exist_ok=True)
    humans=sorted({x.rsplit('_',1)[0] for x in os.listdir(args.target_root) if '_' in x})
    views=parse_views(args.eval_target_views)
    psnr=[]; ssim=[]
    for h in humans:
        for v in views:
            pred=f"{args.pred_root}/{h}_{v:03d}_novel00.jpg"; tgt=f"{args.target_root}/{h}_{v:03d}/0.jpg"
            if not (os.path.exists(pred) and os.path.exists(tgt)): continue
            g=imageio.imread(pred).astype(np.float32)/255.; t=imageio.imread(tgt).astype(np.float32)/255.
            if g.shape!=t.shape: g=np.array(imageio.core.util.Array(g));
            m=mse(g,t); psnr.append(10*np.log10(1.0/max(m,1e-8))); ssim.append(skimage.metrics.structural_similarity(g,t,channel_axis=2,data_range=1))
            imageio.imwrite(f"{ours}/{h}_{v:03d}.png",(g*255).astype(np.uint8)); imageio.imwrite(f"{gt}/{h}_{v:03d}.png",(t*255).astype(np.uint8))
    print('PSNR',float(np.mean(psnr))); print('SSIM',float(np.mean(ssim)))
    lp=LPIPS(net='alex'); lp=lp.cuda() if torch.cuda.is_available() else lp
    vals=[]
    for gp,tp in zip(sorted(glob.glob(f'{ours}/*.png')),sorted(glob.glob(f'{gt}/*.png'))):
        g=torch.from_numpy(imageio.imread(gp).astype(np.float32)/127.5-1).permute(2,0,1)[None]
        t=torch.from_numpy(imageio.imread(tp).astype(np.float32)/127.5-1).permute(2,0,1)[None]
        if torch.cuda.is_available(): g,t=g.cuda(),t.cuda()
        vals.append(lp(g,t).item())
    print('LPIPS_ALEX',float(np.mean(vals)) if vals else float('nan'))

if __name__=='__main__': main()
