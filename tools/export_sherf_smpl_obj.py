import argparse, json
from pathlib import Path
import numpy as np, torch, trimesh, smplx

def pick(x,pid):
    a=np.array(x)
    if a.ndim==1: return a
    if a.shape[0]==1:return a[0]
    return a[pid]

def load_params(path,pid):
    d=np.load(path,allow_pickle=True)
    if 'smpl' in d: s=d['smpl'].item() if hasattr(d['smpl'],'item') else d['smpl']
    else: s=d
    return {k:pick(s[k],pid).astype(np.float32) for k in ['betas','global_orient','body_pose','transl']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-root',required=True); ap.add_argument('--out-root',required=True)
    ap.add_argument('--dataset-root',required=True); ap.add_argument('--smpl-model-root',required=True); ap.add_argument('--pose-id',type=int,default=0)
    ap.add_argument('--gender',default='neutral'); ap.add_argument('--max-subjects',type=int); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args()
    rr=Path(a.raw_root); out=Path(a.out_root); out.mkdir(parents=True,exist_ok=True)
    mapping_path=Path(a.dataset_root)/'subject_mapping.json'
    if mapping_path.exists(): mapping=json.loads(mapping_path.read_text())
    else:
        raws=[x.strip() for x in (rr/'human_list.txt').read_text().splitlines() if x.strip()]; mapping={f'rp{i:03d}':r for i,r in enumerate(raws)}
    items=sorted(mapping.items())[:a.max_subjects] if a.max_subjects else sorted(mapping.items())
    model=smplx.create(model_path=a.smpl_model_root,model_type='smpl',gender=a.gender,batch_size=1,create_global_orient=False,create_body_pose=False,create_betas=False,create_transl=False)
    for sid,raw in items:
        p=rr/raw/'outputs_re_fitting'/'refit_smpl_2nd.npz'
        if not p.exists(): raise FileNotFoundError(p)
        prm=load_params(p,a.pose_id)
        outp=out/f'{sid}.obj'
        if outp.exists() and not a.overwrite: raise FileExistsError(outp)
        o=model(betas=torch.tensor(prm['betas'])[None],global_orient=torch.tensor(prm['global_orient'])[None],body_pose=torch.tensor(prm['body_pose'])[None],transl=torch.tensor(prm['transl'])[None])
        mesh=trimesh.Trimesh(vertices=o.vertices.detach().cpu().numpy()[0],faces=model.faces,process=False); mesh.export(outp)
        print(f'[smpl] {sid} -> {outp}')
if __name__=='__main__': main()
