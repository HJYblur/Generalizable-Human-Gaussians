import argparse, re, shutil
from pathlib import Path
import numpy as np, trimesh
PAT=re.compile(r'(rp\d{3})')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fit-root',required=True); ap.add_argument('--out-root',required=True)
    ap.add_argument('--max-subjects',type=int); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args()
    fr=Path(a.fit_root); out=Path(a.out_root); out.mkdir(parents=True,exist_ok=True)
    files=sorted([p for p in fr.rglob('*') if p.suffix.lower() in ['.obj','.pkl']])
    c=0
    for p in files:
        m=PAT.search(p.stem)
        if not m: continue
        sid=m.group(1); dst=out/f'{sid}.obj'
        if dst.exists() and not a.overwrite: raise FileExistsError(dst)
        if p.suffix.lower()=='.obj': shutil.copy2(p,dst); print(f'[smplx] copy {p} -> {dst}')
        else:
            d=np.load(p,allow_pickle=True)
            if 'vertices' not in d or 'faces' not in d: print(f'[warn] missing vertices/faces: {p}'); continue
            trimesh.Trimesh(vertices=np.array(d['vertices']),faces=np.array(d['faces']),process=False).export(dst); print(f'[smplx] export {p} -> {dst}')
        c+=1
        if a.max_subjects and c>=a.max_subjects: break
    print(f'[smplx] done {c}')
if __name__=='__main__': main()
