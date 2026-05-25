import argparse
import shlex
import subprocess
from pathlib import Path


def find_smpl_objs(smpl_obj_root: Path):
    return sorted([p for p in smpl_obj_root.glob('rp*.obj') if p.is_file()])


def run_transfer(command_template: str, src_obj: Path, dst_obj: Path, workdir: Path | None = None):
    cmd = command_template.format(src=src_obj.as_posix(), dst=dst_obj.as_posix())
    print(f"[transfer] {cmd}")
    proc = subprocess.run(shlex.split(cmd), cwd=workdir, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Transfer command failed ({proc.returncode}) for {src_obj}: {cmd}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run external SMPL->SMPL-X transfer tool for each SMPL OBJ and write SMPL-X OBJ outputs. "
            "This script does not implement transfer itself; it orchestrates the official tool."
        )
    )
    parser.add_argument('--smpl-obj-root', required=True, help='Input folder containing rpXXX.obj SMPL meshes')
    parser.add_argument('--out-root', required=True, help='Output folder for transferred rpXXX.obj SMPL-X meshes')
    parser.add_argument('--command-template', required=True,
                        help='Shell command template with placeholders {src} and {dst}. Example: '
                             '"python /path/to/transfer.py --input {src} --output {dst}"')
    parser.add_argument('--workdir', default=None, help='Optional working directory for transfer command')
    parser.add_argument('--max-subjects', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    smpl_obj_root = Path(args.smpl_obj_root)
    out_root = Path(args.out_root)
    workdir = Path(args.workdir) if args.workdir else None

    if not smpl_obj_root.exists():
        raise FileNotFoundError(f"Missing smpl obj root: {smpl_obj_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    smpl_objs = find_smpl_objs(smpl_obj_root)
    if args.max_subjects is not None:
        smpl_objs = smpl_objs[:args.max_subjects]

    if not smpl_objs:
        raise FileNotFoundError(f"No rp*.obj files found in {smpl_obj_root}")

    print(f"[transfer] found {len(smpl_objs)} SMPL objs")

    for src_obj in smpl_objs:
        dst_obj = out_root / src_obj.name
        if dst_obj.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {dst_obj}. Use --overwrite to replace.")

        run_transfer(args.command_template, src_obj, dst_obj, workdir=workdir)

        if not dst_obj.exists():
            raise FileNotFoundError(
                f"Transfer command completed but output missing: {dst_obj}. "
                "Check command-template writes to {dst}."
            )

        print(f"[transfer] done {src_obj.name} -> {dst_obj}")

    print(f"[transfer] completed {len(smpl_objs)} subjects")


if __name__ == '__main__':
    main()
