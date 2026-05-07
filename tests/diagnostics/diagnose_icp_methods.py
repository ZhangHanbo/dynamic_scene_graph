"""Per-frame diagnostic: log init/output T_co for chain vs anchor."""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from tests.visualize_pipeline import TrajectoryData, K
from perception.icp_pose import _back_project, _voxelize
from scipy.spatial.transform import Rotation
import open3d as o3d


VOXEL = 0.005
THRESH = 0.020
MAX_ITER = 30


def run_icp(src_pts, tgt_pts, init_T):
    src = _voxelize(src_pts, VOXEL)
    tgt = _voxelize(tgt_pts, VOXEL)
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, THRESH, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER),
    )
    return (np.asarray(result.transformation, dtype=np.float64),
            float(result.fitness), float(result.inlier_rmse))


def rot_diff_deg(R1, R2):
    R = R1.T @ R2
    try:
        return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
    except Exception:
        return float('nan')


def main(traj='bottle_box_2', step=2,
         out_csv='tests/icp_chain_vs_anchor_log.csv'):
    data = TrajectoryData(trajectory=traj, step=step)
    refs_chain = {}
    refs_anchor = {}
    rows = []

    for local_i, idx in enumerate(data.indices):
        frame = data.load_frame(idx)
        if frame is None:
            continue
        rgb, depth, dets = frame
        for d in dets:
            oid = d.get('id')
            if oid is None:
                continue
            pts_cam = _back_project(d['mask'], depth, K)
            if pts_cam is None:
                continue
            centroid_now = pts_cam.mean(axis=0)

            if oid not in refs_chain:
                ref_points = pts_cam - centroid_now
                T_init = np.eye(4); T_init[:3, 3] = centroid_now
                refs_chain[oid] = {'ref': ref_points, 'prev_T_co': T_init.copy()}
                refs_anchor[oid] = {'ref': ref_points}
                continue

            init_chain = refs_chain[oid]['prev_T_co'].copy()
            init_chain[:3, 3] = centroid_now

            init_anchor = np.eye(4)
            init_anchor[:3, 3] = centroid_now

            T_chain, fit_chain, rmse_chain = run_icp(
                refs_chain[oid]['ref'], pts_cam, init_chain)
            T_anchor, fit_anchor, rmse_anchor = run_icp(
                refs_anchor[oid]['ref'], pts_cam, init_anchor)

            r_init_diff = rot_diff_deg(init_chain[:3, :3], init_anchor[:3, :3])
            r_out_diff = rot_diff_deg(T_chain[:3, :3], T_anchor[:3, :3])
            t_out_diff_mm = np.linalg.norm(T_chain[:3, 3] - T_anchor[:3, 3]) * 1000

            rows.append({
                'frame': idx, 'oid': oid,
                'init_chain_R_from_I_deg': rot_diff_deg(init_chain[:3,:3], np.eye(3)),
                'init_diff_R_deg': r_init_diff,
                'fit_chain': fit_chain, 'fit_anchor': fit_anchor,
                'rmse_chain_mm': rmse_chain * 1000,
                'rmse_anchor_mm': rmse_anchor * 1000,
                'out_R_diff_deg': r_out_diff,
                'out_t_diff_mm': t_out_diff_mm,
            })

            if (fit_chain >= 0.9 and rmse_chain <= 0.015
                    and np.isfinite(T_chain).all()):
                refs_chain[oid]['prev_T_co'] = T_chain.copy()
            else:
                refs_chain[oid]['prev_T_co'][:3, 3] = centroid_now

    print(f'Logged {len(rows)} (frame, oid) ICP comparisons.')
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'Saved CSV to {out_csv}')

    arr = {k: np.asarray([r[k] for r in rows], dtype=float) for k in fieldnames}
    init_R = arr['init_diff_R_deg']
    out_R = arr['out_R_diff_deg']
    out_t = arr['out_t_diff_mm']
    fit_c, fit_a = arr['fit_chain'], arr['fit_anchor']
    rms_c, rms_a = arr['rmse_chain_mm'], arr['rmse_anchor_mm']

    def stats(name, x, fmt='.2f'):
        return (f'{name:30s} median={np.median(x):{fmt}}  '
                f'mean={x.mean():{fmt}}  p95={np.percentile(x,95):{fmt}}  '
                f'max={x.max():{fmt}}')

    print()
    print('=== INIT DIFFERENCES (warmstart given to ICP) ===')
    print(stats('init R diff (deg)', init_R))
    print()
    print('=== OUTPUT DIFFERENCES (after ICP convergence) ===')
    print(stats('output R diff (deg)', out_R, '.3f'))
    print(stats('output t diff (mm)', out_t, '.3f'))
    print()
    print('=== FITNESS / RMSE COMPARISON ===')
    fit_eq = (np.abs(fit_c - fit_a) < 0.001).sum()
    print(f'  Fitness identical (within 0.001): {fit_eq} / {len(rows)}')
    rmse_diff_um = np.abs(rms_c - rms_a) * 1000
    print(stats('  RMSE diff (μm)', rmse_diff_um))
    print()
    print('=== HOW OFTEN DOES THE INIT MATTER? ===')
    for thr in [1, 5, 10, 30, 60]:
        mask = init_R > thr
        if mask.sum() == 0:
            continue
        out_diffs = out_R[mask]
        converged_same = (out_diffs < 1.0).sum()
        diverged = (out_diffs > 5.0).sum()
        print(f'  init diff > {thr:>2}° (n={mask.sum():>4}): '
              f'output diff <1° in {converged_same:>4} '
              f'({100*converged_same/mask.sum():>3.0f}%) [same minimum], '
              f'output diff >5° in {diverged:>4} '
              f'({100*diverged/mask.sum():>3.0f}%) [different minima]')


if __name__ == '__main__':
    main()
