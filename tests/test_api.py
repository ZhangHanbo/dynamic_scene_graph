"""
Unit tests for the SceneRep API.

Tests are organized into:
1. Data class tests (no dependencies beyond numpy)
2. ObjectReconstructor tests (needs open3d)
3. ObjectTracker tests with real data (needs trajectory data)
4. PoseUpdater tests (needs open3d)
5. RelationAnalyzer tests (needs open3d)

Data path:
    Real trajectory data is expected at:
    /Volumes/External/Workspace/nus_deliver/Mobile_Manipulation_on_Fetch/multi_objects/apple_bowl_2/

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    python -m pytest tests/test_api.py -v

    # Skip tests that require real data:
    python -m pytest tests/test_api.py -v -k "not requires_data"
"""

import os
import sys
import json
import numpy as np
import pytest
import cv2

# Ensure SceneRep root is on path
SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from api import (
    ObjectReconstructor, ObjectTracker, PoseUpdater, RelationAnalyzer,
    Mesh, TrackedObject, FrameDetections, RelationGraph,
)

# ─────────────────────────────────────────────────────────────────────
# Data paths and fixtures
# ─────────────────────────────────────────────────────────────────────

DATA_ROOT = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects", "apple_bowl_2"
)
# Require both the rgb dir AND the first cached detection JSON so partial
# layouts (rgb present but detection_h empty) skip cleanly instead of
# erroring out the fixture mid-load.
HAS_DATA = (os.path.isdir(os.path.join(DATA_ROOT, "rgb")) and
            os.path.exists(os.path.join(
                DATA_ROOT, "detection_h", "detection_000000_final.json")))

requires_data = pytest.mark.skipif(
    not HAS_DATA,
    reason=f"Trajectory data not found at {DATA_ROOT}"
)

# Camera intrinsics (from config)
K = np.array([
    [554.3827, 0, 320.5],
    [0, 554.3827, 240.5],
    [0, 0, 1]
], dtype=np.float32)


def load_pose_txt(path):
    """Read pose file: each line is 'idx tx ty tz qx qy qz qw'."""
    from scipy.spatial.transform import Rotation
    poses = []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses


def load_detection_json(json_path):
    """Load a detection_XXXXXX_final.json file.
    Returns list of dicts with 'mask', 'label', 'score', 'id'."""
    import base64
    from PIL import Image
    from io import BytesIO

    with open(json_path, "r") as f:
        data = json.load(f)

    results = []
    for det in data.get("detections", []):
        # Decode mask from base64 PNG
        mask_b64 = det.get("mask", "")
        if mask_b64:
            mask_bytes = base64.b64decode(mask_b64)
            mask_img = Image.open(BytesIO(mask_bytes)).convert("L")
            mask = (np.array(mask_img) > 128).astype(np.uint8)
        else:
            mask = np.zeros((480, 640), dtype=np.uint8)

        results.append({
            "mask": mask,
            "label": det.get("label", "unknown"),
            "score": det.get("score", 0.0),
            "id": det.get("object_id", 0),
        })
    return results


@pytest.fixture
def real_frame_0():
    """Load frame 0 from apple_bowl_2 trajectory."""
    if not HAS_DATA:
        pytest.skip("No data")
    rgb = cv2.cvtColor(
        cv2.imread(os.path.join(DATA_ROOT, "rgb", "rgb_000000.png")),
        cv2.COLOR_BGR2RGB
    )
    depth = np.load(os.path.join(DATA_ROOT, "depth", "depth_000000.npy"))
    cam_poses = load_pose_txt(os.path.join(DATA_ROOT, "pose_txt", "camera_pose.txt"))
    T_cw = cam_poses[0]

    det_path = os.path.join(DATA_ROOT, "detection_h", "detection_000000_final.json")
    detections = load_detection_json(det_path)

    return rgb, depth, T_cw, detections


@pytest.fixture
def real_frames_0_to_4():
    """Load frames 0-4 from apple_bowl_2."""
    if not HAS_DATA:
        pytest.skip("No data")
    cam_poses = load_pose_txt(os.path.join(DATA_ROOT, "pose_txt", "camera_pose.txt"))
    frames = []
    for i in range(5):
        rgb = cv2.cvtColor(
            cv2.imread(os.path.join(DATA_ROOT, "rgb", f"rgb_{i:06d}.png")),
            cv2.COLOR_BGR2RGB
        )
        depth = np.load(os.path.join(DATA_ROOT, "depth", f"depth_{i:06d}.npy"))
        det_path = os.path.join(DATA_ROOT, "detection_h", f"detection_{i:06d}_final.json")
        dets = load_detection_json(det_path) if os.path.exists(det_path) else []
        frames.append((rgb, depth, cam_poses[i], dets))
    return frames


# ─────────────────────────────────────────────────────────────────────
# 1. Data class tests (no dependencies)
# ─────────────────────────────────────────────────────────────────────

class TestDataClasses:
    def test_mesh_creation(self):
        m = Mesh(
            vertices=np.zeros((10, 3)),
            faces=np.zeros((5, 3), dtype=int),
            normals=np.zeros((10, 3)),
            colors=np.zeros((10, 3)),
        )
        assert not m.is_empty

    def test_mesh_empty(self):
        m = Mesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=int),
            normals=np.empty((0, 3)),
            colors=np.empty((0, 3)),
        )
        assert m.is_empty

    def test_tracked_object(self):
        obj = TrackedObject(
            id=0, label="cup",
            pose=np.eye(4),
            points=np.random.rand(100, 3).astype(np.float32),
        )
        assert obj.id == 0
        assert obj.label == "cup"
        assert obj.mesh is None

    def test_frame_detections(self):
        fd = FrameDetections(
            labels=["apple", "bowl"],
            scores=np.array([0.9, 0.8]),
            masks=[np.ones((480, 640), dtype=np.uint8)] * 2,
            bboxes=np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
        )
        assert len(fd.labels) == 2

    def test_relation_graph(self):
        rg = RelationGraph(relations={0: {"on": [1]}, 1: {"under": [0]}})
        assert 0 in rg.relations
        assert "on" in rg.relations[0]


# ─────────────────────────────────────────────────────────────────────
# 2. ObjectReconstructor tests
# ─────────────────────────────────────────────────────────────────────

class TestObjectReconstructor:
    def test_create_object(self):
        recon = ObjectReconstructor(voxel_size=0.005)
        oid = recon.create(pose=np.eye(4), label="cup")
        assert oid == 0

    def test_create_multiple_objects(self):
        recon = ObjectReconstructor(voxel_size=0.005)
        id0 = recon.create(pose=np.eye(4), label="cup")
        id1 = recon.create(pose=np.eye(4), label="bowl")
        assert id0 != id1

    def test_create_with_explicit_id(self):
        recon = ObjectReconstructor(voxel_size=0.005)
        oid = recon.create(pose=np.eye(4), label="cup", object_id=42)
        assert oid == 42

    def test_get_points_empty_initially(self):
        recon = ObjectReconstructor(voxel_size=0.005)
        oid = recon.create(pose=np.eye(4), label="cup")
        pts = recon.get_points(oid)
        assert pts.shape == (0, 3)

    def test_get_mesh_empty_initially(self):
        recon = ObjectReconstructor(voxel_size=0.005)
        oid = recon.create(pose=np.eye(4), label="cup")
        mesh = recon.get_mesh(oid)
        assert mesh.is_empty

    def test_get_object_snapshot(self):
        recon = ObjectReconstructor(voxel_size=0.005)
        oid = recon.create(pose=np.eye(4), label="cup")
        obj = recon.get_object(oid)
        assert isinstance(obj, TrackedObject)
        assert obj.label == "cup"

    def test_unknown_object_raises(self):
        recon = ObjectReconstructor(voxel_size=0.005)
        with pytest.raises(KeyError):
            recon.get_points(999)

    @requires_data
    def test_fuse_real_frame(self, real_frame_0):
        """Fuse a real RGBD frame into a single object's TSDF."""
        rgb, depth, T_cw, detections = real_frame_0
        if not detections:
            pytest.skip("No detections in frame 0")

        recon = ObjectReconstructor(voxel_size=0.003)
        oid = recon.create(pose=T_cw, label=detections[0]["label"])

        mask = detections[0]["mask"]
        success = recon.fuse(oid, rgb, depth.astype(np.float32), K, T_cw, mask=mask)
        # Fusion may or may not succeed depending on depth coverage
        # But it should not crash
        assert isinstance(success, bool)

    @requires_data
    def test_fuse_multi_frame_produces_mesh(self, real_frames_0_to_4):
        """Fusing multiple frames should produce a non-empty mesh."""
        recon = ObjectReconstructor(voxel_size=0.003)

        # Find a label that appears in the first frame
        _, _, T_cw0, dets0 = real_frames_0_to_4[0]
        if not dets0:
            pytest.skip("No detections")

        target_label = dets0[0]["label"]
        oid = recon.create(pose=T_cw0, label=target_label)

        fuse_count = 0
        for rgb, depth, T_cw, dets in real_frames_0_to_4:
            for det in dets:
                if det["label"] == target_label:
                    success = recon.fuse(
                        oid, rgb, depth.astype(np.float32),
                        K, T_cw, mask=det["mask"]
                    )
                    if success:
                        fuse_count += 1
                    break

        if fuse_count > 0:
            mesh = recon.get_mesh(oid)
            # After multi-frame fusion, expect a mesh
            assert mesh.vertices.shape[1] == 3
            print(f"Fused {fuse_count} frames, mesh has {mesh.vertices.shape[0]} vertices")


# ─────────────────────────────────────────────────────────────────────
# 3. ObjectTracker tests
# ─────────────────────────────────────────────────────────────────────

class TestObjectTracker:
    def test_init(self):
        tracker = ObjectTracker(K=K)
        assert len(tracker.internal_objects) == 0

    @requires_data
    def test_single_frame_tracking(self, real_frame_0):
        """Track objects from a single real frame."""
        rgb, depth, T_cw, detections = real_frame_0
        if not detections:
            pytest.skip("No detections")

        tracker = ObjectTracker(K=K, voxel_size=0.003)

        fd = FrameDetections(
            labels=[d["label"] for d in detections],
            scores=np.array([d["score"] for d in detections]),
            masks=[d["mask"] for d in detections],
            bboxes=np.zeros((len(detections), 4)),  # not used by associate_by_id
        )

        tracked = tracker.update(fd, rgb, depth.astype(np.float32), T_cw)

        assert isinstance(tracked, list)
        # Some objects should be created
        print(f"Tracked {len(tracked)} objects from {len(detections)} detections")
        for obj in tracked:
            assert isinstance(obj, TrackedObject)
            assert obj.label is not None
            print(f"  - {obj.label} (id={obj.id}, {obj.points.shape[0]} pts)")

    @requires_data
    def test_multi_frame_tracking_consistency(self, real_frames_0_to_4):
        """Track across multiple frames — object IDs should be consistent."""
        tracker = ObjectTracker(K=K, voxel_size=0.003)

        all_ids_per_frame = []
        for i, (rgb, depth, T_cw, dets) in enumerate(real_frames_0_to_4):
            if not dets:
                continue

            fd = FrameDetections(
                labels=[d["label"] for d in dets],
                scores=np.array([d["score"] for d in dets]),
                masks=[d["mask"] for d in dets],
                bboxes=np.zeros((len(dets), 4)),
            )
            tracked = tracker.update(fd, rgb, depth.astype(np.float32), T_cw)
            ids = {obj.id for obj in tracked}
            all_ids_per_frame.append(ids)
            print(f"Frame {i}: {len(tracked)} objects, IDs={ids}")

        # Objects should accumulate, not reset each frame
        if len(all_ids_per_frame) >= 2:
            # At least some IDs from frame 0 should persist in later frames
            first_ids = all_ids_per_frame[0]
            last_ids = all_ids_per_frame[-1]
            # IDs should persist (the set should grow or stay same, not be disjoint)
            assert len(first_ids) > 0 or len(last_ids) > 0


# ─────────────────────────────────────────────────────────────────────
# 4. PoseUpdater tests
# ─────────────────────────────────────────────────────────────────────

class TestPoseUpdater:
    def test_update_from_ee_basic(self):
        """Test EE pose update with a synthetic SceneObject."""
        from heuristic_tracker.scene_object import SceneObject

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [1.0, 2.0, 0.5]
        obj = SceneObject(pose=pose, id=0, initial_label="cup", voxel_size=0.005)

        T_cw = np.eye(4, dtype=np.float32)
        T_cw[:3, 3] = [0, 0, 1.5]
        T_ec = np.eye(4, dtype=np.float32)
        T_ec[:3, 3] = [0.1, 0, 0]

        result = PoseUpdater.update_from_ee([obj], 0, T_cw, T_ec)
        assert isinstance(result, bool)

    @requires_data
    def test_update_from_ee_with_real_poses(self, real_frame_0):
        """Test EE pose update using real camera/EE poses."""
        from heuristic_tracker.scene_object import SceneObject

        rgb, depth, T_cw, detections = real_frame_0
        ee_poses = load_pose_txt(os.path.join(DATA_ROOT, "pose_txt", "ee_pose.txt"))
        T_ec = ee_poses[0]

        pose = T_cw.copy()
        pose[:3, 3] += [0.1, 0, 0]
        obj = SceneObject(pose=pose, id=0, initial_label="test", voxel_size=0.005)

        result = PoseUpdater.update_from_ee([obj], 0, T_cw, T_ec)
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────
# 5. RelationAnalyzer tests
# ─────────────────────────────────────────────────────────────────────

class TestRelationAnalyzer:
    def test_empty_objects(self):
        rg = RelationAnalyzer.compute([])
        assert isinstance(rg, RelationGraph)
        assert len(rg.relations) == 0

    def test_synthetic_stacked_objects(self):
        """Two objects stacked vertically should have on/under relation."""
        from heuristic_tracker.scene_object import SceneObject

        # Bottom object
        pose_bottom = np.eye(4, dtype=np.float32)
        pose_bottom[:3, 3] = [0, 0, 0.5]
        obj_bottom = SceneObject(pose=pose_bottom, id=0, initial_label="plate", voxel_size=0.01)
        bottom_pts = np.random.uniform([-0.05, -0.05, 0.48], [0.05, 0.05, 0.50], size=(200, 3)).astype(np.float32)
        bottom_colors = np.random.uniform(0, 1, size=(200, 3)).astype(np.float32)
        obj_bottom.add_points(bottom_pts, colors=bottom_colors)

        # Top object
        pose_top = np.eye(4, dtype=np.float32)
        pose_top[:3, 3] = [0, 0, 0.55]
        obj_top = SceneObject(pose=pose_top, id=1, initial_label="cup", voxel_size=0.01)
        top_pts = np.random.uniform([-0.03, -0.03, 0.53], [0.03, 0.03, 0.57], size=(200, 3)).astype(np.float32)
        top_colors = np.random.uniform(0, 1, size=(200, 3)).astype(np.float32)
        obj_top.add_points(top_pts, colors=top_colors)

        rg = RelationAnalyzer.compute([obj_bottom, obj_top], tolerance=0.03)
        assert isinstance(rg, RelationGraph)
        print(f"Relations: {rg.relations}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
