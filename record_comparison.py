#!/usr/bin/env python3
"""LIBERO 원본 vs LIBERO-PRO OOD 비교 영상 녹화."""
import os, numpy as np

def record(suite_name, label, out_dir, seed=42):
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    from libero.libero.utils.video_utils import VideoWriter

    bd = benchmark.get_benchmark_dict()
    suite = bd[suite_name]()
    bddl_path = suite.get_task_bddl_file_path(0)
    task_name = suite.get_task_names()[0]
    init_states = suite.get_task_init_states(0)

    print(f"  [{label}] {suite_name} — {task_name}")

    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    try:
        obs = env.set_init_state(init_states[0])
        obs = env.reset()

        np.random.seed(seed)
        video_path = os.path.join(out_dir, f"{suite_name}")
        n_frames = 90  # 3초

        with VideoWriter(video_path, save_video=True, fps=30) as writer:
            writer.append_image(obs["agentview_image"][::-1])
            for _ in range(n_frames - 1):
                action = np.random.uniform(-0.3, 0.3, env.env.action_dim)
                action[-1] = 0
                obs, _, _, _ = env.step(action)
                writer.append_image(obs["agentview_image"][::-1])

        size = os.path.getsize(os.path.join(video_path, "video.mp4"))
        print(f"    -> {video_path}/video.mp4 ({size/1024:.0f} KB)")
    finally:
        env.close()

def main():
    out_dir = "/workspace/LIBERO-PRO/test_outputs/comparison"
    os.makedirs(out_dir, exist_ok=True)

    # (원본, OOD, 비교 설명)
    pairs = [
        ("libero_goal",    "libero_goal_swap",    "위치 교환 (Swap)"),
        ("libero_spatial", "libero_spatial_object","객체 외형 교체 (Object)"),
        ("libero_10",      "libero_10_swap",      "LIBERO-10 위치 교환 (Swap)"),
    ]

    for original, ood, desc in pairs:
        print(f"\n=== {desc} ===")
        record(original, "Original", out_dir, seed=42)
        record(ood,      "OOD",      out_dir, seed=42)

    print(f"\nDone! Compare side by side:")
    for original, ood, desc in pairs:
        print(f"  {desc}:")
        print(f"    원본: {out_dir}/{original}/video.mp4")
        print(f"    OOD:  {out_dir}/{ood}/video.mp4")

if __name__ == "__main__":
    main()
