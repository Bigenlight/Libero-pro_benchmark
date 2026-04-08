#!/usr/bin/env python3
"""LIBERO-PRO OOD 환경 영상 녹화 스크립트."""
import os, sys, numpy as np

def main():
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    from libero.libero.utils.video_utils import VideoWriter

    out_dir = "/workspace/LIBERO-PRO/test_outputs/pro_videos"
    os.makedirs(out_dir, exist_ok=True)

    # 녹화할 OOD suite 목록
    suites = [
        ("libero_goal_swap", "Goal Swap"),
        ("libero_spatial_object", "Spatial Object"),
        ("libero_goal_lan", "Goal Language"),
        ("libero_10_swap", "LIBERO-10 Swap"),
    ]

    bd = benchmark.get_benchmark_dict()
    n_frames = 60  # 2초 (30fps)

    for suite_name, label in suites:
        print(f"\n=== {label} ({suite_name}) ===")
        suite = bd[suite_name]()
        bddl_path = suite.get_task_bddl_file_path(0)
        task_name = suite.get_task_names()[0]
        init_states = suite.get_task_init_states(0)
        print(f"  Task: {task_name}")

        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
        try:
            obs = env.set_init_state(init_states[0])
            obs = env.reset()

            video_path = os.path.join(out_dir, suite_name)
            with VideoWriter(video_path, save_video=True, fps=30) as writer:
                writer.append_image(obs["agentview_image"][::-1])
                for i in range(n_frames - 1):
                    # 랜덤 action으로 로봇이 움직이게
                    action = np.random.uniform(-0.3, 0.3, env.env.action_dim)
                    action[-1] = 0  # gripper는 고정
                    obs, reward, done, info = env.step(action)
                    writer.append_image(obs["agentview_image"][::-1])

            size = os.path.getsize(os.path.join(video_path, "video.mp4"))
            print(f"  Saved: {video_path}/video.mp4 ({size/1024:.0f} KB, {n_frames} frames)")
        finally:
            env.close()

    print(f"\nDone! Videos in: {out_dir}")

if __name__ == "__main__":
    main()
