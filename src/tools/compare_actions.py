#!/usr/bin/env python
"""
Compare actions from different policy types (e.g., 'pi05' vs 'smolvla') in the same plot.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import draccus
import matplotlib.pyplot as plt
import numpy as np
from rich import print


def resolve_file_case_insensitive(data_path: Path, filename: str) -> Path | None:
    """Return a matching file path even when filename case differs."""
    exact_match = data_path / filename
    if exact_match.exists():
        return exact_match

    target_name = filename.lower()
    for candidate in data_path.glob("*.npy"):
        if candidate.name.lower() == target_name:
            return candidate
    return None


def extract_repo_name_from_filename(filename: str, file_stem: str, episode: int, tail: str) -> str | None:
    """Extract repo name from files like '<stem>_<repo>_episode_<N>_<tail>.npy'."""
    pattern = rf"^{re.escape(file_stem)}_(.+)_episode_{episode}_{re.escape(tail)}\\.npy$"
    match = re.match(pattern, filename)
    if match is None:
        return None
    return match.group(1)


def compute_action_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """
    Compute a bounded similarity score in [0, 1] for two aligned action sequences.

    We use an NRMSE-based score because action trajectories are time-aligned and we care
    about absolute numerical closeness, not just shape correlation.
    """
    valid_steps = min(len(reference), len(candidate))
    if valid_steps == 0:
        return float("nan")

    ref = reference[:valid_steps]
    cand = candidate[:valid_steps]
    rmse = np.sqrt(np.mean((ref - cand) ** 2))

    # Normalize by the teleop range to make scores comparable across dimensions.
    scale = np.ptp(ref)
    if scale < 1e-8:
        scale = max(np.max(np.abs(ref)), 1.0)

    nrmse = rmse / scale
    return 1.0 / (1.0 + nrmse)


def compute_eef_trajectory_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute similarity for aligned 3D EEF trajectories using normalized pointwise RMSE."""
    valid_steps = min(len(reference), len(candidate))
    if valid_steps == 0:
        return float("nan")

    ref = np.asarray(reference[:valid_steps], dtype=np.float64).reshape(valid_steps, 3)
    cand = np.asarray(candidate[:valid_steps], dtype=np.float64).reshape(valid_steps, 3)
    pointwise_error = np.linalg.norm(ref - cand, axis=1)
    rmse = np.sqrt(np.mean(pointwise_error**2))

    ref_extent = np.ptp(ref, axis=0)
    scale = np.linalg.norm(ref_extent)
    if scale < 1e-8:
        scale = max(np.max(np.linalg.norm(ref, axis=1)), 1.0)

    nrmse = rmse / scale
    return 1.0 / (1.0 + nrmse)


def plot_series_comparison(
    reference: np.ndarray,
    policy_results: list[dict],
    title_prefix: str,
    output_file: Path,
    ylim: tuple[float, float] | None = None,
    reference_label: str = "Teleop",
    dimension_labels: list[str] | None = None,
) -> None:
    """Plot aligned reference/policy trajectories for each dimension and save the figure."""
    series_dim = reference.shape[1]
    ncols = 2 if series_dim > 1 else 1
    nrows = int(np.ceil(series_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(series_dim):
        axes[i].plot(reference[:, i], label=reference_label, linestyle="--", alpha=0.7, linewidth=1.5)

    best_policy_by_dim = {}
    for i in range(series_dim):
        candidates = [
            (result["policy_type"], result["similarities"][i])
            for result in policy_results
            if i < len(result["similarities"]) and np.isfinite(result["similarities"][i])
        ]
        if candidates:
            best_policy_by_dim[i] = max(candidates, key=lambda item: item[1])[0]

    for result in policy_results:
        policy_type = result["policy_type"]
        policy_series = result["series"]
        color = result["color"]
        for i in range(min(series_dim, policy_series.shape[1])):
            similarity = result["similarities"][i]
            similarity_text = f"sim={similarity:.3f}" if np.isfinite(similarity) else "sim=n/a"
            best_marker = " * " if best_policy_by_dim.get(i) == policy_type else ""
            axes[i].plot(
                policy_series[:, i],
                label=f"Policy ({policy_type}, {similarity_text}){best_marker}",
                alpha=0.7,
                color=color,
            )

    for i in range(series_dim):
        title = dimension_labels[i] if dimension_labels is not None else f"Dimension {i}"
        axes[i].set_title(title)
        if ylim is not None:
            axes[i].set_ylim(*ylim)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    for i in range(series_dim, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title_prefix)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[bright_green]saved to {output_file}[/bright_green]")


def plot_eef_comparison_combined(
    teleop_eefs: np.ndarray,
    eef_policy_results: list[dict],
    title_prefix: str,
    output_file: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot left/right EEF 3D trajectories and xyz time-series in one combined figure."""
    fig = plt.figure(figsize=(16, 10))
    left_3d_ax = fig.add_subplot(2, 2, 1, projection="3d")
    right_3d_ax = fig.add_subplot(2, 2, 2, projection="3d")
    left_2d_ax = fig.add_subplot(2, 2, 3)
    right_2d_ax = fig.add_subplot(2, 2, 4)

    teleop_left = teleop_eefs[:, 0:3]
    teleop_right = teleop_eefs[:, 3:6]

    left_3d_ax.plot(
        teleop_left[:, 0], teleop_left[:, 1], teleop_left[:, 2], label="Teleop", linestyle="--", linewidth=2
    )
    right_3d_ax.plot(
        teleop_right[:, 0],
        teleop_right[:, 1],
        teleop_right[:, 2],
        label="Teleop",
        linestyle="--",
        linewidth=2,
    )

    left_3d_ax.scatter(
        teleop_left[0, 0], teleop_left[0, 1], teleop_left[0, 2], marker="o", s=40, label="Teleop start"
    )
    right_3d_ax.scatter(
        teleop_right[0, 0], teleop_right[0, 1], teleop_right[0, 2], marker="o", s=40, label="Teleop start"
    )

    xyz_labels = ["X", "Y", "Z"]
    xyz_colors = ["tab:red", "tab:green", "tab:blue"]
    for dim_idx, (label, color) in enumerate(zip(xyz_labels, xyz_colors, strict=True)):
        left_2d_ax.plot(
            teleop_left[:, dim_idx],
            label=f"Teleop {label}",
            linestyle="--",
            linewidth=1.8,
            color=color,
            alpha=0.85,
        )
        right_2d_ax.plot(
            teleop_right[:, dim_idx],
            label=f"Teleop {label}",
            linestyle="--",
            linewidth=1.8,
            color=color,
            alpha=0.85,
        )

    for result in eef_policy_results:
        policy_type = result["policy_type"]
        policy_eefs = result["series"]
        policy_left = policy_eefs[:, 0:3]
        policy_right = policy_eefs[:, 3:6]
        left_similarity = compute_eef_trajectory_similarity(teleop_left, policy_left)
        right_similarity = compute_eef_trajectory_similarity(teleop_right, policy_right)
        left_similarity_text = f"sim={left_similarity:.3f}" if np.isfinite(left_similarity) else "sim=n/a"
        right_similarity_text = f"sim={right_similarity:.3f}" if np.isfinite(right_similarity) else "sim=n/a"

        left_3d_ax.plot(
            policy_left[:, 0],
            policy_left[:, 1],
            policy_left[:, 2],
            label=f"Policy {policy_type} ({left_similarity_text})",
            color=result["color"],
            linewidth=2,
        )
        right_3d_ax.plot(
            policy_right[:, 0],
            policy_right[:, 1],
            policy_right[:, 2],
            label=f"Policy {policy_type} ({right_similarity_text})",
            color=result["color"],
            linewidth=2,
        )

        for dim_idx, (label, color) in enumerate(zip(xyz_labels, xyz_colors, strict=True)):
            left_2d_ax.plot(
                policy_left[:, dim_idx],
                label=f"{policy_type} {label} ({left_similarity_text})",
                linewidth=1.5,
                color=color,
                alpha=0.55,
            )
            right_2d_ax.plot(
                policy_right[:, dim_idx],
                label=f"{policy_type} {label} ({right_similarity_text})",
                linewidth=1.5,
                color=color,
                alpha=0.55,
            )

    for axis, axis_title in (
        (left_3d_ax, "Left EEF 3D Trajectory"),
        (right_3d_ax, "Right EEF 3D Trajectory"),
    ):
        axis.set_title(axis_title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.grid(True, alpha=0.3)
        axis.legend()

    for axis, axis_title in ((left_2d_ax, "Left EEF XYZ"), (right_2d_ax, "Right EEF XYZ")):
        axis.set_title(axis_title)
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Position")
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(True, alpha=0.3)
        axis.legend(ncols=2, fontsize=8)

    plt.suptitle(title_prefix)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[bright_green]saved to {output_file}[/bright_green]")


def compare_actions(
    data_dir: str,
    repo_id: str,
    episode: int,
    policy_types: list[str],
    output_dir: str = "outputs",
    ylim: tuple[float, float] | None = None,
):
    """
    Compare actions from different policy types in the same plot.

    Args:
        data_dir: Directory containing saved action files
        repo_id: Repository ID used during replay
        episode: Episode index to compare
        policy_types: List of policy types to compare (e.g., ['zmq', 'agg'])
        output_dir: Directory to save comparison plots
        ylim: Y-axis limits as (ymin, ymax). Pass None to use auto-scaling.
    """
    repo_name = repo_id.replace("/", "_")
    data_path = Path(data_dir)
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]

    # Load teleop actions once (they should be the same regardless of policy type)
    teleop_file = resolve_file_case_insensitive(
        data_path, f"teleop_actions_{repo_name}_episode_{episode}_tel.npy"
    )
    if teleop_file is None:
        # Try with the first policy type in the list
        teleop_file = resolve_file_case_insensitive(
            data_path, f"teleop_actions_{repo_name}_episode_{episode}_{policy_types[0]}.npy"
        )

    if teleop_file is None:
        # Fall back to any repo prefix for this episode.
        fallback_teleop_files = sorted(data_path.glob(f"teleop_actions_*_episode_{episode}_tel.npy"))
        if fallback_teleop_files:
            teleop_file = fallback_teleop_files[0]
            detected_repo_name = extract_repo_name_from_filename(
                teleop_file.name,
                file_stem="teleop_actions",
                episode=episode,
                tail="tel",
            )
            if detected_repo_name is not None:
                repo_name = detected_repo_name
            print(f"[yellow]repo_id prefix mismatch; using detected teleop file:[/yellow] {teleop_file.name}")

    if teleop_file is None:
        print(
            "Teleop actions file not found: "
            f"teleop_actions_{repo_name}_episode_{episode}_tel.npy or "
            f"teleop_actions_{repo_name}_episode_{episode}_{policy_types[0]}.npy"
        )
        return

    teleop_actions = np.load(teleop_file)

    action_dim = teleop_actions.shape[1]
    policy_results = []
    for idx, policy_type in enumerate(policy_types):
        policy_file = resolve_file_case_insensitive(
            data_path, f"policy_actions_{repo_name}_episode_{episode}_{policy_type}.npy"
        )

        if policy_file is None:
            fallback_policy_files = sorted(
                data_path.glob(f"policy_actions_*_episode_{episode}_{policy_type}.npy")
            )
            if fallback_policy_files:
                policy_file = fallback_policy_files[0]
                print(f"[yellow]using fallback policy actions file:[/yellow] {policy_file.name}")
            else:
                print(
                    f"Policy actions file not found: policy_actions_{repo_name}_episode_{episode}_{policy_type}.npy"
                )
                continue

        policy_actions = np.load(policy_file)
        similarities = [
            compute_action_similarity(teleop_actions[:, i], policy_actions[:, i])
            for i in range(min(action_dim, policy_actions.shape[1]))
        ]

        policy_results.append(
            {
                "policy_type": policy_type,
                "series": policy_actions,
                "color": colors[idx % len(colors)],
                "similarities": similarities,
            }
        )
    output_path = Path(output_dir)
    save_file = (
        output_path / f"action_comparison_{repo_name}_episode_{episode}_{'_vs_'.join(policy_types)}.png"
    )
    plot_series_comparison(
        reference=teleop_actions,
        policy_results=policy_results,
        title_prefix=(
            f"Action Comparison: {repo_name} Episode {episode}\n"
            f"(Teleop vs {', '.join(policy_types)} policies, similarity=1/(1+NRMSE))"
        ),
        output_file=save_file,
        ylim=ylim,
        reference_label="Teleop",
        dimension_labels=[f"Action Dimension {i}" for i in range(action_dim)],
    )

    teleop_eef_file = resolve_file_case_insensitive(
        data_path, f"teleop_eef_{repo_name}_episode_{episode}_tel.npy"
    )
    if teleop_eef_file is None:
        fallback_teleop_eef_files = sorted(data_path.glob(f"teleop_eef_*_episode_{episode}_tel.npy"))
        if fallback_teleop_eef_files:
            teleop_eef_file = fallback_teleop_eef_files[0]
            print(f"[yellow]using fallback teleop EEF file:[/yellow] {teleop_eef_file.name}")
        else:
            return

    teleop_eefs = np.load(teleop_eef_file)
    eef_policy_results = []
    for idx, policy_type in enumerate(policy_types):
        policy_eef_file = resolve_file_case_insensitive(
            data_path, f"policy_eef_{repo_name}_episode_{episode}_{policy_type}.npy"
        )
        if policy_eef_file is None:
            fallback_policy_eef_files = sorted(
                data_path.glob(f"policy_eef_*_episode_{episode}_{policy_type}.npy")
            )
            if fallback_policy_eef_files:
                policy_eef_file = fallback_policy_eef_files[0]
                print(f"[yellow]using fallback policy EEF file:[/yellow] {policy_eef_file.name}")
            else:
                print(
                    f"Policy EEF file not found: policy_eef_{repo_name}_episode_{episode}_{policy_type}.npy"
                )
                continue

        policy_eefs = np.load(policy_eef_file)
        similarities = [
            compute_action_similarity(teleop_eefs[:, i], policy_eefs[:, i])
            for i in range(min(teleop_eefs.shape[1], policy_eefs.shape[1]))
        ]
        eef_policy_results.append(
            {
                "policy_type": policy_type,
                "series": policy_eefs,
                "color": colors[idx % len(colors)],
                "similarities": similarities,
            }
        )

    if not eef_policy_results:
        return

    if teleop_eefs.shape[1] >= 6:
        eef_combined_file = (
            output_path / f"eef_comparison_{repo_name}_episode_{episode}_{'_vs_'.join(policy_types)}.png"
        )
        plot_eef_comparison_combined(
            teleop_eefs=teleop_eefs,
            eef_policy_results=eef_policy_results,
            title_prefix=(
                f"EEF Comparison: {repo_name} Episode {episode}\n(Top: 3D trajectories, Bottom: XYZ time-series)"
            ),
            output_file=eef_combined_file,
            ylim=ylim,
        )


@dataclass
class CompareActionsConfig:
    data_dir: str = "outputs/compare_actions"
    repo_id: str = ""
    episodes: str = "0,1,2"
    # Comma-separated list, e.g. "zmq,agg".
    policy_types: str = "zmq"
    output_dir: str = "outputs"
    ylim_min: float | None = None
    ylim_max: float | None = None


@draccus.wrap()
def main(config: CompareActionsConfig) -> int:
    if not config.repo_id:
        raise ValueError("--repo-id is required")

    policy_types = [part.strip() for part in config.policy_types.split(",") if part.strip()]
    if not policy_types:
        raise ValueError("--policy-types must include at least one policy, e.g. --policy-types pi05,smolvla")

    if (config.ylim_min is None) ^ (config.ylim_max is None):
        raise ValueError("Provide both --ylim-min and --ylim-max together")

    ylim: tuple[float, float] | None = None
    if config.ylim_min is not None and config.ylim_max is not None:
        if config.ylim_min >= config.ylim_max:
            raise ValueError("Expected --ylim-min < --ylim-max")
        ylim = (config.ylim_min, config.ylim_max)

    episodes = [int(ep.strip()) for ep in config.episodes.split(",") if ep.strip()]
    for episode in episodes:
        compare_actions(
            data_dir=config.data_dir,
            repo_id=config.repo_id,
            episode=episode,
            policy_types=policy_types,
            output_dir=config.output_dir,
            ylim=ylim,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
