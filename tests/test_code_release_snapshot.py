import subprocess
from zipfile import ZipFile

from scripts.storage import assemble_legal_paper_release as release


def test_code_release_preserves_installation_files_and_excludes_private_workspaces(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    files = {
        "README.md": "Code installation instructions\n",
        "CITATION.cff": "title: TeamGym\n",
        "LICENSE": "TeamGym license\n",
        ".python-version": "3.11\n",
        "uv.lock": "version = 1\n",
        "pyproject.toml": "[build-system]\nrequires = ['setuptools']\n",
        "setup.py": "from setuptools import setup\nsetup()\n",
        "requirements/lightweight-py311.txt": "openai==1.109.1\n",
        "rllm/LICENSE": "Bundled rLLM license\n",
        "rllm/data/datasets/example/train.parquet": "Local example dataset\n",
        "rllm/registry/dataset_registry.json": '{"example": {"train": "/local/train.parquet"}}\n',
        "teamgym/core/schema.py": "SCHEMA_VERSION = 1\n",
        "teamgym/prompts/legal/question.txt": "Question prompt\n",
        "docs/manuscript.tex": "Local manuscript\n",
        "docs/rebuttal/draft.md": "Local reviewer response\n",
        "rebuttal/private_campaign.json": '{"seed": 17}\n',
        "rebuttal/controller.py": "# Private campaign controller\n",
        "evaluation/run.sh": "#!/usr/bin/env bash\n",
        "evaluation/configs/schedules.tsv": "# Reference schedules\n",
        "evaluation/submissions/private_campaign/controller.py": "# Local campaign controller\n",
        "figures/research_plot.py": "# Historical plotting script\n",
        ".vscode/settings.json": "{}\n",
        ".codex": "Local agent settings\n",
        ".local_data/config/machine.env": "# Local machine paths\n",
        ".cursorignore": "Local editor exclusions\n",
        ".latexmkrc": "# Local manuscript settings\n",
        "distributed_reasoning_dataset_plan.md": "Historical research plan\n",
        "trajectories_merged_20260919.csv": "dataset,success\n",
        "decision_option_permutation_map.jsonl": "{}\n",
        "decision_option_permutation_map.pretty.json": "[]\n",
        "decision_option_permutation_report.json": "{}\n",
        "teamgym/rl/paper.py": "# Public training entry point\n",
        "teamgym/rl/train_backend.sh": "#!/usr/bin/env bash\n",
        "teamgym/rl/campaign_20260918_finance_final_h200_jobs.json": "{}\n",
        "teamgym/rl/CUAD_WRITTEN_DATASET_LOG.md": "Historical dataset log\n",
        "teamgym/rl/QUERY_PRIVATE_EXPERIMENT_LOG.md": "Historical experiment log\n",
        "teamgym/rl/run_query_private_32token_training_h200.sh": "# Retired launcher\n",
        "teamgym/rl/analysis/old_run/summary.json": "{}\n",
        "teamgym/rl/rollout/workflow.py": "# Public rollout runtime\n",
        "teamgym/rl/tools/plot_reward.py": "# Public training reward plots\n",
        "teamgym/rl/analyze_probing.py": "# Retired research analysis\n",
        "teamgym/rl/compare_probing_runs.py": "# Retired comparison\n",
        "teamgym/rl/plot_query_echo_trajectory.py": "# Retired research plot\n",
        "teamgym/rl/run_finance_final_policy_cross_replay.py": "# Retired replay\n",
        "teamgym/rl/build_participant_replay_bank.py": "# Retired replay preparation\n",
        "evaluation/reporting/render_taekyun_2_reports.py": "# Retired batch renderer\n",
        "evaluation/reporting/translate_taekyun_2_trajectories.py": "# Retired label translator\n",
        "evaluation/reporting/visualize_taekyun_2_eval.py": "# Retired legacy plots\n",
        "teamgym/visualization/interaction_trajectory_report.py": "# Public trajectory renderer\n",
        "scripts/storage/build_legal_allocated_archive.py": "# Public dataset archive builder\n",
        "scripts/storage/stage_legal_data.py": "# Public archive staging utility\n",
        "scripts/storage/build_legal_allocated_archive.sbatch": "# Historical cluster submission\n",
        "scripts/storage/migrate_pool_artifacts_to_scratch_0714.sh": "# Historical storage migration\n",
        "scripts/data_generation/slurm/legal_written_construction/generate.sbatch": "# Public construction submission\n",
        "evaluation/reporting/rescore_numeric_short_answers.py": "# Public numeric rescoring utility\n",
        "teamgym/data/long_context_reasoning/visualize.py": "# Public dataset visualization utility\n",
        "evaluation/reporting/analyze_legal_communication_process.py": "# Retired experiment analysis\n",
        "evaluation/reporting/render_legal_dataset_statistics.py": "# Retired appendix wrapper\n",
        "evaluation/reporting/legal_problem_instance_piu_distribution_statistics.json": "{}\n",
        "tests/test_legal_communication_process.py": "# Retired analysis test\n",
    }
    for name, contents in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    monkeypatch.setattr(
        release.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    # The path filter must exclude local workspaces even if a future caller
    # accidentally includes them among the directories to export.
    monkeypatch.setattr(release, "CODE_ROOTS", (*release.CODE_ROOTS, "docs", "rebuttal", "figures", ".vscode", ".local_data"))
    private_files = {
        ".codex", ".cursorignore", ".latexmkrc",
        ".local_data/config/machine.env",
        "rllm/data/datasets/example/train.parquet",
        "rllm/registry/dataset_registry.json",
        "distributed_reasoning_dataset_plan.md", "trajectories_merged_20260919.csv",
        "decision_option_permutation_map.jsonl", "decision_option_permutation_map.pretty.json",
        "decision_option_permutation_report.json",
        "teamgym/rl/campaign_20260918_finance_final_h200_jobs.json",
        "teamgym/rl/CUAD_WRITTEN_DATASET_LOG.md",
        "teamgym/rl/QUERY_PRIVATE_EXPERIMENT_LOG.md",
        "teamgym/rl/run_query_private_32token_training_h200.sh",
        "teamgym/rl/analyze_probing.py",
        "teamgym/rl/compare_probing_runs.py",
        "teamgym/rl/plot_query_echo_trajectory.py",
        "teamgym/rl/run_finance_final_policy_cross_replay.py",
        "teamgym/rl/build_participant_replay_bank.py",
        "evaluation/reporting/render_taekyun_2_reports.py",
        "evaluation/reporting/translate_taekyun_2_trajectories.py",
        "evaluation/reporting/visualize_taekyun_2_eval.py",
        "scripts/storage/build_legal_allocated_archive.sbatch",
        "scripts/storage/migrate_pool_artifacts_to_scratch_0714.sh",
        "evaluation/reporting/analyze_legal_communication_process.py",
        "evaluation/reporting/render_legal_dataset_statistics.py",
        "evaluation/reporting/legal_problem_instance_piu_distribution_statistics.json",
        "tests/test_legal_communication_process.py",
    }
    monkeypatch.setattr(release, "ROOT_FILES", (*release.ROOT_FILES, *sorted(private_files)))
    output = tmp_path / "code_snapshot.zip"
    release.build_code_snapshot(repo, output)

    with ZipFile(output) as archive:
        private_prefixes = (
            "docs/", "rebuttal/", "evaluation/submissions/", "figures/", ".vscode/", ".local_data/",
            "teamgym/rl/analysis/",
        )
        assert not any(name.startswith(private_prefixes) for name in archive.namelist())
        assert private_files.isdisjoint(archive.namelist())
        for name, contents in files.items():
            assert (repo / name).read_text() == contents
            if name not in private_files and not name.startswith(private_prefixes):
                assert archive.read(name).decode() == contents
