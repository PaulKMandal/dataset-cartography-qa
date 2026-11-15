from pathlib import Path


def test_release_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "scripts/dynamics_to_cartography.py",
        "scripts/select_qa_subsets.py",
        "scripts/run_experiment_panel.py",
        "scripts/aggregate_metrics.py",
        "scripts/make_paper_figures.py",
        "scripts/run_qa_smoke.sh",
        "scripts/run_full_panel.sh",
    ]:
        assert (root / rel).exists()


def test_shell_scripts_are_executable():
    root = Path(__file__).resolve().parents[1]
    for rel in ["scripts/run_qa_smoke.sh", "scripts/run_full_panel.sh"]:
        assert (root / rel).stat().st_mode & 0o111
