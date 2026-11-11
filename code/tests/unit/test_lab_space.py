import io
import json
import sys
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import lab_space as ls


def write_nemo(path: Path, payload: bytes = b"w" * 4096, keep: int | None = None) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tar:
        info = tarfile.TarInfo("./model_weights.ckpt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    data = buf.getvalue()
    path.write_bytes(data if keep is None else data[:keep])


def make_model(app: Path, name: str, status: str | None = "complete", ckpt: int = 2048, nemo: bool = True,
               keep: int | None = None) -> Path:
    d = app / "models" / name
    d.mkdir(parents=True)
    if nemo:
        write_nemo(d / "parakeet-ft.nemo", keep=keep)
    if status:
        (d / "run.json").write_text(json.dumps({"status": status}))
    if ckpt:
        (d / "ckpt" / "dev").mkdir(parents=True)
        (d / "ckpt" / "dev" / "best.ckpt").write_bytes(b"c" * ckpt)
    return d


def score(app: Path, name: str) -> None:
    d = app / "reports" / "finetune-torgo" / f"{name}-test-nemo"
    d.mkdir(parents=True)
    (d / "results.json").write_text("{}")
    (d / "hypotheses.jsonl").write_text("{}\n")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "APP", tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def roles(app: Path) -> dict[str, str]:
    return {m.rel: m.role for m in ls.scan_models()}


def test_a_cut_off_nemo_is_not_intact(app):
    write_nemo(app / "ok.nemo")
    write_nemo(app / "cut.nemo", keep=1500)
    assert ls.nemo_intact(app / "ok.nemo")
    assert not ls.nemo_intact(app / "cut.nemo")
    (app / "junk.nemo").write_bytes(b"not a tar")
    assert not ls.nemo_intact(app / "junk.nemo")


def test_roles(app):
    make_model(app, "parakeet-ft-lab-07-v2-M02")
    score(app, "lab-07-v2-M02")
    make_model(app, "parakeet-ft-f2-M02-adapter8")
    score(app, "f2-M02-adapter8")
    make_model(app, "parakeet-ft-f2-M01-bottom8-5epochs")
    make_model(app, "parakeet-ft-f2-M04-adapter8")
    make_model(app, "parakeet-ft-lab-09-b2-M01", status="running", nemo=False)
    make_model(app, "parakeet-ft-lab-09-b4-M01", keep=1500)
    make_model(app, "parakeet-ft-prototype-severe-focus-bottom8")
    (app / "models" / "parakeet-tdt-0.6b-v2-int8").mkdir()
    make_model(app, "lab-baselines/whole-block-M01")
    r = roles(app)
    assert r["parakeet-ft-lab-07-v2-M02"] == "HOLD"
    assert r["parakeet-ft-f2-M02-adapter8"] == "FINISHED"
    assert r["parakeet-ft-f2-M01-bottom8-5epochs"] == "PROTECTED"
    assert r["parakeet-ft-prototype-severe-focus-bottom8"] == "PROTECTED"
    assert r["lab-baselines/whole-block-M01"] == "PROTECTED"
    assert r["parakeet-ft-f2-M04-adapter8"] == "UNSCORED"
    assert r["parakeet-ft-lab-09-b2-M01"] == "INCOMPLETE"
    assert r["parakeet-ft-lab-09-b4-M01"] == "INCOMPLETE"
    assert r["parakeet-tdt-0.6b-v2-int8"] == "OTHER"
    (app / "reports" / "lab-15-int8-screen").mkdir(parents=True)
    (app / "reports" / "lab-15-int8-screen" / "screen.json").write_text("{}")
    assert roles(app)["parakeet-ft-lab-07-v2-M02"] == "FINISHED"


def test_decode_names_do_not_borrow_each_others_scores(app):
    make_model(app, "parakeet-ft-f2-M02-adapter8")
    score(app, "f2-M02-adapter8-seed18")
    assert roles(app)["parakeet-ft-f2-M02-adapter8"] == "UNSCORED"


def test_clean_removes_only_ckpt_of_intact_finished_runs(app, capsys):
    done = make_model(app, "parakeet-ft-f2-M02-adapter8")
    running = make_model(app, "parakeet-ft-lab-09-b2-M01", status="running", nemo=False)
    cut = make_model(app, "parakeet-ft-lab-09-b4-M01", keep=1500)
    proto = make_model(app, "parakeet-ft-prototype-x")
    args = Namespace(apply=False, also_package_caches=False)
    ls.cmd_clean(args)
    assert (done / "ckpt").exists()
    args.apply = True
    ls.cmd_clean(args)
    assert not (done / "ckpt").exists() and (done / "parakeet-ft.nemo").exists()
    assert not (proto / "ckpt").exists() and (proto / "parakeet-ft.nemo").exists()
    assert (running / "ckpt").exists() and (cut / "ckpt").exists()


def test_pack_unpack_round_trip_and_tamper_detection(app, tmp_path):
    ft = app / "reports" / "finetune-torgo" / "x-test-nemo"
    ft.mkdir(parents=True)
    (ft / "results.json").write_text('{"wer": 0.5}')
    (ft / "report.html").write_text("<html>")
    (app / "reports" / "lab-01-a").mkdir()
    (app / "reports" / "lab-01-a" / "notes.md").write_text("tracked")
    (app / "reports" / "lab-01-a" / "table.json").write_text("{}")
    (app / "reports" / "private-thing").mkdir()
    (app / "reports" / "private-thing" / "x.json").write_text("{}")
    (app / "logs").mkdir()
    (app / "logs" / "lab-1003-summaries.txt").write_text("ok")
    (app / "logs" / "server.log").write_text("not a lab log")
    make_model(app, "parakeet-ft-a")
    files, _ = ls.result_files()
    assert files == ["logs/lab-1003-summaries.txt", "models/parakeet-ft-a/run.json",
                     "reports/finetune-torgo/x-test-nemo/results.json", "reports/lab-01-a/table.json"]
    out = app / "transfers" / "b.tgz"
    assert ls.cmd_pack(Namespace(stamp=None, out=str(out))) == 0
    assert (app / "transfers" / "b.tgz.sha256").exists()
    other = tmp_path / "other"
    assert ls.cmd_unpack(Namespace(bundle=str(out), into=str(other))) == 0
    assert (other / "reports/lab-01-a/table.json").read_text() == "{}"
    assert ls.cmd_unpack(Namespace(bundle=str(out), into=str(other))) == 0
    (other / "reports/lab-01-a/table.json").write_text("changed")
    assert ls.cmd_unpack(Namespace(bundle=str(out), into=str(other))) == 1
    assert (other / "reports/lab-01-a/table.json").read_text() == "changed"


def test_bundle_rejects_unsafe_paths_and_damage(app):
    for bad in ("../x", "/etc/passwd", "data/a.json", "reports\\a"):
        with pytest.raises(ValueError):
            ls.safe_member(bad)
    bundle = app / "evil.tgz"
    payload = b"hello"
    manifest = {"files": [{"path": "reports/a.json", "size": 5, "sha256": "0" * 64}]}
    with tarfile.open(bundle, "w:gz") as tar:
        for name, data in (("MANIFEST.json", json.dumps(manifest).encode()), ("reports/a.json", payload)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="damaged"):
        ls.inspect_bundle(bundle)
    assert ls.cmd_unpack(Namespace(bundle=str(bundle), into=str(app / "dest"))) == 1
    assert not (app / "dest").exists() or not any((app / "dest").rglob("*.json"))


def test_git_classification(tmp_path):
    root = tmp_path
    (root / "app/reports/f1").mkdir(parents=True)
    (root / "app/reports/f1/notes.md").write_text("n")
    (root / "app/scripts").mkdir(parents=True)
    (root / "app/scripts/a.py").write_text("x")
    (root / "app/models/m").mkdir(parents=True)
    (root / "app/models/m/parakeet-ft.nemo").write_text("x")
    (root / "app/data").mkdir()
    (root / "app/data/corpus.db").write_text("x")
    (root / "app/reports/f1/big.json").write_bytes(b"x" * (2 * ls.MB))
    (root / "app/src/willspeak/models").mkdir(parents=True)
    (root / "app/src/willspeak/models/proj.npz").write_text("x")
    cases = {
        "app/reports/f1/notes.md": "COMMIT", "app/scripts/a.py": "COMMIT", "app/models/m/parakeet-ft.nemo": "NEVER",
        "app/data/corpus.db": "NEVER", "app/reports/f1/big.json": "REVIEW", "app/src/willspeak/models/proj.npz": "REVIEW"}
    for path, want in cases.items():
        assert ls.classify_git(root, " M", path)[0] == want, path
    assert ls.classify_git(root, " D", "app/scripts/gone.py")[0] == "REVIEW"


def test_archive_copies_verifies_then_removes_only_on_request(app, tmp_path):
    (app / "reports" / "lab-15-int8-screen").mkdir(parents=True)
    (app / "reports" / "lab-15-int8-screen" / "screen.json").write_text("{}")
    a = make_model(app, "parakeet-ft-lab-07-v2-M02")
    score(app, "lab-07-v2-M02")
    dest = tmp_path / "usb"
    ls.cmd_archive(Namespace(to=str(dest), apply=False, remove_after_verify=True))
    assert not dest.exists() and a.exists()
    ls.cmd_archive(Namespace(to=str(dest), apply=True, remove_after_verify=False))
    copy = dest / "models/parakeet-ft-lab-07-v2-M02"
    assert (copy / "parakeet-ft.nemo").read_bytes() == (a / "parakeet-ft.nemo").read_bytes()
    assert not (copy / "ckpt").exists() and a.exists()
    ls.cmd_archive(Namespace(to=str(dest), apply=True, remove_after_verify=True))
    assert not a.exists() and (copy / "parakeet-ft.nemo").exists()


def test_archive_keeps_the_original_when_the_copy_differs(app, tmp_path):
    (app / "reports" / "lab-15-int8-screen").mkdir(parents=True)
    (app / "reports" / "lab-15-int8-screen" / "screen.json").write_text("{}")
    a = make_model(app, "parakeet-ft-lab-07-v2-M02")
    score(app, "lab-07-v2-M02")
    clash = tmp_path / "usb/models/parakeet-ft-lab-07-v2-M02"
    clash.mkdir(parents=True)
    (clash / "parakeet-ft.nemo").write_bytes(b"something else")
    ls.cmd_archive(Namespace(to=str(tmp_path / "usb"), apply=True, remove_after_verify=True))
    assert a.exists()


def git(cwd: Path, *args: str) -> str:
    import subprocess
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": __import__("os").environ["PATH"], "HOME": str(cwd)}
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True, env=env).stdout


def test_results_travel_on_a_side_branch_and_come_back_without_staging(tmp_path, monkeypatch):
    lab = tmp_path / "lab"
    (lab / "app/reports/lab-01-a").mkdir(parents=True)
    (lab / ".gitignore").write_text("app/reports/**\n!app/reports/**/\n!app/reports/**/notes.md\n")
    (lab / "app/reports/lab-01-a/notes.md").write_text("n")
    git(lab, "init", "-q", "-b", "main")
    git(lab, "add", "-A")
    git(lab, "commit", "-q", "-m", "base")
    head = git(lab, "rev-parse", "HEAD")
    (lab / "app/reports/lab-01-a/table.json").write_text('{"a": 1}')
    (lab / "app/reports/lab-01-a/notes.md").write_text("changed notes")
    monkeypatch.setattr(ls, "APP", lab / "app")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    assert ls.cmd_git_results(Namespace(commit=False, stamp="t1", branch=None)) == 0
    assert "lab-results-t1" not in git(lab, "branch")
    assert ls.cmd_git_results(Namespace(commit=True, stamp="t1", branch=None)) == 0
    assert git(lab, "rev-parse", "HEAD") == head and git(lab, "diff", "--cached") == ""
    assert git(lab, "show", "lab-results-t1:app/reports/lab-01-a/table.json") == '{"a": 1}'
    assert "notes.md" in git(lab, "ls-tree", "-r", "--name-only", "lab-results-t1")
    assert git(lab, "show", "lab-results-t1:app/reports/lab-01-a/notes.md") == "n"
    assert (lab / "app/reports/lab-01-a/table.json").exists()
    assert ls.cmd_git_results(Namespace(commit=True, stamp="t1", branch=None)) == 1
    mac = tmp_path / "mac"
    git(tmp_path, "clone", "-q", str(lab), str(mac))
    git(mac, "fetch", "-q", "origin", "lab-results-t1")
    (mac / "app/reports/lab-01-a/notes.md").write_text("my own notes")
    monkeypatch.setattr(ls, "APP", mac / "app")
    assert ls.cmd_fetch_results(Namespace(ref="origin/lab-results-t1", into=None)) == 0
    assert (mac / "app/reports/lab-01-a/table.json").read_text() == '{"a": 1}'
    assert (mac / "app/reports/lab-01-a/notes.md").read_text() == "my own notes"
    assert git(mac, "status", "--short").split() == ["M", "app/reports/lab-01-a/notes.md"]
    assert ls.cmd_fetch_results(Namespace(ref="origin/lab-results-t1", into=None)) == 0
    (mac / "app/reports/lab-01-a/table.json").write_text("different")
    assert ls.cmd_fetch_results(Namespace(ref="origin/lab-results-t1", into=None)) == 1


def test_lab_models_go_only_after_their_decodes_are_in_a_results_branch(tmp_path, monkeypatch):
    lab = tmp_path / "lab"
    app = lab / "app"
    app.mkdir(parents=True)
    (lab / ".gitignore").write_text("app/reports/\napp/models/\n")
    (app / "README.md").write_text("x")
    git(lab, "init", "-q", "-b", "main")
    git(lab, "add", "-A")
    git(lab, "commit", "-q", "-m", "base")
    monkeypatch.setattr(ls, "APP", app)
    for key in ("NAME", "EMAIL"):
        monkeypatch.setenv(f"GIT_AUTHOR_{key}", "t")
        monkeypatch.setenv(f"GIT_COMMITTER_{key}", "t")
    (app / "reports" / "lab-15-int8-screen").mkdir(parents=True)
    (app / "reports" / "lab-15-int8-screen" / "screen.json").write_text("{}")
    model = make_model(app, "parakeet-ft-lab-07-v2-M02")
    keep = make_model(app, "lab-baselines/whole-block-M01")
    score(app, "lab-07-v2-M02")
    args = Namespace(apply=True, also_package_caches=False, lab_models=True)
    ls.cmd_clean(args)
    assert model.exists()
    assert ls.cmd_git_results(Namespace(commit=True, stamp="s", branch=None)) == 0
    ls.cmd_clean(args)
    assert not model.exists() and keep.exists()
    assert (app / "reports/finetune-torgo/lab-07-v2-M02-test-nemo/results.json").exists()


def test_the_size_cap_counts_only_what_is_new_and_splits_it_over_branches(tmp_path, monkeypatch):
    lab = tmp_path / "lab"
    folder = lab / "app/reports/lab-01-a"
    folder.mkdir(parents=True)
    (lab / ".gitignore").write_text("app/reports/**\n!app/reports/**/\n!app/reports/**/notes.md\n")
    (folder / "notes.md").write_text("n")
    git(lab, "init", "-q", "-b", "main")
    git(lab, "add", "-A")
    git(lab, "commit", "-q", "-m", "base")
    monkeypatch.setattr(ls, "APP", lab / "app")
    monkeypatch.setattr(ls, "GIT_RESULT_TOTAL_CAP", 7000)
    for key in ("NAME", "EMAIL"):
        monkeypatch.setenv(f"GIT_AUTHOR_{key}", "t")
        monkeypatch.setenv(f"GIT_COMMITTER_{key}", "t")
    for name in "abc":
        (folder / f"{name}.json").write_text(name * 3000)
    assert ls.cmd_git_results(Namespace(commit=True, stamp="t1", branch=None)) == 0
    branches = git(lab, "branch", "--list", "lab-results-*").split()
    assert branches == ["lab-results-t1-p1of2", "lab-results-t1-p2of2"]
    first = git(lab, "ls-tree", "-r", "--name-only", "lab-results-t1-p1of2").split()
    last = git(lab, "ls-tree", "-r", "--name-only", "lab-results-t1-p2of2").split()
    assert "app/reports/lab-01-a/a.json" in first and "app/reports/lab-01-a/b.json" in first
    assert "app/reports/lab-01-a/c.json" not in first
    assert "app/reports/lab-01-a/c.json" in last and "app/reports/lab-01-a/a.json" in last
    assert ls.cmd_git_results(Namespace(commit=True, stamp="t2", branch=None)) == 0
    assert "lab-results-t2" not in git(lab, "branch")
    assert ls.cmd_git_results(Namespace(commit=True, stamp="t1", branch=None)) == 1
    (folder / "d.json").write_text("d" * 3000)
    assert ls.cmd_git_results(Namespace(commit=True, stamp="t3", branch=None)) == 0
    assert "app/reports/lab-01-a/d.json" in git(lab, "ls-tree", "-r", "--name-only", "lab-results-t3").split()
    assert "app/reports/lab-01-a/a.json" in git(lab, "ls-tree", "-r", "--name-only", "lab-results-t3").split()


def test_sweep_summaries_and_the_sentences_only_table_are_result_folders():
    for name in ("lab-big-l", "lab-big-g2", "sentence-only", "lab-19-int8-cost", "f2-acoustic-only", "finetune-torgo"):
        assert ls.RESULT_REPORT_DIRS.match(name), name
    assert not ls.RESULT_REPORT_DIRS.match("Report Context.md")


def test_personal_device_models_are_protected_and_never_bundled(app):
    make_model(app, "parakeet-ft-a")
    make_model(app, "personal/someone-adapter")
    make_model(app, "personal/severe-base")
    assert roles(app)["personal"] == "PROTECTED"
    files, _ = ls.result_files()
    assert files == ["models/parakeet-ft-a/run.json"]
