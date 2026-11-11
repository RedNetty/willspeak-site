import io
import json
import stat
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/asr"))
import lab_space as LS  # noqa: E402
import model_export as ME  # noqa: E402

FAKE_RCLONE = textwrap.dedent('''\
    #!{python}
    import hashlib, json, os, sys
    root = os.environ["FAKE_RCLONE_ROOT"]
    a = sys.argv[1:]
    def path(spec):
        return os.path.join(root, spec.split(":", 1)[1].lstrip("/"))
    if os.environ.get("FAKE_RCLONE_BROKEN"):
        sys.stderr.write("couldn't login"); sys.exit(3)
    cmd = a[0]
    if cmd == "lsd":
        sys.exit(0)
    if cmd == "rcat":
        target = path(a[-1]); os.makedirs(os.path.dirname(target), exist_ok=True)
        data = sys.stdin.buffer.read()
        if os.environ.get("FAKE_RCLONE_CORRUPT") and target.endswith(".zip"):
            data = data[:-3] + b"xyz"
        open(target, "wb").write(data); sys.exit(0)
    if cmd == "lsjson":
        p = path(a[-1])
        if not os.path.isfile(p):
            sys.stderr.write("directory not found"); sys.exit(3)
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        print(json.dumps([{{"Size": os.path.getsize(p), "Hashes": {{"md5": md5}}}}])); sys.exit(0)
    if cmd == "cat":
        p = path(a[-1])
        if not os.path.isfile(p): sys.exit(3)
        sys.stdout.write(open(p).read()); sys.exit(0)
    if cmd == "copyto":
        import shutil; shutil.copyfile(path(a[1]), a[2]); sys.exit(0)
    if cmd == "lsf":
        print("\\n".join(sorted(os.listdir(path(a[1]))))); sys.exit(0)
    sys.exit(2)
''')


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(LS, "APP", tmp_path / "app")
    monkeypatch.setattr(ME, "APP", tmp_path / "app")
    (tmp_path / "app/models").mkdir(parents=True)
    (tmp_path / "app/reports/lab-15-int8-screen").mkdir(parents=True)
    (tmp_path / "app/reports/lab-15-int8-screen/screen.json").write_text("{}")
    monkeypatch.setattr(LS, "git_head", lambda: "test")
    monkeypatch.setattr(ME.lab_space, "git_head", lambda: "test")
    return tmp_path / "app"


def make_model(app: Path, name: str, scored=True, weights=b"w" * 5000):
    folder = app / "models" / f"parakeet-ft-{name}"
    (folder / "ckpt").mkdir(parents=True)
    (folder / "ckpt/epoch=1.ckpt").write_bytes(b"c" * 3000)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("model_weights.ckpt")
        info.size = len(weights)
        tar.addfile(info, io.BytesIO(weights))
    (folder / "parakeet-ft.nemo").write_bytes(buf.getvalue())
    (folder / "run.json").write_text(json.dumps({"status": "complete"}))
    (folder / "curve.csv").write_text("epoch,wer\n1,0.5\n")
    if scored:
        d = app / "reports/finetune-torgo" / f"{name}-test-nemo"
        d.mkdir(parents=True)
        (d / "results.json").write_text('{"wer": 0.5}')
    return folder


def stored_names(dest: Path):
    return sorted(p.name for p in dest.iterdir())


def test_folder_export_zips_verifies_and_keeps_the_original(app, tmp_path, capsys):
    folder = make_model(app, "lab-09-a-M01")
    dest = tmp_path / "drive"
    assert ME.main(["export", "--to", str(dest), "--apply"]) == 0
    assert stored_names(dest) == ["parakeet-ft-lab-09-a-M01.manifest.json", "parakeet-ft-lab-09-a-M01.zip"]
    manifest = json.loads((dest / "parakeet-ft-lab-09-a-M01.manifest.json").read_text())
    assert set(manifest["files"]) == {"parakeet-ft.nemo", "run.json", "curve.csv"}
    with zipfile.ZipFile(dest / "parakeet-ft-lab-09-a-M01.zip") as zf:
        assert zf.testzip() is None
        assert "parakeet-ft-lab-09-a-M01/parakeet-ft.nemo" in zf.namelist()
    assert (folder / "parakeet-ft.nemo").exists()


def test_dry_run_changes_nothing(app, tmp_path):
    make_model(app, "lab-09-a-M01")
    assert ME.main(["export", "--to", str(tmp_path / "drive")]) == 0
    assert not (tmp_path / "drive").exists()


def test_free_after_verify_removes_only_the_big_files(app, tmp_path):
    folder = make_model(app, "lab-09-a-M01")
    ME.main(["export", "--to", str(tmp_path / "drive"), "--apply", "--free-after-verify"])
    assert sorted(p.name for p in folder.iterdir()) == ["curve.csv", "run.json"]


def test_protected_models_are_uploaded_on_request_but_never_freed(app, tmp_path):
    folder = make_model(app, "f2-M01-bottom8-5epochs")
    ME.main(["export", "--to", str(tmp_path / "d1"), "--apply"])
    assert not (tmp_path / "d1").exists() or not list((tmp_path / "d1").glob("*.zip"))
    ME.main(["export", "--to", str(tmp_path / "d2"), "--apply", "--include-protected", "--free-after-verify"])
    assert list((tmp_path / "d2").glob("*.zip")) and (folder / "parakeet-ft.nemo").exists()


def test_a_second_export_is_a_no_op_and_changed_contents_need_replace(app, tmp_path, capsys):
    folder = make_model(app, "lab-09-a-M01")
    dest = str(tmp_path / "drive")
    ME.main(["export", "--to", dest, "--apply"])
    capsys.readouterr()
    ME.main(["export", "--to", dest, "--apply"])
    assert "already-there" in capsys.readouterr().out
    (folder / "curve.csv").write_text("changed\n")
    assert ME.main(["export", "--to", dest, "--apply", "--free-after-verify"]) == 1
    assert (folder / "parakeet-ft.nemo").exists()
    assert ME.main(["export", "--to", dest, "--apply", "--replace"]) == 0


def test_until_free_gb_stops_early(app, tmp_path, monkeypatch, capsys):
    make_model(app, "lab-09-a-M01")
    make_model(app, "lab-09-b-M01")
    monkeypatch.setattr(ME.shutil, "disk_usage", lambda p: type("U", (), {"free": 500 * 2 ** 30})())
    ME.main(["export", "--to", str(tmp_path / "drive"), "--apply", "--until-free-gb", "100"])
    assert "stopping" in capsys.readouterr().out and not list((tmp_path / "drive").glob("*.zip"))


def test_import_restores_and_checks_checksums(app, tmp_path):
    folder = make_model(app, "lab-09-a-M01")
    dest = str(tmp_path / "drive")
    ME.main(["export", "--to", dest, "--apply", "--free-after-verify"])
    assert not (folder / "parakeet-ft.nemo").exists()
    assert ME.main(["import", "parakeet-ft-lab-09-a-M01", "--from", dest]) == 1
    assert ME.main(["import", "parakeet-ft-lab-09-a-M01", "--from", dest, "--force"]) == 0
    assert LS.nemo_intact(folder / "parakeet-ft.nemo")
    zip_path = tmp_path / "drive/parakeet-ft-lab-09-a-M01.zip"
    data = bytearray(zip_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    zip_path.write_bytes(bytes(data))
    (folder / "parakeet-ft.nemo").unlink()
    assert ME.main(["import", "parakeet-ft-lab-09-a-M01", "--from", dest, "--force"]) == 1
    assert not (folder / "parakeet-ft.nemo").exists()


def rclone_stub(tmp_path, monkeypatch):
    script = tmp_path / "rclone"
    script.write_text(FAKE_RCLONE.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    root = tmp_path / "remote"
    root.mkdir()
    monkeypatch.setenv("FAKE_RCLONE_ROOT", str(root))
    return str(script), root


def test_rclone_destination_streams_checks_md5_and_frees(app, tmp_path, monkeypatch):
    stub, root = rclone_stub(tmp_path, monkeypatch)
    folder = make_model(app, "lab-09-a-M01")
    code = ME.main(["export", "--to", "rclone:fake:willspeak", "--rclone", stub, "--apply", "--free-after-verify"])
    assert code == 0 and not (folder / "parakeet-ft.nemo").exists()
    assert sorted(p.name for p in (root / "willspeak").iterdir()) == ["parakeet-ft-lab-09-a-M01.manifest.json", "parakeet-ft-lab-09-a-M01.zip"]
    assert ME.main(["check", "--to", "rclone:fake:willspeak", "--rclone", stub, "--paranoid"]) == 0
    assert ME.main(["import", "parakeet-ft-lab-09-a-M01", "--from", "rclone:fake:willspeak", "--rclone", stub, "--force"]) == 0
    assert (folder / "parakeet-ft.nemo").exists()


def test_a_corrupted_upload_is_caught_and_nothing_is_freed(app, tmp_path, monkeypatch):
    stub, root = rclone_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_RCLONE_CORRUPT", "1")
    folder = make_model(app, "lab-09-a-M01")
    code = ME.main(["export", "--to", "rclone:fake:willspeak", "--rclone", stub, "--apply", "--free-after-verify"])
    assert code == 1 and (folder / "parakeet-ft.nemo").exists()
    assert not (root / "willspeak/parakeet-ft-lab-09-a-M01.manifest.json").exists()


def test_a_broken_rclone_login_is_reported_before_anything_is_zipped(app, tmp_path, monkeypatch, capsys):
    stub, root = rclone_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_RCLONE_BROKEN", "1")
    make_model(app, "lab-09-a-M01")
    assert ME.main(["export", "--to", "rclone:fake:willspeak", "--rclone", stub, "--apply"]) == 2
    assert "NOT READY" in capsys.readouterr().out


def test_missing_rclone_says_how_to_install_it(app, tmp_path, capsys):
    make_model(app, "lab-09-a-M01")
    assert ME.main(["export", "--to", "rclone:gdrive:x", "--rclone", str(tmp_path / "nope"), "--apply"]) == 2
    assert "not installed" in capsys.readouterr().out
