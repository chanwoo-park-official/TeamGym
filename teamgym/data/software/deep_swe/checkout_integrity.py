"""Container-side checks for a clean pinned checkout and unchanged file content.

The fingerprint covers HEAD, index entries, tracked bytes, symlinks and modes.
Nonignored untracked files also make the checkout dirty. This is an after-run
invariant, not a sandbox proof against transient changes during execution.
"""

CHECKOUT_HELPERS = r'''
def checkout_snapshot(expected_commit=None):
    import hashlib, os, pathlib, subprocess
    def git(*args):
        return subprocess.check_output(["git", *args])
    head = git("rev-parse", "HEAD").decode().strip()
    if expected_commit is not None:
        expected_head = git("rev-parse", expected_commit + "^{commit}").decode().strip()
        if head != expected_head:
            raise RuntimeError("Unexpected base commit: " + head)
    status = git("status", "--porcelain", "--untracked-files=normal").decode()
    digest = hashlib.sha256()
    digest.update(head.encode())
    digest.update(git("ls-files", "--stage", "-z"))
    for raw in sorted(set(git("ls-files", "-z").split(b"\0")) - {b""}):
        path = pathlib.Path(os.fsdecode(raw))
        digest.update(raw + b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(str(path.stat().st_mode).encode() + b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing-or-non-file")
    return {"head": head, "status": status, "sha256": digest.hexdigest(),
            "clean": not bool(status)}
'''


def checkout_unchanged(result: dict) -> bool:
    """Require positive content evidence, never just an unchanged status string."""
    before, after = result.get("checkout_before"), result.get("checkout_after")
    return bool(
        isinstance(before, dict) and isinstance(after, dict)
        and before.get("clean") is True and after.get("clean") is True
        and isinstance(before.get("sha256"), str) and len(before["sha256"]) == 64
        and before.get("head") == after.get("head") == result.get("resolved_base_commit", result.get("base_commit"))
        and before["sha256"] == after.get("sha256")
        and result.get("tracked_checkout_unchanged") is True
    )
