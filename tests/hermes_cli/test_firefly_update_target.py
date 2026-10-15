"""The fork's default update notice must describe the target `firefly-main`."""

from pathlib import Path

import pytest


def test_main_channel_targets_firefly_main_on_the_fork():
    from hermes_cli.source_releases import resolve_source_target

    target = resolve_source_target("main", repository="Kamisato-Yuna/hermes-agent")

    assert target.repository == "Kamisato-Yuna/hermes-agent"
    assert target.branch == "firefly-main"


def test_default_update_source_matches_update_command_target():
    from types import SimpleNamespace

    from hermes_cli.main_install_repair import _resolve_update_branch
    from hermes_cli.source_releases import DEFAULT_UPDATE_BRANCH

    assert DEFAULT_UPDATE_BRANCH == "firefly-main"
    assert _resolve_update_branch(SimpleNamespace(branch=None)) == DEFAULT_UPDATE_BRANCH


@pytest.mark.parametrize("current_branch", ["main", "HEAD", None])
def test_passive_update_check_uses_firefly_main_not_checkout_branch(
    tmp_path, monkeypatch, current_branch
):
    from hermes_cli import config, source_check, steward, update_channel
    from hermes_cli.source_check import _Checkout

    root = tmp_path / "checkout"
    root.mkdir()
    home = tmp_path / "profile"
    home.mkdir()
    head = "a" * 40
    target = "b" * 40
    checkout = _Checkout(
        root=root,
        git="git",
        embedded=None,
        head=head,
        current_branch=current_branch,
        origin="https://github.com/Kamisato-Yuna/hermes-agent.git",
        repository="Kamisato-Yuna/hermes-agent",
        dirty=False,
    )
    monkeypatch.setattr(steward, "read_install_stamp", lambda _root: {})
    monkeypatch.setattr(source_check, "_unsupported_reason", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(source_check, "_read_checkout", lambda *_args: checkout)
    monkeypatch.setattr(config, "require_readable_config_before_write", lambda _path: {})
    monkeypatch.setattr(update_channel, "resolve_update_channel", lambda *_args, **_kwargs: "main")
    monkeypatch.setattr(update_channel, "channel_record", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(update_channel, "rides_default_channel", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(source_check, "_cached_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(source_check, "_write_cache", lambda *_args, **_kwargs: None)

    seen = []
    compares = []

    def branch_tip(repository, branch, *_args, **_kwargs):
        seen.append((repository, branch))
        return target, False, None

    def compare(current, remote, repository):
        compares.append((current, remote, repository))
        return {"ahead_by": 3, "commits": []}

    monkeypatch.setattr(source_check, "_branch_tip", branch_tip)
    monkeypatch.setattr(source_check, "_github_compare", compare)
    status = source_check.check_for_updates(install_root=root, home=home, force=True)

    assert seen == [("Kamisato-Yuna/hermes-agent", "firefly-main")]
    assert compares == [(head, target, "Kamisato-Yuna/hermes-agent")]
    assert status["branch"] == "firefly-main"
    assert status["targetSha"] == target
    assert status["behind"] == 3


def test_startup_banner_reads_firefly_main_not_origin_main(monkeypatch, tmp_path):
    from hermes_cli import banner

    revisions = []
    counts = []

    def git_stdout(args, *, cwd, timeout=5):
        revisions.append(args[-1])
        return "firefly01" if args[-1] == "origin/firefly-main" else "local001"

    def git_count(args, *, cwd):
        counts.append(args[-1])
        return 2

    monkeypatch.setattr(banner.source_check, "_git_stdout", git_stdout)
    monkeypatch.setattr(banner.source_check, "_git_count", git_count)

    state = banner._compute_git_banner_state(tmp_path)

    assert revisions == ["origin/firefly-main", "HEAD"]
    assert counts == ["origin/firefly-main..HEAD"]
    assert state == {"upstream": "firefly01", "local": "local001", "ahead": 2}


def test_embedded_update_probe_uses_firefly_main(monkeypatch):
    from hermes_cli import banner

    seen = []
    tip = "f" * 40

    def branch_tip(repository, branch):
        seen.append((repository, branch))
        return tip

    monkeypatch.setattr(banner, "_github_branch_tip", branch_tip)

    assert banner._upstream_main_sha() == tip
    assert seen == [("Kamisato-Yuna/hermes-agent", "firefly-main")]


def test_update_count_compare_uses_fork_repository(monkeypatch):
    import urllib.request

    from hermes_cli import banner

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ahead_by": 2, "commits": []}'

    urls = []

    def open_url(request, timeout):
        urls.append(request.full_url)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    monkeypatch.setattr(banner, "_compare_payload_cache", {})

    result = banner._github_compare("a" * 40, "b" * 40)

    assert result == {"ahead_by": 2, "commits": []}
    assert urls == ["https://api.github.com/repos/Kamisato-Yuna/hermes-agent/compare/"
                    + "a" * 40 + "..." + "b" * 40]
