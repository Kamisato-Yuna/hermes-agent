"""Update warns when a legacy profile receives an empty secrets file."""
from pathlib import Path

from hermes_cli.profiles import backfill_profile_envs
from hermes_cli.update_cmd_maint import _sync_profiles_after_update


def test_update_notices_legacy_profiles_without_copying_secrets(tmp_path, monkeypatch, capsys):
    home = tmp_path / '.hermes'
    legacy = home / 'profiles' / 'legacy'
    existing = home / 'profiles' / 'existing'
    legacy.mkdir(parents=True)
    existing.mkdir()
    (legacy / 'config.yaml').write_text('{}\n')
    original = 'EXISTING_SECRET=existing-value\n'
    (existing / '.env').write_text(original)
    (home / '.env').write_text('API_KEY=private-root-value\nTELEGRAM_BOT_TOKEN=private-bot-value\n')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr('hermes_cli.profiles.seed_profile_skills', lambda *args, **kwargs: {})

    _sync_profiles_after_update()
    output = capsys.readouterr().out
    assert 'legacy' in output
    assert 'not copied' in output
    assert 'hermes -p legacy' in output
    assert 'private-root-value' not in output and 'private-bot-value' not in output
    assert 'private-' not in (legacy / '.env').read_text()
    assert (existing / '.env').read_text() == original

    _sync_profiles_after_update()
    assert 'not copied' not in capsys.readouterr().out


def test_quiet_backfill_does_not_print(tmp_path, monkeypatch, capsys):
    home = tmp_path / '.hermes'
    legacy = home / 'profiles' / 'legacy'
    legacy.mkdir(parents=True)
    (legacy / 'config.yaml').write_text('{}\n')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    assert backfill_profile_envs(quiet=True) == ['legacy']
    assert capsys.readouterr().out == ''
