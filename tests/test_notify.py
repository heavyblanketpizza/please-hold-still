import subprocess

from please_hold_still import notify as nt


def test_script_escapes_quotes():
    script = nt.notification_script('Done "now"', "path\\to", sound="Glass")
    assert script == (
        'display notification "path\\\\to" with title "Done \\"now\\"" sound name "Glass"'
    )


def test_not_a_mac_just_prints(monkeypatch, capsys):
    monkeypatch.setattr(nt.sys, "platform", "linux")
    assert nt.notify("Title", "msg") is False
    assert "[Title] msg" in capsys.readouterr().out


def test_mac_calls_osascript(monkeypatch):
    calls = []
    monkeypatch.setattr(nt.sys, "platform", "darwin")
    monkeypatch.setattr(nt.shutil, "which", lambda name: "/usr/bin/osascript")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(nt.subprocess, "run", fake_run)
    assert nt.notify("T", "m") is True
    assert calls[0][:2] == ["osascript", "-e"] and "display notification" in calls[0][2]
