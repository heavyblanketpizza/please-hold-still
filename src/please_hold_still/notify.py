"""Desktop notifications, so long-running jobs can tell you when they finish."""

from __future__ import annotations

import shutil
import subprocess
import sys


def _applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notification_script(title: str, message: str, sound: str = "Glass") -> str:
    """The AppleScript that shows a notification with a sound."""
    return (
        f"display notification {_applescript_string(message)} "
        f"with title {_applescript_string(title)} sound name {_applescript_string(sound)}"
    )


def notify(title: str, message: str, sound: str = "Glass") -> bool:
    """Print the message and, on a Mac, also pop up a notification with a sound.

    Returns True if a notification was shown. The first time, macOS may ask
    whether "Script Editor" may send notifications. Allow it.
    """
    print(f"[{title}] {message}")
    if sys.platform != "darwin" or shutil.which("osascript") is None:
        return False
    try:
        result = subprocess.run(
            ["osascript", "-e", notification_script(title, message, sound)],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
