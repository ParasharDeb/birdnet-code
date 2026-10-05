"""CLI helpers.

Species names here include emoji, and the Windows console still defaults to
cp1252, which raises UnicodeEncodeError on them. Force UTF-8 with a replacement
fallback so a console limitation never takes a script down.
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # not a real tty, or already redirected
        pass
