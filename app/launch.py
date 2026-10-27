"""Entry point of the .app bundle: start the size-capped log capture before anything else is imported, then run the app.

    python -m app.launch --log ~/Library/Logs/maya-tts.log
"""
import os
import sys


def main():
    args = sys.argv[1:]
    if "--log" in args:
        i = args.index("--log")
        path = os.path.expanduser(args[i + 1])
        del sys.argv[1 + i:3 + i]
        from app.logcap import capture
        capture(path)
    from app.main import main as run
    run()


if __name__ == "__main__":
    main()
