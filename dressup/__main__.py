import sys
def run():
    if getattr(sys, "frozen", False) and sys.stdout is None:  # windowed exe
        import io; sys.stdout = sys.stderr = io.StringIO()
    from dressup.cli import main
    r = main(sys.argv[1:])
    if r is not None: sys.exit(r)
    from dressup.gui import run_gui
    sys.exit(run_gui(sys.argv))
if __name__ == "__main__": run()
