"""PyInstaller entry point. A double-clicked .app passes no arguments: it opens the live viewer with the defaults
(backend auto -> LLVM on macOS, 256^2, 30 FPS cap, until the window is closed)."""
from rest_ecro_live.app import main

if __name__ == '__main__':
    main()
