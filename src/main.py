"""
Main entry point for the Music to MIDI application.
"""
import sys
import logging
from pathlib import Path

from src.utils.logger import setup_logger


def main():
    """Main entry point."""
    # Setup logging
    log_dir = Path.home() / ".music-to-midi" / "logs"
    logger = setup_logger(log_dir=str(log_dir))

    logger.info("Starting Music to MIDI application")

    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt

        from src.gui.main_window import MainWindow
        from src.models.data_models import Config

        # Enable high DPI scaling
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("Music to MIDI")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("mason369")

        # Apply dark theme style
        app.setStyle("Fusion")

        # Create and show main window
        config = Config()
        window = MainWindow(config)
        window.show()

        logger.info("Application window shown")

        # Run event loop
        sys.exit(app.exec())

    except ImportError as e:
        logger.error(f"Failed to import PyQt6: {e}")
        print("Error: PyQt6 is required. Install with: pip install PyQt6")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
