"""Render the app icon (a rounded tile with the Tamil letter 'ம') to an .icns via iconutil."""
import subprocess
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath


def render(size):
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    m = size * 0.1
    r = QRectF(m, m, size - 2 * m, size - 2 * m)
    g = QLinearGradient(r.topLeft(), r.bottomRight())
    g.setColorAt(0, QColor("#ff7a45"))
    g.setColorAt(1, QColor("#c2185b"))
    path = QPainterPath()
    path.addRoundedRect(r, size * 0.18, size * 0.18)
    p.fillPath(path, g)
    p.setPen(QColor("white"))
    f = QFont("Tamil Sangam MN")
    f.setPixelSize(int(size * 0.5))
    p.setFont(f)
    p.drawText(r, Qt.AlignCenter, "ம")
    p.end()
    return img


def main(out):
    QGuiApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory() as d:
        iconset = Path(d) / "AppIcon.iconset"
        iconset.mkdir()
        for s in (16, 32, 128, 256, 512):
            render(s).save(str(iconset / f"icon_{s}x{s}.png"))
            render(2 * s).save(str(iconset / f"icon_{s}x{s}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", out], check=True)


if __name__ == "__main__":
    main(sys.argv[1])
