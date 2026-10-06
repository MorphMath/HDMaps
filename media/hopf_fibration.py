# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyvista", "pillow"]
# ///
"""Render the README banner: fibers of the Hopf fibration next to the HDMaps wordmark.

Run with ``uv run media/hopf_fibration.py``. Writes ``hopf-fibration-banner.png`` (light theme)
and ``hopf-fibration-banner-dark.png`` (dark theme) next to this script.
"""

import math
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pyvista as pv  # pyright: ignore[reportMissingImports]
from PIL import Image, ImageDraw, ImageFont  # pyright: ignore[reportMissingImports]

HERE = Path(__file__).parent
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/ibmplexsans/IBMPlexSans%5Bwdth,wght%5D.ttf"

WIDTH, HEIGHT = 2560, 640
ART_HEIGHT = 540
GAP = 140
TITLE, TAGLINE = "HDMaps", "Horizontal Diffusion Maps in Python"
THEMES = {
    "hopf-fibration-banner.png": ("#1F2328", "#59636E"),
    "hopf-fibration-banner-dark.png": ("#E6EDF3", "#9198A1"),
}


def fiber(eta: float, phi: float, n: int = 260) -> np.ndarray:
    """Fiber over the point (eta, phi) of the 2-sphere, stereographically projected to R^3."""
    t = np.linspace(0, 2 * np.pi, n)
    c, s = math.cos(eta / 2), math.sin(eta / 2)
    x1, x2, x3, x4 = c * np.cos(t), c * np.sin(t), s * np.cos(t + phi), s * np.sin(t + phi)
    return np.column_stack([x1, x2, x3]) / (1 - x4)[:, None]


def oklch_to_hex(lightness: float, chroma: float, hue: float) -> str:
    a, b = chroma * math.cos(hue), chroma * math.sin(hue)
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )

    def gamma(c: float) -> int:
        c = max(0.0, min(1.0, c))
        return round(255 * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055))

    return "#" + "".join(f"{gamma(c):02x}" for c in rgb)


def render_rosette(path: Path, n: int = 160, lobes: int = 8, eta0: float = 1.05, amp: float = 0.35) -> None:
    """Fibers over a closed curve on the 2-sphere that winds in and out `lobes` times, viewed down the axis."""
    pl = pv.Plotter(off_screen=True, window_size=[2000, 2000])
    for i in range(n):
        u = i / n
        phi = 2 * math.pi * u
        pts = fiber(eta0 + amp * math.sin(lobes * phi), phi)
        pl.add_mesh(
            pv.Spline(pts, len(pts)).tube(radius=0.007, n_sides=16),  # pyright: ignore[reportArgumentType]
            color=oklch_to_hex(0.68, 0.14, 2 * math.pi * u + 3.6),
            smooth_shading=True, ambient=0.35, diffuse=0.65, specular=0.2, specular_power=20,
        )
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(0.0, 0.0, 9.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    pl.reset_camera()  # pyright: ignore[reportCallIssue]
    pl.camera.zoom(1.1)
    pl.screenshot(str(path), transparent_background=True)


def font(path: Path, size: int, weight: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(path), size)
    f.set_variation_by_axes([weight, 100])  # wght, wdth
    return f


def compose(art: Image.Image, font_path: Path) -> None:
    title_font, tag_font = font(font_path, 210, 500), font(font_path, 62, 400)
    for name, (title_color, tag_color) in THEMES.items():
        im = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        tb = d.textbbox((0, 0), TITLE, font=title_font)
        gb = d.textbbox((0, 0), TAGLINE, font=tag_font)
        text_w = max(tb[2] - tb[0], gb[2] - gb[0] + 4)
        x0 = int((WIDTH - (art.width + GAP + text_w)) // 2)
        im.alpha_composite(art, (x0, (HEIGHT - art.height) // 2))

        tx = x0 + art.width + GAP
        gap = 28
        y = (HEIGHT - ((tb[3] - tb[1]) + gap + (gb[3] - gb[1]))) // 2
        d.text((tx - tb[0], y - tb[1]), TITLE, font=title_font, fill=title_color)
        y += (tb[3] - tb[1]) + gap
        d.text((tx - gb[0] + 4, y - gb[1]), TAGLINE, font=tag_font, fill=tag_color)
        im.save(HERE / name, optimize=True)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        font_path = Path(tmp) / "IBMPlexSans.ttf"
        urllib.request.urlretrieve(FONT_URL, font_path)
        art_path = Path(tmp) / "rosette.png"
        render_rosette(art_path)
        art = Image.open(art_path).convert("RGBA")
        art = art.crop(art.getbbox())
        art = art.resize((round(art.width * ART_HEIGHT / art.height), ART_HEIGHT), Image.Resampling.LANCZOS)
        compose(art, font_path)


if __name__ == "__main__":
    main()
