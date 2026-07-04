"""Generate procedural texture PNGs for HomeBot-3D, committed as static bytes.

Run once (deterministic, seeded); the PNGs under homebot3d/assets/textures/ are
committed. Stdlib + NumPy only — no Pillow. Same PNG-writer pattern as
scripts/render_dev.py.

    python scripts/gen_textures.py
"""
import os
import struct
import zlib
import numpy as np

SIZE = 256
OUT = os.path.join(os.path.dirname(__file__), "..", "homebot3d", "assets", "textures")


def write_png(path, arr):
    arr = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def _noise(rng, scale):
    return rng.normal(0, scale, (SIZE, SIZE, 1))


def floor_wood(rng):
    base = np.array([150.0, 110.0, 70.0])
    img = np.ones((SIZE, SIZE, 3)) * base
    plank_h = 32
    for y0 in range(0, SIZE, plank_h):
        shade = rng.uniform(-18, 18)
        img[y0:y0 + plank_h] += shade
        img[y0:y0 + 2] -= 40                       # seam between planks
    img += _noise(rng, 8)                          # grain
    return img


def floor_tile(rng):
    base = np.array([205.0, 200.0, 190.0])
    img = np.ones((SIZE, SIZE, 3)) * base
    tile = 64
    for k in range(0, SIZE, tile):
        img[k:k + 3] -= 60                         # grout lines
        img[:, k:k + 3] -= 60
    img += _noise(rng, 5)
    return img


def wall_paint(rng):
    base = np.array([222.0, 216.0, 205.0])
    img = np.ones((SIZE, SIZE, 3)) * base
    img += _noise(rng, 4)
    return img


def wood(rng):
    base = np.array([120.0, 82.0, 48.0])
    img = np.ones((SIZE, SIZE, 3)) * base
    x = np.arange(SIZE)
    streak = 12 * np.sin(x / 6.0 + rng.uniform(0, 6))
    img += streak[None, :, None]                   # vertical grain
    img += _noise(rng, 6)
    return img


def fabric(rng):
    base = np.array([95.0, 105.0, 120.0])
    img = np.ones((SIZE, SIZE, 3)) * base
    img += _noise(rng, 10)                         # woven speckle
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    gens = {
        "floor_wood.png": floor_wood,
        "floor_tile.png": floor_tile,
        "wall_paint.png": wall_paint,
        "wood.png": wood,
        "fabric.png": fabric,
    }
    for i, (name, fn) in enumerate(gens.items()):
        rng = np.random.default_rng(1000 + i)      # fixed seed per texture
        write_png(os.path.join(OUT, name), fn(rng))
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
