import math
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import argparse, time
try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ─── Глобальные параметры ───
W, H = 1920, 1080
FRAMES = 180
FPS = 15
OUT = "galaxy.mp4"

# Шум
OCTAVES = 5
PERSIST = 0.55
LACUN = 2.0
WARP_STRENGTH = 0.55
NOISE_PP_ARM = 4
NY = 1.5
PY = 64

# Звёзды галактик
N_SPIRAL_BASE = 11000
STAR_COUNT_POW = 1.0
STAR_SIZE_POW = 2.5
STAR_SIZE_MAX = 4.5

# ─── Фоновые звёзды ───
N_FIELD = 20000
FIELD_SEED = 999
FIELD_CLUSTER_FRACTION   = 0.85
FIELD_CLUSTER_SIZE_MIN   = 200
FIELD_CLUSTER_SIZE_MAX   = 600
FIELD_CLUSTER_RADIUS     = 80.0
FIELD_CLUSTER_RADIUS_VAR = 0.6
FIELD_CLUSTER_WEIGHT_POW = 2.0
FIELD_CLUSTER_BRIGHT_VAR = 0.25

# Цвет космоса
C_SPACE = np.array([2, 2, 10], dtype=np.float32)

# Блюр звёзд
SIZE_BUCKETS = [1.5, 2.5]
BLUR_SIGMAS  = [0.7, 1.6, 3.2]
BLUR_GAINS   = [1.6, 2.6, 4.0]

# ─── Палитры ───
PALETTES = [
    ((210, 75, 30),  (150, 190, 255), (255, 220, 160)),
    ((190, 50, 60),  (255, 170, 120), (255, 200, 140)),
    ((80, 60, 160),  (200, 200, 255), (255, 220, 220)),
    ((40, 100, 160), (170, 220, 255), (200, 230, 255)),
    ((180, 130, 40), (255, 240, 200), (255, 220, 160)),
    ((120, 40, 120), (220, 180, 255), (255, 200, 230)),
    ((200, 40, 100), (255, 200, 200), (255, 220, 200)),
    ((60, 130, 60),  (200, 255, 200), (240, 255, 220)),
]


# ─── Perlin ───
_GRADS = np.array([
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1],
], dtype=np.float32)
_GRADS /= np.linalg.norm(_GRADS, axis=1, keepdims=True)


def _fade(t): return t*t*t*(t*(t*6.0-15.0)+10.0)


def _hash(xi, yi, px, py, seed):
    xi = np.mod(xi, px).astype(np.uint64)
    yi = np.mod(yi, py).astype(np.uint64)
    h = xi * np.uint64(0x27d4eb2d) + yi * np.uint64(0x165667b1) + np.uint64(seed)
    h &= np.uint64(0xffffffff); h ^= (h >> 15)
    h *= np.uint64(0x2545f491); h &= np.uint64(0xffffffff)
    h ^= (h >> 13)
    return (h & np.uint64(7)).astype(np.int64)


def _perlin2(x, y, px, py, seed):
    xi = np.floor(x).astype(np.int64); yi = np.floor(y).astype(np.int64)
    xf = x - xi; yf = y - yi
    u = _fade(xf); v = _fade(yf)
    h00 = _hash(xi, yi, px, py, seed)
    h10 = _hash(xi+1, yi, px, py, seed)
    h01 = _hash(xi, yi+1, px, py, seed)
    h11 = _hash(xi+1, yi+1, px, py, seed)
    g00 = _GRADS[h00,0]*xf + _GRADS[h00,1]*yf
    g10 = _GRADS[h10,0]*(xf-1.0) + _GRADS[h10,1]*yf
    g01 = _GRADS[h01,0]*xf + _GRADS[h01,1]*(yf-1.0)
    g11 = _GRADS[h11,0]*(xf-1.0) + _GRADS[h11,1]*(yf-1.0)
    nx0 = g00 + u*(g10-g00); nx1 = g01 + u*(g11-g01)
    return nx0 + v*(nx1-nx0)


def _fbm(x, y, px, py, seed=0):
    total = np.zeros_like(x); amp = 1.0; freq = 1.0; mx = 0.0
    for o in range(OCTAVES):
        total += amp * _perlin2(x*freq, y*freq, int(px*freq), int(py*freq), seed + o*131)
        mx += amp; amp *= PERSIST; freq *= LACUN
    return total / mx


# ═══════════════════════════════════════════════════════════
# GALAXY
# ═══════════════════════════════════════════════════════════
class Galaxy:
    def __init__(self, gx, gy, scale, brightness, rot_k,
                 tilt_angle_deg=65.0, tilt_axis_deg=0.0,
                 n_arms=4, pitch_k=4.0, arm_width=0.9,
                 disk_radius=100.0, core_radius=100.0,
                 palette=0, seed_offset=0,
                 pulse_freq=2, pulse_phase=0.0, pulse_depth=0.15):
        self.gx = gx
        self.gy = gy
        self.scale = scale
        self.brightness = brightness
        self.rot_k = rot_k
        self.seed_offset = seed_offset

        self.R_DISK = disk_radius
        self.R_CORE = core_radius
        self.LOG_R_CORE = math.log(self.R_CORE + 1.0)

        d, c, b = PALETTES[palette % len(PALETTES)]
        self.C_DUST_arr  = np.array(d, dtype=np.float32)
        self.C_CORE_arr  = np.array(c, dtype=np.float32)
        self.C_BULGE_arr = np.array(b, dtype=np.float32)
        self.palette_idx = palette

        self.tilt_angle_deg = tilt_angle_deg
        self.tilt_axis_deg = tilt_axis_deg
        self.COS_TILT = math.cos(math.radians(tilt_angle_deg))
        self.TILT_BETA = math.radians(tilt_axis_deg)
        self.COS_BETA = math.cos(self.TILT_BETA)
        self.SIN_BETA = math.sin(self.TILT_BETA)

        self.N_ARMS = n_arms
        self.PITCH_K = pitch_k
        self.ARM_WIDTH = arm_width

        self.NT = self.N_ARMS * NOISE_PP_ARM
        self.NX = self.NT / (2.0 * math.pi)
        self.PX = NOISE_PP_ARM

        self.ROT_PER_FRAME = (self.rot_k * (2.0 * math.pi / self.N_ARMS) / FRAMES)
        self.pulse_freq = pulse_freq
        self.pulse_phase = pulse_phase
        self.pulse_depth = pulse_depth

        self._compute_roi()
        if self.visible:
            self._build_geometry()
            self._build_stars()

    def _compute_roi(self):
        R_extent = 6.5 * self.R_DISK * self.scale + 30
        self.x0 = max(0, int(self.gx - R_extent))
        self.x1 = min(W, int(self.gx + R_extent) + 1)
        self.y0 = max(0, int(self.gy - R_extent))
        self.y1 = min(H, int(self.gy + R_extent) + 1)
        self.roi_W = self.x1 - self.x0
        self.roi_H = self.y1 - self.y0
        self.visible = (self.roi_W > 0) and (self.roi_H > 0)
        if self.visible:
            self._build_roi_mask()

    def _build_roi_mask(self):
        yy, xx = np.mgrid[0:self.roi_H, 0:self.roi_W].astype(np.float32)
        d_edge = np.minimum(
            np.minimum(xx, (self.roi_W - 1) - xx),
            np.minimum(yy, (self.roi_H - 1) - yy),
        )
        fade_len = 0.10 * min(self.roi_W, self.roi_H)
        m = np.clip(d_edge / fade_len, 0.0, 1.0)
        m = m * m * (3.0 - 2.0 * m)
        self.roi_mask = m.astype(np.float32)

    def _tilt_fwd(self, x, y):
        xr =  x * self.COS_BETA + y * self.SIN_BETA
        yr = -x * self.SIN_BETA + y * self.COS_BETA
        yr = yr * self.COS_TILT
        return xr * self.COS_BETA - yr * self.SIN_BETA, \
               xr * self.SIN_BETA + yr * self.COS_BETA

    def _tilt_inv(self, x, y):
        xr =  x * self.COS_BETA + y * self.SIN_BETA
        yr = -x * self.SIN_BETA + y * self.COS_BETA
        yr = yr / self.COS_TILT
        return xr * self.COS_BETA - yr * self.SIN_BETA, \
               xr * self.SIN_BETA + yr * self.COS_BETA

    def _build_geometry(self):
        yy, xx = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        dx = (xx - self.gx) / self.scale
        dy = (yy - self.gy) / self.scale
        DX, DY = self._tilt_inv(dx, dy)
        self.RAD      = np.hypot(DX, DY) + 1e-6
        self.LOG_R    = np.log(self.RAD + 1.0)
        self.TH_WORLD = np.arctan2(DY, DX)

    def _spiral_density(self, r, th):
        lr = np.log(np.maximum(r, 1.0) + 1.0) - self.LOG_R_CORE
        phase = th - self.PITCH_K * lr
        af = phase * self.N_ARMS / (2.0 * np.pi)
        af -= np.round(af)
        arm_dist = af * (2.0 * np.pi / self.N_ARMS)
        spiral = np.exp(-(arm_dist / self.ARM_WIDTH) ** 2)
        radial = np.exp(-r / self.R_DISK)
        return spiral * radial + 0.01 * radial

    def _build_stars(self):
        """Классическое равномерное спиральное распределение (без кластеров)."""
        rng = np.random.default_rng(11 + self.seed_offset * 977)
        N_spiral = max(200, int(N_SPIRAL_BASE * self.scale ** STAR_COUNT_POW))

        max_disk_r = self.R_DISK * 5.0
        N_cand = N_spiral * 6
        cand_r  = max_disk_r * np.sqrt(rng.random(N_cand))
        cand_th = rng.uniform(-np.pi, np.pi, N_cand)
        dens    = self._spiral_density(cand_r, cand_th)
        probs   = dens / dens.sum()
        pick    = rng.choice(N_cand, size=N_spiral, replace=False, p=probs)

        self.star_r  = cand_r[pick]
        self.star_th = cand_th[pick]
        self.N_spiral = N_spiral

        self.star_bright = rng.random(N_spiral) ** STAR_SIZE_POW
        self.star_size   = 0.6 + STAR_SIZE_MAX * rng.random(N_spiral) ** 3.0

        STAR_TINTS = np.array([
            [255, 255, 255], [200, 220, 255], [255, 240, 210],
            [255, 200, 150], [255, 170, 190],
        ], dtype=np.float32)
        tint_p = np.array([0.45, 0.22, 0.18, 0.10, 0.05])
        self.star_tint = STAR_TINTS[rng.choice(5, size=N_spiral, p=tint_p)]

        TWINKLE_FREQS = np.array([1, 2, 3, 5, 7], dtype=np.float32)
        TWINKLE_P = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
        self.tw_freq  = TWINKLE_FREQS[rng.choice(len(TWINKLE_FREQS), size=N_spiral, p=TWINKLE_P)]
        self.tw_phase = rng.random(N_spiral)
        self.tw_depth = np.full(N_spiral, 0.20, dtype=np.float32)

    def render_nebula(self, phi, t):
        th_local = self.TH_WORLD + phi
        so = self.seed_offset * 1000

        n1 = _fbm(th_local * self.NX + 3.7, self.LOG_R * NY + 1.2, self.PX, PY, seed=17 + so)
        n2 = _fbm(th_local * self.NX - 2.1, self.LOG_R * NY - 0.8, self.PX, PY, seed=53 + so)

        th_w = th_local + WARP_STRENGTH * n1 * 0.35
        lr_w = self.LOG_R + WARP_STRENGTH * n2 * 0.30

        phase = th_w - self.PITCH_K * (lr_w - self.LOG_R_CORE)
        af = phase * self.N_ARMS / (2.0 * np.pi)
        af -= np.round(af)
        arm_dist = af * (2.0 * np.pi / self.N_ARMS)

        spiral = np.exp(-(arm_dist / self.ARM_WIDTH) ** 2)
        radial = np.exp(-self.RAD / self.R_DISK)
        core   = np.exp(-(self.RAD / self.R_CORE) ** 2)

        n3 = _fbm(th_local * self.NX * 2.0 + 5.5, self.LOG_R * NY * 2.0, self.PX * 2, PY, seed=91 + so)
        n_mod = 0.35 + 0.65 * (n3 * 0.5 + 0.5)

        dust_density = np.clip((spiral * radial + core * 0.35) * n_mod, 0.0, 1.0)
        core_protect = 1.0 - np.exp(-(self.RAD / (self.R_CORE * 1.2)) ** 2)
        dust_density *= core_protect

        core_glow  = np.exp(-(self.RAD / (self.R_CORE * 0.85)) ** 1.8)
        inner_glow = np.exp(-(self.RAD / (self.R_CORE * 2.5)) ** 1.5) * 0.55

        pulse = 1.0 - self.pulse_depth * (
            0.5 - 0.5 * np.cos(2.0 * np.pi * (self.pulse_phase + self.pulse_freq * t / FRAMES))
        )
        core_glow  = core_glow  * pulse
        inner_glow = inner_glow * pulse

        out = (dust_density[..., None] * self.C_DUST_arr
               + (core_glow * 0.9)[..., None] * self.C_CORE_arr
               + inner_glow[..., None] * self.C_BULGE_arr)
        return out * self.brightness * self.roi_mask[..., None]

    def render_stars(self, phi, t):
        th_world_s = self.star_th - phi
        xd = self.star_r * np.cos(th_world_s)
        yd = self.star_r * np.sin(th_world_s)

        xt, yt = self._tilt_fwd(xd, yd)

        sx = np.round(self.gx + xt * self.scale - self.x0).astype(np.int32)
        sy = np.round(self.gy + yt * self.scale - self.y0).astype(np.int32)

        valid = (sx >= 0) & (sx < self.roi_W) & (sy >= 0) & (sy < self.roi_H)
        sx = sx[valid]; sy = sy[valid]
        b  = self.star_bright[valid]
        sz = self.star_size[valid]
        tv = self.star_tint[valid]

        phase_t = self.tw_phase[valid] + self.tw_freq[valid] * (t / FRAMES)
        twinkle = 1.0 - self.tw_depth[valid] * (0.5 - 0.5 * np.cos(2.0 * np.pi * phase_t))
        b = b * twinkle * self.brightness

        out = np.zeros((self.roi_H, self.roi_W, 3), dtype=np.float32)

        if not _HAS_SCIPY:
            flat = sy * self.roi_W + sx
            for c in range(3):
                out[..., c] = np.bincount(flat, weights=b * tv[:, c],
                                          minlength=self.roi_H * self.roi_W
                                          ).reshape(self.roi_H, self.roi_W).astype(np.float32)
            return out * self.roi_mask[..., None]

        bucket = np.digitize(sz, SIZE_BUCKETS)
        for bidx in range(3):
            m = bucket == bidx
            if not m.any():
                continue
            sy_b = sy[m]; sx_b = sx[m]; b_b = b[m]; tv_b = tv[m]
            flat = sy_b * self.roi_W + sx_b
            sigma = BLUR_SIGMAS[bidx]
            gain  = BLUR_GAINS[bidx]
            for c in range(3):
                m_c = np.bincount(flat, weights=b_b * tv_b[:, c],
                                  minlength=self.roi_H * self.roi_W
                                  ).reshape(self.roi_H, self.roi_W).astype(np.float32)
                out[..., c] += gaussian_filter(m_c, sigma=sigma) * gain
        return out * self.roi_mask[..., None]

    def render(self, phi, t):
        return self.render_nebula(phi, t) + self.render_stars(phi, t)


# ═══════════════════════════════════════════════════════════
# FIELD STARS — кластеризованное звёздное небо
# ═══════════════════════════════════════════════════════════
FIELD = None


def build_field_stars(seed):                       
    """Возвращает dict с массивами x, y, bright, size, tint, tw_*."""
    rng = np.random.default_rng(FIELD_SEED + seed)

    # ── Крайний случай: поле отключено ──
    if N_FIELD <= 0:
        empty_f32 = np.zeros(0, dtype=np.float32)
        empty_tint = np.zeros((0, 3), dtype=np.float32)
        return dict(
            x=empty_f32, y=empty_f32,
            bright=empty_f32, size=empty_f32,
            tint=empty_tint,
            tw_freq=empty_f32, tw_phase=empty_f32, tw_depth=empty_f32,
        )

    N_total = N_FIELD
    n_clustered = int(N_total * FIELD_CLUSTER_FRACTION)
    n_free      = N_total - n_clustered

    # ── Кластеры ──
    avg_size = max(1, (FIELD_CLUSTER_SIZE_MIN + FIELD_CLUSTER_SIZE_MAX) // 2)
    n_clusters = max(1, n_clustered // avg_size)

    margin = 40
    cluster_cx = rng.uniform(margin, W - margin, n_clusters)
    cluster_cy = rng.uniform(margin, H - margin, n_clusters)

    # Степенной разброс размеров кластеров
    weights = rng.random(n_clusters) ** FIELD_CLUSTER_WEIGHT_POW
    weights /= weights.sum()
    counts = rng.multinomial(n_clustered, weights)

    # Радиусы кластеров
    sig_lo = FIELD_CLUSTER_RADIUS * (1.0 - FIELD_CLUSTER_RADIUS_VAR)
    sig_hi = FIELD_CLUSTER_RADIUS * (1.0 + FIELD_CLUSTER_RADIUS_VAR)
    sigmas = rng.uniform(sig_lo, sig_hi, n_clusters)

    # Яркостный множитель кластера: крупные ярче
    cluster_bright = 1.0 + FIELD_CLUSTER_BRIGHT_VAR * (weights / weights.max() - 0.5) * 2.0
    cluster_bright = np.clip(cluster_bright, 0.6, 1.5)

    x_list = []
    y_list = []
    bm_list = []

    for i in range(n_clusters):
        n = counts[i]
        if n == 0:
            continue
        sigma = sigmas[i]
        dx = rng.normal(0.0, sigma, n)
        dy = rng.normal(0.0, sigma, n)
        x_list.append(cluster_cx[i] + dx)
        y_list.append(cluster_cy[i] + dy)
        bm_list.append(np.full(n, cluster_bright[i], dtype=np.float32))

    # ── Свободные звёзды — равномерно по кадру ──
    if n_free > 0:
        x_list.append(rng.uniform(0, W, n_free))
        y_list.append(rng.uniform(0, H, n_free))
        bm_list.append(np.full(n_free, 0.75, dtype=np.float32))

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    bright_mult = np.concatenate(bm_list)
    N = len(x)

    # Яркость и размер
    bright = rng.random(N) ** STAR_SIZE_POW * 0.6 * bright_mult
    size   = 0.6 + STAR_SIZE_MAX * rng.random(N) ** 3.0

    STAR_TINTS = np.array([
        [255, 255, 255], [200, 220, 255], [255, 240, 210],
        [255, 200, 150], [255, 170, 190],
    ], dtype=np.float32)
    tint_p = np.array([0.45, 0.22, 0.18, 0.10, 0.05])
    tint = STAR_TINTS[rng.choice(5, size=N, p=tint_p)]

    TWINKLE_FREQS = np.array([1, 2, 3, 5, 7], dtype=np.float32)
    TWINKLE_P = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
    tw_freq  = TWINKLE_FREQS[rng.choice(len(TWINKLE_FREQS), size=N, p=TWINKLE_P)]
    tw_phase = rng.random(N)
    tw_depth = np.full(N, 0.55, dtype=np.float32)

    return dict(x=x, y=y, bright=bright, size=size, tint=tint,
                tw_freq=tw_freq, tw_phase=tw_phase, tw_depth=tw_depth)


def render_field(t):
    if FIELD is None or len(FIELD["x"]) == 0:
        return np.zeros((H, W, 3), dtype=np.float32)

    fs = FIELD
    sx = np.round(fs["x"]).astype(np.int32)
    sy = np.round(fs["y"]).astype(np.int32)
    valid = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    sx = sx[valid]; sy = sy[valid]
    b  = fs["bright"][valid]
    sz = fs["size"][valid]
    tv = fs["tint"][valid]

    phase_t = fs["tw_phase"][valid] + fs["tw_freq"][valid] * (t / FRAMES)
    twinkle = 1.0 - fs["tw_depth"][valid] * (0.5 - 0.5 * np.cos(2.0 * np.pi * phase_t))
    b = b * twinkle

    out = np.zeros((H, W, 3), dtype=np.float32)
    if not _HAS_SCIPY:
        flat = sy * W + sx
        for c in range(3):
            out[..., c] = np.bincount(flat, weights=b * tv[:, c],
                                      minlength=H * W).reshape(H, W).astype(np.float32)
        return out

    bucket = np.digitize(sz, SIZE_BUCKETS)
    for bidx in range(3):
        m = bucket == bidx
        if not m.any():
            continue
        sy_b = sy[m]; sx_b = sx[m]; b_b = b[m]; tv_b = tv[m]
        flat = sy_b * W + sx_b
        sigma = BLUR_SIGMAS[bidx]
        gain  = BLUR_GAINS[bidx]
        for c in range(3):
            m_c = np.bincount(flat, weights=b_b * tv_b[:, c],
                              minlength=H * W).reshape(H, W).astype(np.float32)
            out[..., c] += gaussian_filter(m_c, sigma=sigma) * gain
    return out


# ═══════════════════════════════════════════════════════════
# GLOBALS
# ═══════════════════════════════════════════════════════════
GALAXIES = []


def _find_position(rng, occupied, r_eff, sep, margin_px, max_tries=100):
    best_xy = None
    best_slack = -1e9

    for _ in range(max_tries):
        gx = rng.uniform(margin_px * W, (1 - margin_px) * W)
        gy = rng.uniform(margin_px * H, (1 - margin_px) * H)

        min_slack = float('inf')
        for (ex, ey, er) in occupied:
            d = math.hypot(gx - ex, gy - ey)
            need = (r_eff + er) * sep
            slack = d - need
            if slack < min_slack:
                min_slack = slack
            if min_slack < best_slack:
                break

        if min_slack >= 0:
            return gx, gy
        if min_slack > best_slack:
            best_slack = min_slack
            best_xy = (gx, gy)

    return best_xy


def _init(cfg):
    global GALAXIES, FIELD
    GALAXIES = []
    FIELD = build_field_stars(cfg["seed"])

    rng = np.random.default_rng(cfg["seed"])

    # ── Шаг 1. Сэмплируем параметры ──
    specs = []
    if cfg["use_main"]:
        specs.append(dict(
            is_main=True,
            scale=cfg["main_scale"], brightness=cfg["main_bright"],
            rot_k=cfg["main_rot_k"], tilt=cfg["main_tilt"], tilt_axis=0.0,
            n_arms=cfg["main_arms"], pitch=cfg["main_pitch"],
            arm_width=cfg["main_arm_width"],
            disk=cfg["main_disk"], core_ratio=cfg["main_core_ratio"],
            palette=cfg["main_palette"], seed_offset=0,
            fixed_pos=(W * cfg["main_x"], H * cfg["main_y"]),
        ))

    n_far = cfg["n_far"]
    mode = cfg["depth_mode"]
    j = cfg["depth_jitter"]

    for i in range(n_far):
        pulse_freq  = int(rng.integers(cfg["pulse_freq_min"], cfg["pulse_freq_max"] + 1))
        pulse_phase = float(rng.random())
        pulse_depth = float(rng.uniform(cfg["pulse_depth_min"], cfg["pulse_depth_max"]))

        if mode == "geometric":
            base = cfg["depth_start"] * (cfg["depth_ratio"] ** i)
        elif mode == "power":
            t = i / (n_far - 1) if n_far > 1 else 0.0
            base = cfg["depth_start"] * (1.0 - t) ** cfg["depth_power"]
        else:
            base = None

        if base is None:
            scale = rng.uniform(cfg["far_scale_min"], cfg["far_scale_max"])
            bright = rng.uniform(cfg["far_bright_min"], cfg["far_bright_max"])
        else:
            scale = base * rng.uniform(1.0 - j, 1.0 + j)
            scale = float(np.clip(scale, cfg["depth_min_scale"], 1.0))
            bright = max(0.15, scale ** 1.4) * rng.uniform(1.0 - j, 1.0 + j)
            bright = float(np.clip(bright, 0.12, 0.95))

        disk = rng.uniform(cfg["far_disk_min"], cfg["far_disk_max"])
        specs.append(dict(
            is_main=False,
            scale=scale,
            brightness=bright,
            rot_k=int(rng.choice([1, 2, 3, -1, -2, -3])),
            tilt=rng.uniform(cfg["far_tilt_min"], cfg["far_tilt_max"]),
            tilt_axis=rng.uniform(0, 360),
            n_arms=int(rng.integers(cfg["far_arms_min"], cfg["far_arms_max"] + 1)),
            pitch=rng.uniform(cfg["far_pitch_min"], cfg["far_pitch_max"]),
            arm_width=rng.uniform(cfg["far_arm_width_min"], cfg["far_arm_width_max"]),
            disk=disk,
            core_ratio=rng.uniform(cfg["far_core_ratio_min"], cfg["far_core_ratio_max"]),
            palette=int(rng.integers(0, len(PALETTES))),
            seed_offset=int(rng.integers(1, 100000)),
            pulse_freq=pulse_freq,
            pulse_phase=pulse_phase,
            pulse_depth=pulse_depth,
            fixed_pos=None,
        ))

    # ── Шаг 2. Размещаем ──
    occupied = []
    margin_px = 0.06
    sep = cfg["separation"]

    for s in specs:
        if not s["is_main"]:
            continue
        gx, gy = s["fixed_pos"]
        r_eff = s["disk"] * s["scale"] * 2.0
        occupied.append((gx, gy, r_eff))
        GALAXIES.append(Galaxy(
            gx=gx, gy=gy, scale=s["scale"], brightness=s["brightness"],
            rot_k=s["rot_k"], tilt_angle_deg=s["tilt"], tilt_axis_deg=s["tilt_axis"],
            n_arms=s["n_arms"], pitch_k=s["pitch"], arm_width=s["arm_width"],
            disk_radius=s["disk"], core_radius=s["disk"] * s["core_ratio"],
            palette=s["palette"], seed_offset=s["seed_offset"],
            pulse_freq=cfg["main_pulse_freq"],
            pulse_phase=0.0,
            pulse_depth=cfg["main_pulse_depth"],
        ))

    for s in specs:
        if s["is_main"]:
            continue
        r_eff = s["disk"] * s["scale"] * 2.0
        gx, gy = _find_position(rng, occupied, r_eff, sep, margin_px)
        occupied.append((gx, gy, r_eff))
        GALAXIES.append(Galaxy(
            gx=gx, gy=gy, scale=s["scale"], brightness=s["brightness"],
            rot_k=s["rot_k"], tilt_angle_deg=s["tilt"], tilt_axis_deg=s["tilt_axis"],
            n_arms=s["n_arms"], pitch_k=s["pitch"], arm_width=s["arm_width"],
            disk_radius=s["disk"], core_radius=s["disk"] * s["core_ratio"],
            palette=s["palette"], seed_offset=s["seed_offset"],
            pulse_freq=s["pulse_freq"],
            pulse_phase=s["pulse_phase"],
            pulse_depth=s["pulse_depth"],
        ))


def render_frame(t):
    canvas = np.tile(C_SPACE, (H, W, 1))
    canvas += render_field(t)
    for g in GALAXIES:
        if not g.visible:
            continue
        phi = t * g.ROT_PER_FRAME
        layer = g.render(phi, t)
        canvas[g.y0:g.y1, g.x0:g.x1] += layer
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
DEFAULTS = dict(
    use_main=True, main_x=0.50, main_y=0.50,
    main_scale=1.0, main_bright=1.0, main_rot_k=1, main_tilt=65.0,
    main_arms=4, main_pitch=4.0, main_arm_width=0.9,
    main_disk=100.0, main_core_ratio=1.0, main_palette=0,
    n_far=4, seed=42,
    far_scale_min=0.15,   far_scale_max=0.45,
    far_bright_min=0.25,  far_bright_max=0.70,
    far_tilt_min=20.0,    far_tilt_max=85.0,
    far_arms_min=2,       far_arms_max=6,
    far_pitch_min=3.0,    far_pitch_max=6.0,
    far_arm_width_min=0.7, far_arm_width_max=1.3,
    far_disk_min=70.0,    far_disk_max=130.0,
    far_core_ratio_min=0.4, far_core_ratio_max=1.6,
    separation=1.0,
    depth_mode="off",
    depth_start=0.65,
    depth_ratio=0.62,
    depth_power=2.2,
    depth_jitter=0.15,
    depth_min_scale=0.10,
    pulse_freq_min=1,
    pulse_freq_max=3,
    pulse_depth_min=0.10,
    pulse_depth_max=0.25,
    main_pulse_freq=2,
    main_pulse_depth=0.15,
)


def parse_args():
    p = argparse.ArgumentParser(description="Одна большая + N далёких галактик.")
    p.add_argument("mode", choices=["preview", "video"])
    p.add_argument("-t", "--frame", type=int, default=0)
    p.add_argument("-o", "--out", type=str, default=None)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("-W", "--width",  type=int, default=None)
    p.add_argument("-H", "--height", type=int, default=None)
    p.add_argument("-n", "--frames", type=int, default=None)

    p.add_argument("--n-far", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-main", action="store_true")

    # главная
    p.add_argument("--main-scale",  type=float, default=None)
    p.add_argument("--main-bright", type=float, default=None)
    p.add_argument("--main-tilt",   type=float, default=None)
    p.add_argument("--main-x",      type=float, default=None)
    p.add_argument("--main-y",      type=float, default=None)
    p.add_argument("--main-arms",   type=int,   default=None)
    p.add_argument("--main-disk",   type=float, default=None)
    p.add_argument("--main-core-ratio", type=float, default=None)
    p.add_argument("--main-palette", type=int, default=None)

    # дальние
    p.add_argument("--far-scale-min",  type=float, default=None)
    p.add_argument("--far-scale-max",  type=float, default=None)
    p.add_argument("--far-bright-min", type=float, default=None)
    p.add_argument("--far-bright-max", type=float, default=None)
    p.add_argument("--far-tilt-min",   type=float, default=None)
    p.add_argument("--far-tilt-max",   type=float, default=None)
    p.add_argument("--far-arms-min",   type=int,   default=None)
    p.add_argument("--far-arms-max",   type=int,   default=None)
    p.add_argument("--far-disk-min",   type=float, default=None)
    p.add_argument("--far-disk-max",   type=float, default=None)
    p.add_argument("--far-core-ratio-min", type=float, default=None)
    p.add_argument("--far-core-ratio-max", type=float, default=None)
    p.add_argument("--separation", type=float, default=None)

    # глубина
    p.add_argument("--depth-mode", choices=["off", "geometric", "power"], default=None)
    p.add_argument("--depth-start",     type=float, default=None)
    p.add_argument("--depth-ratio",     type=float, default=None)
    p.add_argument("--depth-power",     type=float, default=None)
    p.add_argument("--depth-jitter",    type=float, default=None)
    p.add_argument("--depth-min-scale", type=float, default=None)

    p.add_argument("--slowdown", type=float, default=None)

    # пульсация
    p.add_argument("--pulse-freq-min", type=int, default=None)
    p.add_argument("--pulse-freq-max", type=int, default=None)
    p.add_argument("--pulse-depth-min", type=float, default=None)
    p.add_argument("--pulse-depth-max", type=float, default=None)
    p.add_argument("--main-pulse-freq", type=int, default=None)
    p.add_argument("--main-pulse-depth", type=float, default=None)
    p.add_argument("--no-pulse", action="store_true")

    # фоновые кластеры
    p.add_argument("--field-stars",         type=int,   default=None,
                   help="Всего фоновых звёзд")
    p.add_argument("--field-cluster-frac",  type=float, default=None,
                   help="Доля звёзд в кластерах (0..1)")
    p.add_argument("--field-cluster-radius",type=float, default=None,
                   help="Радиус фонового кластера (px)")
    p.add_argument("--field-cluster-size-min", type=int, default=None,
                   help="Мин. звёзд в кластере")
    p.add_argument("--field-cluster-size-max", type=int, default=None,
                   help="Макс. звёзд в кластере")
    p.add_argument("--field-no-clusters", action="store_true",
                   help="Отключить кластеризацию фона")

    return p.parse_args()


def args_to_cfg(args):
    cfg = dict(DEFAULTS)
    for k in ["seed", "n_far",
              "main_scale", "main_bright", "main_tilt", "main_x", "main_y",
              "main_arms", "main_disk", "main_core_ratio", "main_palette",
              "far_scale_min", "far_scale_max",
              "far_bright_min", "far_bright_max",
              "far_tilt_min", "far_tilt_max",
              "far_arms_min", "far_arms_max",
              "far_disk_min", "far_disk_max",
              "far_core_ratio_min", "far_core_ratio_max",
              "separation",
              "pulse_freq_min", "pulse_freq_max",
              "pulse_depth_min", "pulse_depth_max",
              "main_pulse_freq", "main_pulse_depth",
              "depth_mode", "depth_start", "depth_ratio", "depth_power",
              "depth_jitter", "depth_min_scale"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if args.no_pulse:
        cfg["pulse_depth_min"] = 0.0
        cfg["pulse_depth_max"] = 0.0
        cfg["main_pulse_depth"] = 0.0
    if args.no_main:
        cfg["use_main"] = False
    return cfg


def _apply_field_overrides(args):
    """Применяем CLI-флаги фоновых звёзд к глобалам до _init()."""
    global N_FIELD, FIELD_CLUSTER_FRACTION, FIELD_CLUSTER_RADIUS
    global FIELD_CLUSTER_SIZE_MIN, FIELD_CLUSTER_SIZE_MAX
    if args.field_stars          is not None: N_FIELD = args.field_stars
    if args.field_cluster_frac   is not None: FIELD_CLUSTER_FRACTION = args.field_cluster_frac
    if args.field_cluster_radius is not None: FIELD_CLUSTER_RADIUS = args.field_cluster_radius
    if args.field_cluster_size_min is not None: FIELD_CLUSTER_SIZE_MIN = args.field_cluster_size_min
    if args.field_cluster_size_max is not None: FIELD_CLUSTER_SIZE_MAX = args.field_cluster_size_max
    if args.field_no_clusters:
        FIELD_CLUSTER_FRACTION = 0.0


def save_preview(t, path):
    t0 = time.time()
    frame = render_frame(t)
    Image.fromarray(frame).save(path)
    print(f"Превью t={t} → {path}  ({time.time()-t0:.2f} c)")


def render_video(frames, fps, path):
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_log_level="error")
    total = len(frames)
    for i, t in enumerate(frames):
        print(f"\rкадр {i+1}/{total} (t={t})", end="", flush=True)
        writer.append_data(render_frame(t))
    writer.close()
    print(f"\nГотово: {path}")


def main():
    args = parse_args()
    global W, H, FRAMES
    if args.width:  W = args.width
    if args.height: H = args.height
    if args.frames: FRAMES = args.frames
    if args.slowdown: FRAMES = int(FRAMES * args.slowdown)

    _apply_field_overrides(args)

    cfg = args_to_cfg(args)
    _init(cfg)

    if args.mode == "preview":
        t = max(0, min(FRAMES - 1, args.frame))
        out = args.out or f"galaxy_t{t:02d}.png"
        save_preview(t, out)
    else:
        out = args.out or OUT
        render_video(range(FRAMES), args.fps, out)


if __name__ == "__main__":
    main()