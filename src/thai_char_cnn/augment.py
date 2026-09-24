"""Training-time augmentation, tiered by how fragile each glyph is.

Every class is looked up in `TIER_MEMBERS` (character -> the tiers it belongs to) and gets one
`TierProfile`. A character in several tiers gets the elementwise *most conservative* combination:
the smallest magnitude / probability per field, and any safety flag that one of its tiers sets.
That is how ป, ฎ, ฏ (Tier 2 and Tier 3) end up with Tier 2 rotation/shear plus Tier 3's vertical
care, and how ค (listed under Tier 1 *and* in the ด/ค pair) ends up Tier 2.

| tier | who                                   | idea                                                      |
| ---- | ------------------------------------- | --------------------------------------------------------- |
| 1    | glyphs with no close visual neighbour | full range of everything                                  |
| 2    | hook / loop confusable pairs          | rotation <= 5 deg, rare small shear, no cutout, mild blur  |
| 3    | ascender / descender letters          | Tier 1 rotation, small vertical shift, pad-only vertical  |
| 4    | tone marks and vowel diacritics       | +/- 2.5 deg, no shear / elastic / stroke / blur / cutout   |
| 5    | extreme near-duplicates (า / ๅ)       | +/- 2 deg and almost nothing else                          |

Pixel-sized quantities (`*_px`, `blur_sigma`) are given at the 32 px reference size and scaled
with the model input. All geometry -- bbox jitter, affine, elastic -- is one resample, and it is
never allowed to push ink out of frame: translation is clamped to the blank margin and, if the
rotated / sheared glyph no longer fits, it is shrunk until it does ("confirm headroom before
shift/scale"). Bbox jitter only crops into blank margin, so it can't cut a tail off either.

`configs/augment.json` (optional) can override any profile field per tier and move characters
between tiers once the confusion matrix says so:

    {"profiles": {"tier2": {"rotation_deg": 4}}, "tiers": {"ว": [2]}}

No flips: a mirrored Thai glyph is a different glyph or none at all.
"""

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

WHITE = 255
REF_SIZE = 32          # the size the pixel quantities below are written for
INK_LEVEL = 192        # below this a pixel counts as ink for the fit guard, bbox jitter and cutout


@dataclass(frozen=True)
class TierProfile:
    """For every numeric field, bigger means more aggressive; flags are safety switches."""
    rotation_deg: float        # +/- degrees
    shear_deg: float           # +/- degrees, x only, applied with probability shear_p
    shear_p: float
    scale_delta: float         # isotropic scale in [1 - d, 1 + d]; isotropic never distorts aspect
    translate_x: float         # +/- fraction of the side
    translate_y: float
    bbox_jitter: float         # +/- fraction of the side, per side independently
    elastic_p: float
    elastic_px: float          # RMS displacement of the smooth field
    stroke_p: float            # erode or dilate the ink
    stroke_px: float           # radius; 1 px is the smallest kernel (3x3)
    blur_p: float
    blur_sigma: float          # max Gaussian sigma in px
    noise_p: float
    noise_std: float           # max std, fraction of the 0-255 range
    erase_p: float             # cutout of one small patch, centred on an ink pixel
    erase_frac: float          # max patch side, fraction of the image side
    photo_p: float             # brightness / contrast
    brightness: float          # +/- fraction of the 0-255 range
    contrast: float            # ink contrast in [1 - c, 1 + c], paper stays put
    anti_stack: bool = False           # a high rotation roll switches shear off for that sample
    bbox_pad_only_vertical: bool = False  # bbox jitter never trims the top / bottom margin


FLAGS = ("anti_stack", "bbox_pad_only_vertical")

TIER1 = TierProfile(
    rotation_deg=10, shear_deg=5, shear_p=1.0, scale_delta=0.10, translate_x=0.05, translate_y=0.05,
    bbox_jitter=0.08, elastic_p=0.3, elastic_px=1.0, stroke_p=0.3, stroke_px=1,
    blur_p=0.2, blur_sigma=0.9, noise_p=0.3, noise_std=0.08, erase_p=0.25, erase_frac=0.2,
    photo_p=0.5, brightness=0.10, contrast=0.30,
)
TIER_PROFILES = {
    "tier1": TIER1,
    # hook/loop pairs: the risk is a transform that closes a gap or bends just the hook
    "tier2": replace(TIER1, rotation_deg=5, shear_deg=3, shear_p=0.15, anti_stack=True,
                     translate_x=0.04, translate_y=0.04, bbox_jitter=0.06,
                     elastic_p=0.08, elastic_px=0.4,          # below shear in both p and size
                     stroke_p=0.1, noise_std=0.05, blur_p=0.1, blur_sigma=0.5, erase_p=0.0),
    # ascender/descender: rotation/shear inherit Tier 1 (Tier 2 wins where a letter is in both)
    "tier3": replace(TIER1, translate_y=0.02, bbox_jitter=0.05, bbox_pad_only_vertical=True,
                     elastic_p=0.15, elastic_px=0.7, stroke_p=0.15,
                     blur_p=0.15, blur_sigma=0.5, erase_p=0.0),
    # marks: already 1-2 px strokes; geometry is proportionally strong, blur / erosion erase them
    "tier4": replace(TIER1, rotation_deg=2.5, shear_deg=0, shear_p=0.0, scale_delta=0.05,
                     translate_x=0.03, translate_y=0.03, bbox_jitter=0.03,
                     elastic_p=0.0, elastic_px=0.0, stroke_p=0.0, stroke_px=0, blur_p=0.0, blur_sigma=0,
                     noise_p=0.2, noise_std=0.03, erase_p=0.0, erase_frac=0,
                     brightness=0.05, contrast=0.15),
    # า / ๅ: the relative stroke length is the whole signal; augmentation must not make it worse.
    # Noise stays off until deployment data is known to be noisy.
    "tier5": replace(TIER1, rotation_deg=2, shear_deg=0, shear_p=0.0, scale_delta=0.0,
                     translate_x=0.02, translate_y=0.02, bbox_jitter=0.0,
                     elastic_p=0.0, elastic_px=0.0, stroke_p=0.0, stroke_px=0, blur_p=0.0, blur_sigma=0,
                     noise_p=0.0, noise_std=0.0, erase_p=0.0, erase_frac=0,
                     brightness=0.05, contrast=0.15),
}

# character -> tiers. Keep it a table: move a character by editing its row (or via augment.json).
TIER_GROUPS = {
    1: ("กขคงจชซทธนมยรลวหอฮ"            # listed as having no close neighbour
        "ฉฯๆเแโ"                         # not listed; no size/hook twin -> Tier 1
        "๐๑๒๓๔๕๗๘"),                     # digits other than ๖/๙; ๘-็ unconfirmed (1 val confusion)
    2: ("ผฝพฟบปดคถภศษสฎฏ"                # listed pairs (ค is in Tier 1 too; the stricter one wins)
        "ฐฑฒณญฌ"                          # listed with lower confidence
        "๖๙"                              # rotation-ambiguity pair, treated like Tier 2
        "ต"                               # not listed; ต/ด is the 4th most confused pair in val
        "ใไ"                              # not listed; hook-based vowel twin (4 val confusions)
        "ฃ"),                             # not listed; hook variant of ข
    3: ("ปฝฟฤฦ"                           # top extenders
        "ญฎฏ"                             # bottom extenders
        "ฬ"),                             # not listed; has the ฟ-style ascender
    4: "่้๊๋ัิีึืุู็์ํ",
    5: "าๅ",
}


def default_tiers() -> dict[str, tuple[int, ...]]:
    out: dict[str, set[int]] = {}
    for tier, chars in TIER_GROUPS.items():
        for ch in chars:
            out.setdefault(ch, set()).add(tier)
    return {ch: tuple(sorted(t)) for ch, t in sorted(out.items())}


def combine(profiles: list[TierProfile]) -> TierProfile:
    """The most conservative profile: min of every magnitude / probability, any safety flag."""
    kw = {}
    for f in fields(TierProfile):
        vals = [getattr(p, f.name) for p in profiles]
        kw[f.name] = any(vals) if f.name in FLAGS else min(vals)
    return TierProfile(**kw)


@dataclass(frozen=True)
class AugmentConfig:
    profiles: dict = field(default_factory=dict)   # {"tier2": {"rotation_deg": 4, ...}}
    tiers: dict = field(default_factory=dict)      # {"ว": [2]} replaces that character's tiers
    elastic_sigma_px: float = 4.5                  # smoothness of the elastic field (Simard: 4 at 28 px)

    def resolved_profiles(self) -> dict[str, TierProfile]:
        out = dict(TIER_PROFILES)
        for name, over in self.profiles.items():
            out[name] = replace(out[name], **over)
        return out

    def resolved_tiers(self) -> dict[str, tuple[int, ...]]:
        return default_tiers() | {ch: tuple(sorted(set(t))) for ch, t in self.tiers.items()}

    def profile_for(self, character: str) -> TierProfile:
        tiers = self.resolved_tiers().get(character)
        if not tiers:
            raise KeyError(f"character {character!r} has no augmentation tier; add it to TIER_GROUPS")
        profiles = self.resolved_profiles()
        return combine([profiles[f"tier{t}"] for t in tiers])


def load_augment_config(path: Path) -> AugmentConfig:
    """`configs/augment.json` when present, else the tier defaults. Unknown keys are an error, so a
    stale file from the old flat config can't be silently ignored (`note`, `decided_at` are fine)."""
    if not path.exists():
        return AugmentConfig()
    raw = json.loads(path.read_text())
    known = {f.name for f in fields(AugmentConfig)}
    unknown = set(raw) - known - {"note", "decided_at"}
    assert not unknown, f"{path.name}: unknown key(s) {sorted(unknown)}; expected {sorted(known)}"
    cfg = AugmentConfig(**{k: v for k, v in raw.items() if k in known})
    pnames = {f.name for f in fields(TierProfile)}
    for name, over in cfg.profiles.items():
        assert name in TIER_PROFILES, f"{path.name}: unknown profile {name!r}"
        bad = set(over) - pnames
        assert not bad, f"{path.name}: profiles.{name}: unknown field(s) {sorted(bad)}"
    for ch, t in cfg.tiers.items():
        assert t and all(f"tier{x}" in TIER_PROFILES for x in t), f"{path.name}: tiers.{ch}: bad tiers {t}"
    return cfg


def augment_params(cfg: AugmentConfig) -> dict:
    """What a run hashes: every resolved profile and the full character -> tier table."""
    return dict(elastic_sigma_px=cfg.elastic_sigma_px,
                profiles={k: asdict(v) for k, v in cfg.resolved_profiles().items()},
                tiers={ch: list(t) for ch, t in cfg.resolved_tiers().items()})


def _u(lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(torch.rand(()))


def _hit(p: float) -> bool:
    return p > 0 and float(torch.rand(())) < p


class TieredAugment(torch.nn.Module):
    """(uint8 (1, S, S) tensor, class index) -> uint8 (1, S, S) tensor. Ink dark on white paper."""

    def __init__(self, cfg: AugmentConfig, characters: list[str]):
        super().__init__()
        self.characters = list(characters)
        missing = [c for c in self.characters if not cfg.resolved_tiers().get(c)]
        if missing:
            raise KeyError(f"no augmentation tier for {missing}; add them to TIER_GROUPS")
        self.profiles = [cfg.profile_for(c) for c in self.characters]
        self.elastic_sigma_px = cfg.elastic_sigma_px

    def forward(self, x: torch.Tensor, label: int) -> torch.Tensor:
        p = self.profiles[int(label)]
        _, h, w = x.shape
        assert h == w, "augmentation expects a square input"
        k = w / REF_SIZE
        f = x.float()
        f = self._geometry(f, p, k)
        if _hit(p.stroke_p) and p.stroke_px > 0:
            r = max(1, round(p.stroke_px * k))
            if torch.rand(()) < 0.5:
                f = -F.max_pool2d(-f[None], 2 * r + 1, stride=1, padding=r)[0]   # dilate ink
            else:
                f = F.max_pool2d(f[None], 2 * r + 1, stride=1, padding=r)[0]     # erode ink
        if _hit(p.erase_p) and p.erase_frac > 0:
            f = self._erase(f, p.erase_frac)
        if _hit(p.blur_p) and p.blur_sigma > 0:
            s = _u(0.5, 1.0) * p.blur_sigma * k
            ks = 2 * math.ceil(2 * s) + 1
            f = TF.gaussian_blur(f, [ks, ks], [s, s])
        if _hit(p.photo_p):
            c = _u(1 - p.contrast, 1 + p.contrast)
            f = WHITE - (WHITE - f) * c + _u(-p.brightness, p.brightness) * WHITE
        if _hit(p.noise_p) and p.noise_std > 0:
            f = f + torch.randn_like(f) * _u(0, p.noise_std) * WHITE
        return f.round().clamp(0, WHITE).to(torch.uint8)

    def _geometry(self, f: torch.Tensor, p: TierProfile, k: float) -> torch.Tensor:
        """bbox jitter -> affine -> elastic as one resample, with ink kept inside the frame.

        Coordinates are pixels centred on the image centre, (x, y) with y down. The forward map is
        q = M p + t; grid_sample needs the inverse. Sampling runs on the inverted image (ink bright,
        paper 0) with zero padding, so what enters from outside is clean white paper.
        """
        s_ = f.shape[-1]
        half = s_ / 2
        yy, xx = torch.nonzero(f[0] < INK_LEVEL, as_tuple=True)
        ink = torch.stack([xx, yy], 1).float() - (s_ - 1) / 2            # (N, 2)

        # bbox jitter: a new box per side, cropping only into blank margin, squared, fit to frame
        b, zoom = torch.zeros(2), 1.0
        if p.bbox_jitter > 0 and len(ink):
            j = p.bbox_jitter * s_
            lo_v = 0.0 if p.bbox_pad_only_vertical else -1.0
            d = [_u(-1, 1) * j, _u(-1, 1) * j, _u(lo_v, 1) * j, _u(lo_v, 1) * j]   # l, r, t, b; + pads
            (x0, y0), (x1, y1) = ink.min(0).values - 0.5, ink.max(0).values + 0.5
            left, right = min(-half - d[0], float(x0)), max(half + d[1], float(x1))
            top, bottom = min(-half - d[2], float(y0)), max(half + d[3], float(y1))
            b = torch.tensor([(left + right) / 2, (top + bottom) / 2])
            zoom = s_ / max(right - left, bottom - top)                    # square box: no aspect change

        th = math.radians(_u(-p.rotation_deg, p.rotation_deg))
        sh = 0.0
        if _hit(p.shear_p) and not (p.anti_stack and abs(math.degrees(th)) > p.rotation_deg / 2):
            sh = math.radians(_u(-p.shear_deg, p.shear_deg))
        scale = _u(1 - p.scale_delta, 1 + p.scale_delta)
        rot = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        shear = torch.tensor([[1.0, math.tan(sh)], [0.0, 1.0]])
        m = scale * zoom * rot @ shear
        t_aff = torch.tensor([_u(-p.translate_x, p.translate_x), _u(-p.translate_y, p.translate_y)]) * s_

        elastic = _hit(p.elastic_p) and p.elastic_px > 0
        amp = p.elastic_px * k
        if len(ink):
            lim = (s_ - 1) / 2 - (math.ceil(2 * amp) if elastic else 0)
            q = ink @ m.T
            span = (q.max(0).values - q.min(0).values).clamp_min(1e-6)
            fit = min(1.0, float((2 * lim / span).min()))
            if fit < 1.0:                                                   # shrink until it fits
                m, q = m * fit, q * fit
            t = t_aff - m @ b
            t = torch.maximum(torch.minimum(t, lim - q.max(0).values), -lim - q.min(0).values)
        else:
            t = t_aff - m @ b

        mi = torch.linalg.inv(m)
        theta = torch.cat([mi, (-mi @ (t * 2 / s_))[:, None]], 1)[None]     # normalised coords
        grid = F.affine_grid(theta, [1, 1, s_, s_], align_corners=False)
        if elastic:
            sig = self.elastic_sigma_px * k
            ks = 2 * math.ceil(3 * sig) + 1
            d = TF.gaussian_blur(torch.rand(2, s_, s_) * 2 - 1, [ks, ks], [sig, sig])
            d = d / d.pow(2).mean().sqrt().clamp_min(1e-8) * amp * 2 / s_   # RMS = amp px
            grid = grid + d.permute(1, 2, 0)[None]
        out = F.grid_sample((WHITE - f)[None], grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)[0]
        return WHITE - out

    def _erase(self, f: torch.Tensor, frac: float) -> torch.Tensor:
        s_ = f.shape[-1]
        yy, xx = torch.nonzero(f[0] < INK_LEVEL, as_tuple=True)
        if not len(yy):
            return f
        i = int(torch.randint(len(yy), ()))
        side = max(1, round(_u(frac / 2, frac) * s_))
        y0, x0 = max(0, int(yy[i]) - side // 2), max(0, int(xx[i]) - side // 2)
        f = f.clone()
        f[:, y0:y0 + side, x0:x0 + side] = WHITE
        return f


def build_train_transform(cfg: AugmentConfig, characters: list[str]) -> TieredAugment:
    """`characters[i]` is the glyph of class index i; the transform is called as `tf(pixels, label)`."""
    return TieredAugment(cfg, characters)
