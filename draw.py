import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import random
import torch
import os
import signal
import pickle
from utils import add_result, get_results
from patchfm import Forecaster, PatchFMConfig

from matplotlib import font_manager as _fm
_installed = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ["Arial Rounded MT Bold", "Nunito", "Ubuntu", "DejaVu Sans"]
                               if f in _installed] or ["DejaVu Sans"]

# --- Model setup ---
config = PatchFMConfig(compile=False, full_leakage=False)
model = Forecaster(config)

# --- Dataset management ---
DATASET_OPTIONS = ["simple.npy", "medium.npy", "dic_stocks.pkl"]
DATASET_LABELS = {
    "simple.npy": "Facile",
    "medium.npy": "Moyen",
    "dic_stocks.pkl": "Bourse (expert)",
}
current_dataset_name = None
data = None
current_dataset_index = 0
current_stock_name: str | None = None
radio_datasets = None

def _load_dataset_by_name(name: str) -> bool:
    """Load a dataset file by name into the global `data`. Returns True on success."""
    global data, current_dataset_name, current_dataset_index
    try:
        # Resolve relative to this script directory
        base_dir = os.path.dirname(__file__)
        name = os.path.join("data", name)
        path = os.path.join(base_dir, name)
        if name.endswith('.pkl'):
            with open(path, 'rb') as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, dict) or not loaded:
                raise ValueError("pickle must be a non-empty dict of name->np.array")
            # ensure arrays
            fixed = {}
            for k, v in loaded.items():
                arr = np.asarray(v, dtype=float).reshape(-1)
                fixed[k] = arr
            data = fixed
        else:
            arr = np.load(path)
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            data = arr
        current_dataset_name = name
        if os.path.basename(name) in DATASET_OPTIONS:
            current_dataset_index = DATASET_OPTIONS.index(os.path.basename(name))
        print(f"Dataset chargé: {name} (shape={getattr(data, 'shape', None)})")
        return True
    except Exception as exc:
        print(f"Impossible de charger {name}: {exc}")
        return False

def _dataset_label() -> str:
    return f"Niveau : {_dataset_display_name(current_dataset_name or '?')}"

def _dataset_display_name(name: str) -> str:
    base = os.path.basename(name)
    return DATASET_LABELS.get(base, os.path.splitext(base)[0])

def _dataset_name_from_label(label: str) -> str:
    # Reverse lookup from display label to filename
    for k, v in DATASET_LABELS.items():
        if v == label:
            return k
    # Fallback: assume label is a filename if unmatched
    return label

MAX_CTX = 256
FORECAST_HORIZON = 32


# --- Signal configuration ---
n_future = FORECAST_HORIZON
x = None
y = None
x_obs = None
y_obs = None
x_future = None
y_future = None
n_obs = None
y_lim = None
current_signal_index = None
fig = None
ax = None
button_new_signal = None
button_erase = None
button_validate = None
button_close = None
button_model = None
model_pred = None
mse_pred = None
button_dataset = None
human_pred = None
human_mse = None
ranking_ax = None
radio_ax = None

# --- Palette and layout ---
COLOR_OBS = "#ff8b3d"          # orange vif
COLOR_FUTURE = "#118ab2"       # bleu joyeux
COLOR_PRED = "#ef476f"         # rose vif
COLOR_DRAWN = "#06d6a0"        # vert menthe
COLOR_WINDOW = "#ffd166"       # jaune clair
COLOR_MODEL = "#7b2cbf"        # violet modèle
AX_FACE = "#ffffff"
FIG_FACE = "#eef3ff"
TEXT_DARK = "#2d3047"
TEXT_MUTED = "#8a90a6"
hint_artist = None

# --- One visual theme per level (background colours + a few decorations) ---
THEMES = {
    "simple.npy": dict(name="prairie", fig="#e9f7e1", ax="#fbfff6", text="#2d3047", muted="#7d8a76",
                       deco=["\U0001F33B", "\U0001F98B", "\U0001F331"]),
    "medium.npy": dict(name="océan", fig="#e0f1fb", ax="#f7fcff", text="#1f3a5f", muted="#6f8aa6",
                       deco=["\U0001F420", "\U0001F433", "\U0001FAE7"]),
    "dic_stocks.pkl": dict(name="espace", fig="#1c2045", ax="#262b57", text="#f4f5ff", muted="#aab0d6",
                           zone="#7b83ff", zone_alpha=0.22, deco=["\U0001F680", "\U0001FA90", "\U0001F31F"]),
}
DECO_SPOTS = [(0.09, 0.95), (0.70, 0.95), (0.86, 0.59), (0.94, 0.55)]
theme_artists = []

# --- Player avatar (chosen on the welcome screen) ---
AVATARS = ["\U0001F98A", "\U0001F43C", "\U0001F438", "\U0001F984", "\U0001F42F"]
AVATAR = AVATARS[0]
avatar_artist = None

def _show_popup(message: str, duration_ms: int = 2200, face_color: str = "#ff4b5c"):
    """
    Show a transient popup message on the figure and remove it after duration_ms.
    Uses the figure's canvas timer so removal happens on the GUI thread. Cancels any
    previously displayed popup so they don't overlap or persist.
    """
    global fig

    # If no figure available, fallback to printing
    if fig is None:
        print(message)
        return

    # Cancel previous popup/timer if any
    try:
        prev_timer = getattr(_show_popup, "_timer", None)
        if prev_timer is not None:
            try:
                prev_timer.stop()
            except Exception:
                pass
        for a in getattr(_show_popup, "_artists", []):
            try:
                a.remove()
            except Exception:
                pass
    except Exception:
        pass

    # create centered text near top of figure
    txt = fig.text(
        0.5, 0.97, message,
        ha="center", va="top", fontsize=16, fontweight="bold", color="#ffffff",
        bbox=dict(facecolor=face_color, boxstyle="round,pad=0.6", alpha=0.95, edgecolor="none"),
        zorder=50
    )

    fig.canvas.draw_idle()

    # schedule removal using the figure's GUI timer
    timer = fig.canvas.new_timer(interval=int(duration_ms))

    def _remove_popup():
        try:
            txt.remove()
            fig.canvas.draw_idle()
        except Exception:
            pass

    timer.add_callback(_remove_popup)
    timer.start()

    # store references so we can cancel/clear if a new popup is shown
    _show_popup._timer = timer
    _show_popup._artists = [txt]


# --- Winner box (shown a moment after the robot finished drawing) ---
WINNER_DELAY_MS = 2000
WINNER_SHOW_MS = 5000
winner_artists = []
winner_timers = []


def _stop_winner_timers():
    for tm in winner_timers:
        try:
            tm.stop()
        except Exception:
            pass
    winner_timers.clear()


def _hide_winner_box(redraw: bool = True):
    for a in winner_artists:
        try:
            a.remove()
        except Exception:
            pass
    had_box = bool(winner_artists)
    winner_artists.clear()
    if redraw and had_box and fig is not None:
        fig.canvas.draw_idle()
    return had_box


def _show_winner_box():
    """Big friendly box announcing the winner; confetti if the child won, crying emoji otherwise."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    _hide_winner_box(redraw=False)
    if human_mse is None or mse_pred is None or not np.isfinite(human_mse) or not np.isfinite(mse_pred):
        return
    if human_mse < mse_pred:
        title, col, emo = "Tu as gagné !", "#2bb673", "\U0001F973"
    elif human_mse > mse_pred:
        title, col, emo = "Le robot a gagné...", "#f25f5c", "\U0001F62D"
    else:
        title, col, emo = "Égalité !", "#6c757d", "\U0001F91D"

    # centred over the main plot, in figure coordinates
    pos = ax.get_position()
    w, h = 0.34, 0.40
    x0, y0 = pos.x0 + (pos.width - w) / 2, pos.y0 + (pos.height - h) / 2 + 0.02
    cx = x0 + w / 2
    fw, fh = fig.get_size_inches()
    panel = FancyBboxPatch((x0, y0), w, h, transform=fig.transFigure,
                           boxstyle="round,pad=0,rounding_size=0.03", mutation_aspect=fw / fh,
                           facecolor="#ffffff", edgecolor=col, linewidth=5, zorder=200)
    fig.add_artist(panel)
    winner_artists.append(panel)

    img = _emoji_img(emo)
    if img is not None:
        ab = AnnotationBbox(OffsetImage(img, zoom=70 / img.shape[0]), (cx, y0 + h * 0.70),
                            xycoords="figure fraction", frameon=False, zorder=210)
        fig.add_artist(ab)
        winner_artists.append(ab)
    kw = dict(ha="center", va="center", transform=fig.transFigure, zorder=210)
    winner_artists.append(fig.text(cx, y0 + h * 0.40, title, fontsize=30, fontweight="bold", color=col, **kw))
    winner_artists.append(fig.text(cx, y0 + h * 0.22,
                                   f"Toi : {_fmt_score(human_mse)}     Robot : {_fmt_score(mse_pred)}",
                                   fontsize=16, color=TEXT_DARK, **kw))
    winner_artists.append(fig.text(cx, y0 + h * 0.08, "(clique pour fermer)", fontsize=10,
                                   color="#8a90a6", **kw))
    if human_mse < mse_pred:
        _launch_confetti()
    fig.canvas.draw_idle()

    close_timer = fig.canvas.new_timer(interval=WINNER_SHOW_MS)
    close_timer.single_shot = True
    close_timer.add_callback(_hide_winner_box)
    close_timer.start()
    winner_timers.append(close_timer)


def _schedule_winner_box():
    _stop_winner_timers()
    timer = fig.canvas.new_timer(interval=WINNER_DELAY_MS)
    timer.single_shot = True
    timer.add_callback(_show_winner_box)
    timer.start()
    winner_timers.append(timer)


def _points(mse) -> int:
    """Turn an error into points out of 100 (higher is better, 100 = perfect)."""
    return int(round(100 / (1 + float(mse) / 0.25)))


def _fmt_score(v) -> str:
    return f"{_points(v)} pts"


def _timeout_handler(signum, frame):
    raise TimeoutError("operation timed out")

def _rounded_bg(axis, face, edge="none", radius_px=18, lw=0):
    """Hide the rectangular axes background and draw a rounded one instead."""
    from matplotlib.patches import FancyBboxPatch
    axis.patch.set_visible(False)
    for sp in axis.spines.values():
        sp.set_visible(False)
    bbox = axis.get_window_extent()
    w_px, h_px = max(bbox.width, 1), max(bbox.height, 1)
    patch = FancyBboxPatch((0, 0), 1, 1, transform=axis.transAxes, clip_on=False,
                           boxstyle=f"round,pad=0,rounding_size={radius_px / w_px}",
                           mutation_aspect=w_px / h_px,
                           facecolor=face, edgecolor=edge, linewidth=lw, zorder=0)
    axis.add_patch(patch)
    return patch


def _update_legend(axis: plt.Axes):
    """Place a larger legend box just outside the plot with pretty styling."""
    handles, labels = axis.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique and label and not label.startswith("_"):
            unique[label] = handle

    if not unique:
        axis.legend_.remove() if axis.legend_ else None
        return

    axis.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        borderaxespad=0.9,
        frameon=True,
        facecolor="white",
        edgecolor="#d7d7d7",
        fontsize=11,
        borderpad=0.8,
        labelspacing=0.6,
        handlelength=2.2,
        fancybox=True
    )


# --- AI prediction animation (line grows with a robot at its tip) ---
AI_ANIM_INTERVAL_MS = 70
AI_PAUSE_MS = 2000    # pause after "Valider" before the robot starts drawing
ai_anim_step = None   # None = AI curve fully shown; int = number of points revealed so far
ai_anim_timer = None
ai_line = None
robot_artist = None


_EMOJI_CACHE = {}
# (path, size): bitmap emoji fonts only load at their native size (Noto Color Emoji = 109)
_EMOJI_FONTS = [
    ("/System/Library/Fonts/Apple Color Emoji.ttc", 160),
    ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109),
    ("/usr/share/fonts/noto/NotoColorEmoji.ttf", 109),
    ("/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf", 109),
    ("/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf", 109),
    ("C:/Windows/Fonts/seguiemj.ttf", 160),
]
_EMOJI_FONT = None


def _load_emoji_font():
    global _EMOJI_FONT
    if _EMOJI_FONT is None:
        from PIL import ImageFont
        for path, size in _EMOJI_FONTS:
            try:
                _EMOJI_FONT = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            raise OSError("no colour emoji font found")
    return _EMOJI_FONT


def _emoji_img(chars: str):
    """Render emoji to an RGBA array (matplotlib can't draw colour emoji itself). None if unavailable."""
    if chars not in _EMOJI_CACHE:
        try:
            from PIL import Image, ImageDraw
            font = _load_emoji_font()
            im = Image.new("RGBA", (180 * len(chars) + 20, 200), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((10, 10), chars, font=font, embedded_color=True)
            _EMOJI_CACHE[chars] = np.asarray(im.crop(im.getbbox()))
        except Exception:
            _EMOJI_CACHE[chars] = None
    return _EMOJI_CACHE[chars]


def _emoji_at(text, chars: str, side: str = "left", scale: float = 1.25, pad: float = 0.25):
    """Stick an emoji image next to a Text artist (left, right or top of it)."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    img = _emoji_img(chars)
    if img is None:
        return None
    zoom = text.get_fontsize() * scale / img.shape[0]
    anchor, align = {
        "left": ((0, 0.5), (1 + pad, 0.5)),
        "right": ((1, 0.5), (-pad, 0.5)),
        "top": ((0.5, 1), (0.5, -pad)),
    }[side]
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), anchor, xycoords=text, box_alignment=align,
                        frameon=False, zorder=text.get_zorder() + 1, annotation_clip=False)
    (text.axes if text.axes is not None else text.figure).add_artist(ab)
    return ab


ROBOT_IMG = _emoji_img("\U0001F916")


def _make_robot(xy):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    if ROBOT_IMG is None:
        return ax.plot([xy[0]], [xy[1]], marker="o", markersize=16, color=COLOR_MODEL,
                       markeredgecolor="white", markeredgewidth=2, zorder=10)[0]
    box = AnnotationBbox(OffsetImage(ROBOT_IMG, zoom=0.22), xy, frameon=False, zorder=10)
    ax.add_artist(box)
    return box


def _move_robot(xy):
    if robot_artist is None:
        return
    if hasattr(robot_artist, "xybox"):
        robot_artist.xy = xy
        robot_artist.xybox = xy
    else:
        robot_artist.set_data([xy[0]], [xy[1]])


def _stop_ai_anim():
    global ai_anim_timer, ai_anim_step
    if ai_anim_timer is not None:
        try:
            ai_anim_timer.stop()
        except Exception:
            pass
    ai_anim_timer = None
    ai_anim_step = None


confetti_timer = None
confetti_artist = None


def _stop_confetti():
    global confetti_timer, confetti_artist
    if confetti_timer is not None:
        try:
            confetti_timer.stop()
        except Exception:
            pass
    if confetti_artist is not None:
        try:
            confetti_artist.remove()
        except Exception:
            pass
    confetti_timer = None
    confetti_artist = None


def _launch_confetti(n: int = 140, frames: int = 45):
    """Colourful confetti raining over the whole window (figure coordinates)."""
    global confetti_timer, confetti_artist
    _stop_confetti()
    rng = np.random.default_rng()
    pos = np.column_stack([rng.uniform(0.05, 0.95, n), rng.uniform(0.9, 1.25, n)])
    vel = np.column_stack([rng.normal(0, 0.004, n), rng.uniform(-0.03, -0.012, n)])
    palette = [COLOR_OBS, COLOR_FUTURE, COLOR_PRED, COLOR_DRAWN, COLOR_WINDOW, COLOR_MODEL]
    colors = [palette[i % len(palette)] for i in range(n)]
    from matplotlib.collections import RegularPolyCollection
    confetti_artist = fig.add_artist(RegularPolyCollection(
        4, rotation=np.pi / 4, sizes=rng.uniform(25, 80, n), offsets=pos,
        offset_transform=fig.transFigure, facecolors=colors, edgecolors="none", zorder=300))
    state = {"frame": 0}

    def _tick():
        state["frame"] += 1
        vel[:, 1] -= 0.0008
        pos[:] += vel
        confetti_artist.set_offsets(pos)
        if state["frame"] >= frames:
            _stop_confetti()
        fig.canvas.draw_idle()

    confetti_timer = fig.canvas.new_timer(interval=40)
    confetti_timer.add_callback(_tick)
    confetti_timer.start()
    _launch_confetti._tick = _tick  # handy for headless tests


hint_used = False


def _hint_len() -> int:
    """Number of future points revealed by the hint (last quarter of the zone)."""
    return max(2, len(x_future) // 4)


def _stars_for(points: int) -> int:
    if points >= 90:
        stars = 3
    elif points >= 60:
        stars = 2
    else:
        stars = 1
    return max(1, stars - (1 if hint_used else 0))


def _ai_anim_tick():
    global ai_anim_step
    mp = _normalize_model_pred(model_pred)
    if ai_anim_step is None or mp is None or ai_line is None:
        _stop_ai_anim()
        return
    ai_anim_step += 1
    if ai_anim_step >= len(mp):
        # Done: full redraw reveals the result banner
        _stop_ai_anim()
        _refresh_main_axes()
        if human_mse is not None and mse_pred is not None:
            _robot_say("robot_loses" if human_mse < mse_pred else
                       "robot_wins" if human_mse > mse_pred else "tie")
        _schedule_winner_box()
        return
    ai_line.set_data(x_future[:ai_anim_step], mp[:ai_anim_step])
    _move_robot((x_future[ai_anim_step - 1], mp[ai_anim_step - 1]))
    ax.figure.canvas.draw_idle()


def _start_ai_anim():
    global ai_anim_step, ai_anim_timer
    _stop_ai_anim()
    if _normalize_model_pred(model_pred) is None or fig is None:
        _refresh_main_axes()
        return
    # Short pause first: only the child's curve and the true continuation are visible
    ai_anim_step = 0
    _refresh_main_axes()
    _robot_say("look")

    def _robot_starts():
        global ai_anim_step, ai_anim_timer
        if ai_anim_step != 0:
            return
        ai_anim_step = 1
        _refresh_main_axes()
        _robot_say("thinking")
        ai_anim_timer = fig.canvas.new_timer(interval=AI_ANIM_INTERVAL_MS)
        ai_anim_timer.add_callback(_ai_anim_tick)
        ai_anim_timer.start()

    ai_anim_timer = fig.canvas.new_timer(interval=AI_PAUSE_MS)
    ai_anim_timer.single_shot = True
    ai_anim_timer.add_callback(_robot_starts)
    ai_anim_timer.start()
    _start_ai_anim._robot_starts = _robot_starts  # handy for headless tests


def _clear_drawn_points():
    _stop_ai_anim()
    _stop_confetti()
    _stop_winner_timers()
    _hide_winner_box(redraw=False)
    drawn_x.clear()
    drawn_y.clear()
    global human_pred, human_mse, model_pred, mse_pred
    human_pred = None
    human_mse = None
    model_pred = None
    mse_pred = None


def _normalize_model_pred(p):
    """Return a 1D numpy float array for the model prediction or None.

    Accepts lists, scalars, numpy arrays. On failure returns None.
    """
    if p is None:
        return None
    try:
        arr = np.asarray(p, dtype=float).reshape(-1)
        return arr
    except Exception:
        return None


# --- Robot speech bubble ---
ROBOT_LINES = {
    "hello": ["Salut ! Je suis Robo. Essaie de me battre !"],
    "new": ["Hmm, celle-là a l'air facile...", "Je parie que je vais gagner !",
            "À toi de jouer, humain !", "Je calcule déjà la suite... bip bip",
            "Prêt ? Moi je suis toujours prêt !", "Tu vas voir ce que tu vas voir !"],
    "draw": ["Hmm, intéressant...", "Tu es sûr de toi ?", "Pas mal, pas mal...",
             "Oh, audacieux !"],
    "hint": ["Un indice ? Petit malin !", "Hé, c'est presque de la triche !"],
    "look": ["Voyons voir ce que tu as fait...", "Hmm, pas mal... à mon tour !"],
    "thinking": ["Bip... boup... je calcule !", "Mes circuits chauffent..."],
    "robot_wins": ["Bip boup, victoire !", "Les robots sont trop forts !",
                   "Essaie encore, humain !"],
    "robot_loses": ["Impossible... tu es trop fort !", "Bug dans mes circuits !",
                    "Bravo, tu m'as battu !"],
    "tie": ["Égalité ! On rejoue ?"],
}
robot_bubble = None


def _robot_say(kind: str):
    if robot_bubble is None:
        return
    robot_bubble.set_text(random.choice(ROBOT_LINES[kind]))
    if fig is not None:
        fig.canvas.draw_idle()


# --- Past curve drawn from left to right when a new curve appears ---
PAST_ANIM_FRAMES = 25
past_anim_step = None   # None = past fully shown; int = number of past points revealed
past_anim_timer = None
past_line = None
past_tip = None


def _start_past_anim():
    global past_anim_step, past_anim_timer
    if past_anim_timer is not None:
        past_anim_timer.stop()
    i0, _ = _view_bounds()
    start = min(i0, n_obs - 1)
    step = max(1, (n_obs - start) // PAST_ANIM_FRAMES)
    past_anim_step = start + 1
    _refresh_main_axes()

    def _tick():
        global past_anim_step, past_anim_timer
        if past_anim_step is None or past_line is None:
            return
        past_anim_step += step
        if past_anim_step >= n_obs:
            past_anim_step = None
            past_anim_timer.stop()
            past_anim_timer = None
            past_line.set_data(x_obs, y_obs)
            past_tip.set_data([], [])
        else:
            past_line.set_data(x_obs[:past_anim_step], y_obs[:past_anim_step])
            past_tip.set_data([x_obs[past_anim_step - 1]], [y_obs[past_anim_step - 1]])
        fig.canvas.draw_idle()

    past_anim_timer = fig.canvas.new_timer(interval=40)
    past_anim_timer.add_callback(_tick)
    past_anim_timer.start()
    _start_past_anim._tick = _tick  # handy for headless tests


# --- Magic pencil sparkles ---
SPARKLE_LIFE = 12
sparkles = []           # [x, y, age, colour]
sparkle_artist = None
sparkle_timer = None


def _add_sparkles(xd, yd, n: int = 2):
    global sparkle_timer
    if ax is None:
        return
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for _ in range(n):
        sparkles.append([xd + random.gauss(0, 0.015) * (x1 - x0), yd + random.gauss(0, 0.04) * (y1 - y0), 0,
                         random.choice(["#ffd166", "#ff8fab", "#7ae7ff", "#c77dff", "#ffffff"])])
    if sparkle_timer is None:
        sparkle_timer = fig.canvas.new_timer(interval=50)
        sparkle_timer.add_callback(_sparkle_tick)
        sparkle_timer.start()


def _sparkle_tick():
    global sparkle_timer
    from matplotlib.colors import to_rgba
    for s in sparkles:
        s[2] += 1
        s[1] += 0.004 * (ax.get_ylim()[1] - ax.get_ylim()[0])   # drift upwards
    sparkles[:] = [s for s in sparkles if s[2] < SPARKLE_LIFE]
    if sparkle_artist is not None:
        if sparkles:
            sparkle_artist.set_offsets([(s[0], s[1]) for s in sparkles])
            sparkle_artist.set_sizes([140 * (1 - s[2] / SPARKLE_LIFE) + 20 for s in sparkles])
            sparkle_artist.set_facecolors([to_rgba(s[3], 1 - s[2] / SPARKLE_LIFE) for s in sparkles])
        else:
            sparkle_artist.set_offsets(np.empty((0, 2)))
    if not sparkles and sparkle_timer is not None:
        sparkle_timer.stop()
        sparkle_timer = None
    fig.canvas.draw_idle()


# --- Player avatar following the pencil ---
def _place_avatar(xy):
    """Show the player's avatar just above-left of a point of the main plot (None hides it)."""
    global avatar_artist
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    if avatar_artist is not None:
        try:
            avatar_artist.remove()
        except Exception:
            pass
        avatar_artist = None
    img = _emoji_img(AVATAR)
    if xy is None or img is None:
        return
    avatar_artist = AnnotationBbox(OffsetImage(img, zoom=30 / img.shape[0]), xy, xybox=(-16, 16),
                                   boxcoords="offset points", frameon=False, zorder=11)
    ax.add_artist(avatar_artist)


# --- Level themes ---
def _apply_theme():
    """Recolour the window and decorations for the current level."""
    global FIG_FACE, AX_FACE, TEXT_DARK, TEXT_MUTED
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    theme = THEMES.get(os.path.basename(current_dataset_name or ""), THEMES["simple.npy"])
    FIG_FACE, AX_FACE, TEXT_DARK, TEXT_MUTED = theme["fig"], theme["ax"], theme["text"], theme["muted"]
    if fig is None:
        return
    fig.set_facecolor(FIG_FACE)
    for a in theme_artists:
        try:
            a.remove()
        except Exception:
            pass
    theme_artists.clear()
    for (fx, fy), emo in zip(DECO_SPOTS, theme["deco"] + theme["deco"][:1]):
        img = _emoji_img(emo)
        if img is not None:
            ab = AnnotationBbox(OffsetImage(img, zoom=30 / img.shape[0]), (fx, fy), xycoords="figure fraction",
                                frameon=False, zorder=1)
            fig.add_artist(ab)
            theme_artists.append(ab)
    if radio_ax is not None:
        radio_ax.title.set_color(TEXT_DARK)


# --- Draggable view window (mini overview of the whole signal under the plot) ---
VIEW_POINTS = 3 * FORECAST_HORIZON   # points visible in the main plot
view_start_idx = None   # first visible index; None = latest window (ends with the yellow zone)
overview_ax = None
view_rect = None
dragging_view = False


def _view_bounds():
    i_max = max(0, len(x) - VIEW_POINTS)
    i0 = i_max if view_start_idx is None else int(np.clip(view_start_idx, 0, i_max))
    return i0, min(len(x) - 1, i0 + VIEW_POINTS - 1)


def _apply_view(redraw: bool = True):
    """Fit the main plot to the window selected in the overview."""
    i0, i1 = _view_bounds()
    ax.set_xlim(x[i0], x[i1])
    # Only use what the player is allowed to know: the past (+ the future once validated)
    vals = [y[i0:min(i1 + 1, n_obs)]]
    if human_pred is not None:
        vals += [y_future, human_pred]
        mp = _normalize_model_pred(model_pred)
        if mp is not None:
            vals.append(mp)
    elif hint_used:
        vals.append(y_future[-_hint_len():])
    if drawn_y:
        vals.append(np.asarray(drawn_y))
    seg = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in vals if len(v)])
    lo, hi = np.nanmin(seg), np.nanmax(seg)
    if human_pred is None and i1 >= n_obs:
        # yellow zone visible: leave room above/below to draw the continuation
        pad = max(0.35 * (hi - lo), 0.3)
    else:
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
    ax.set_ylim(lo - pad, hi + pad)
    if view_rect is not None:
        view_rect.set_x(x[i0])
        view_rect.set_width(x[i1] - x[i0])
    if redraw:
        ax.figure.canvas.draw_idle()


def _move_view_to(xdata):
    global view_start_idx
    view_start_idx = int(np.searchsorted(x, xdata)) - VIEW_POINTS // 2
    _apply_view()


def _refresh_overview():
    global view_rect
    if overview_ax is None or x_obs is None:
        return
    from matplotlib.patches import Rectangle
    overview_ax.clear()
    overview_ax.set_facecolor("#ffffff")
    overview_ax.set_xticks([])
    overview_ax.set_yticks([])
    for sp in overview_ax.spines.values():
        sp.set_edgecolor("#c9d3ee")
        sp.set_linewidth(1.5)
    overview_ax.plot(x_obs, y_obs, color=COLOR_OBS, linewidth=1.5)
    overview_ax.axvspan(x_future[0], x_future[-1], color=COLOR_WINDOW, alpha=0.5)
    if human_pred is not None:
        overview_ax.plot(np.r_[x_obs[-1], x_future], np.r_[y_obs[-1], y_future], color=COLOR_OBS, linewidth=1.5)
    overview_ax.set_xlim(x[0], x[-1])
    known = y if human_pred is not None else y_obs
    lo, hi = np.nanmin(known), np.nanmax(known)
    pad = 0.1 * (hi - lo) if hi > lo else 0.1
    overview_ax.set_ylim(lo - pad, hi + pad)
    i0, i1 = _view_bounds()
    view_rect = Rectangle((x[i0], lo - pad), x[i1] - x[i0], hi - lo + 2 * pad,
                          facecolor="#4d96ff", alpha=0.18, edgecolor="#4d96ff", linewidth=2.5)
    overview_ax.add_patch(view_rect)
    t = overview_ax.text(0.035, 1.18, "Glisse le cadre bleu pour voir le passé",
                         transform=overview_ax.transAxes, ha="left", va="center",
                         fontsize=11, color=TEXT_MUTED, clip_on=False)
    _emoji_at(t, "\U0001F449", "left", scale=1.3)


def _refresh_main_axes():
    """Render the observed signal and prediction window on the main axes."""
    global ax
    if ax is None or x_obs is None:
        return
    ax.clear()
    ax.set_facecolor(AX_FACE)
    if ax.figure is not None:
        ax.figure.set_facecolor(FIG_FACE)

    global past_line, past_tip, sparkle_artist
    n_past = n_obs if past_anim_step is None else past_anim_step
    past_line, = ax.plot(x_obs[:n_past], y_obs[:n_past], label="La vraie courbe", color=COLOR_OBS, linewidth=3,
                         solid_capstyle="round")
    past_tip, = ax.plot([x_obs[n_past - 1]] if past_anim_step is not None else [],
                        [y_obs[n_past - 1]] if past_anim_step is not None else [],
                        marker="o", markersize=11, color=COLOR_OBS, markeredgecolor="white",
                        markeredgewidth=2, zorder=8)
    sparkle_artist = ax.scatter([], [], marker="*", s=[], zorder=12, linewidths=0)
    ax.axvline(x_obs[-1], color=COLOR_OBS, linestyle="--", linewidth=1.5)
    theme = THEMES.get(os.path.basename(current_dataset_name or ""), {})
    ax.axvspan(x_future[0], x_future[-1], color=theme.get("zone", COLOR_WINDOW), alpha=theme.get("zone_alpha", 0.3))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#c9d3ee")
        sp.set_linewidth(2)

    # Model prediction (use normalized array)
    if model_pred is not None:
        mp = _normalize_model_pred(model_pred)
        if mp is not None and len(mp) == len(x_future):
            global ai_line, robot_artist
            n = len(mp) if ai_anim_step is None else ai_anim_step
            ai_line, = ax.plot(x_future[:n], mp[:n], color=COLOR_MODEL, linewidth=3,
                               label="Le robot" if n > 0 else "_nolegend_", solid_capstyle="round")
            # n == 0: pause before the robot starts, only the child's curve and the truth are shown
            robot_artist = _make_robot((x_future[n - 1], mp[n - 1])) if n > 0 else None

    # Human validated prediction, with the gap to the truth shaded
    if human_pred is not None and len(human_pred) == len(x_future):
        ax.fill_between(x_future, human_pred, y_future, color=COLOR_PRED, alpha=0.18, linewidth=0)
        # true continuation in the same colour as the past, joined to it
        ax.plot(np.r_[x_obs[-1], x_future], np.r_[y_obs[-1], y_future], color=COLOR_OBS, linewidth=3,
                solid_capstyle="round", label="_nolegend_")
        ax.plot(x_future, human_pred, color=COLOR_PRED, linewidth=3, label="Ta courbe")
    elif hint_used:
        k = _hint_len()
        ax.plot(x_future[-k:], y_future[-k:], color=COLOR_OBS, linewidth=3, linestyle=(0, (2, 2)),
                label="Indice")
        img = _emoji_img("\U0001F4A1")
        if img is not None:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=22 / img.shape[0]),
                                         (x_future[-k], y_future[-k]), xybox=(0, 18),
                                         boxcoords="offset points", frameon=False, zorder=9))

    # Drawn points (pre/post validation)
    global drawn_line, hint_artist
    drawn_line, = ax.plot([], [], color=COLOR_DRAWN, linewidth=4, marker="o", markersize=5,
                          markeredgecolor="white", solid_capstyle="round",
                          label="Ta courbe" if human_pred is None else "_nolegend_")
    _update_drawn_line()
    drawn_line.set_visible(human_pred is None)
    global avatar_artist
    avatar_artist = None   # ax.clear() already removed it
    if human_pred is not None:
        _place_avatar((x_future[-1], human_pred[-1]))
    elif drawn_x:
        _place_avatar((drawn_x[-1], drawn_y[-1]))
    hint_artist = None
    if not drawn_x and human_pred is None:
        hint_artist = ax.text((x_future[0] + x_future[-1]) / 2, 0.45, "Dessine\nla suite\nici !",
                              transform=ax.get_xaxis_transform(), ha="center", va="center",
                              fontsize=17, fontweight="bold", color="#e07a00", linespacing=1.4,
                              bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.8", facecolor="#ffffff",
                                        edgecolor=COLOR_WINDOW, linewidth=2.5, alpha=0.9))
        hint_emoji = _emoji_at(hint_artist, "\u270F\uFE0F", side="top", scale=1.8, pad=0.9)
        hint_artist._emoji = hint_emoji

    _refresh_overview()
    _apply_view(redraw=False)
    ax.set_autoscale_on(False)
    # Title: instructions while playing, gentle result banner once validated
    ds_base = _dataset_display_name(current_dataset_name or "?")
    stock_part = f" | Action : {current_stock_name}" if current_stock_name else ""
    _update_legend(ax)

    human_val = float(human_mse) if human_mse is not None and np.isfinite(human_mse) else None
    ai_val = float(mse_pred) if mse_pred is not None and np.isfinite(mse_pred) else None

    if human_pred is not None and ai_anim_step == 0:
        t = ax.set_title("Compare ta courbe avec la vraie suite !\nLe robot arrive...",
                         fontsize=18, fontweight="bold", color=TEXT_DARK, pad=14)
        _emoji_at(t, "\U0001F440", "left", scale=1.6)
    elif human_pred is not None and ai_anim_step is not None:
        t = ax.set_title("Le robot dessine sa prédiction...\nQui sera le plus proche ?",
                         fontsize=18, fontweight="bold", color=COLOR_MODEL, pad=14)
        _emoji_at(t, "\U0001F916", "left", scale=1.6)
        _emoji_at(t, "\U0001F914", "right", scale=1.6)
    elif human_pred is None or human_val is None:
        t = ax.set_title(f"Devine la suite de la courbe !\nNiveau : {ds_base}{stock_part}",
                         fontsize=18, fontweight="bold", color=TEXT_DARK, pad=14)
        _emoji_at(t, "\U0001F52E", "left", scale=1.6)
    else:
        WIN, LOSE, TIE = "#2bb673", "#f25f5c", "#6c757d"
        if ai_val is None:
            msg, col, emo = "Voici la vraie suite !", TEXT_DARK, "\U0001F440"
            human_col = ai_col = TIE
        elif human_val < ai_val:
            msg, col, emo = "Bravo, tu as battu le robot !", WIN, "\U0001F389"
            human_col, ai_col = WIN, LOSE
        elif human_val > ai_val:
            msg, col, emo = "Le robot gagne cette fois... Essaie encore !", LOSE, "\U0001F4AA"
            human_col, ai_col = LOSE, WIN
        else:
            msg, col, emo = "Égalité !", TIE, "\U0001F91D"
            human_col = ai_col = TIE
        kw = dict(transform=ax.transAxes, va="center", clip_on=False)
        t = ax.text(0.5, 1.13, msg, ha="center", fontsize=20, fontweight="bold", color=col, **kw)
        _emoji_at(t, emo, "left")
        _emoji_at(t, emo, "right")
        t = ax.text(0.46, 1.045, f"Toi : {_fmt_score(human_val)}", ha="right", fontsize=16,
                    fontweight="bold", color=human_col, **kw)
        _emoji_at(t, AVATAR, "left")
        ax.text(0.5, 1.045, "vs", ha="center", fontsize=14, color=TEXT_MUTED, **kw)
        t = ax.text(0.57, 1.045, f"Robot : {_fmt_score(ai_val) if ai_val is not None else '-'}", ha="left",
                    fontsize=16, fontweight="bold", color=ai_col, **kw)
        _emoji_at(t, "\U0001F916", "left")
        t = ax.text(0.01, 1.045, "Ton dessin :", ha="left", fontsize=12, color=TEXT_MUTED, **kw)
        _emoji_at(t, "\u2B50" * _stars_for(_points(human_val)), "right", scale=1.3, pad=0.08)
        ax.text(0.99, 1.045, "zone rose = ton écart", ha="right", fontsize=11, color=TEXT_MUTED, **kw)

    # Update external ranking panel if present (preferred) otherwise draw nothing here
    try:
        global ranking_ax
        if ranking_ax is not None:
            # Clear and render the ranking info inside ranking_ax
            ranking_ax.clear()
            ranking_ax.set_xticks([])
            ranking_ax.set_yticks([])
            ranking_ax.set_facecolor("#ffffff")
            try:
                res = get_results()
                if res is not None and len(res) >= 3:
                    mean_human, mean_ai, count = res[0], res[1], int(res[2])
                else:
                    mean_human, mean_ai, count = None, None, 0
            except Exception:
                mean_human, mean_ai, count = None, None, 0

            # Colors and panel styling
            WIN = "#2ecc71"       # green
            LOSE = "#ff6b6b"      # red
            NEUTRAL = "#6c757d"   # grey
            PANEL_BG = "#ffffff"
            PANEL_EDGE = "#d7d7d7"

            # draw a rounded panel background inside the small axes
            try:
                _rounded_bg(ranking_ax, PANEL_BG, edge="#d6def5", lw=1.5)
            except Exception:
                pass

            # Compute per-line colors (lower loss is better)
            human_col = NEUTRAL
            ai_col = NEUTRAL
            try:
                if mean_human is not None and not np.isnan(mean_human) and mean_ai is not None and not np.isnan(mean_ai):
                    if mean_human < mean_ai:
                        human_col, ai_col = WIN, LOSE
                    elif mean_ai < mean_human:
                        human_col, ai_col = LOSE, WIN
                    else:
                        human_col = ai_col = NEUTRAL
                else:
                    if mean_human is not None and not np.isnan(mean_human):
                        human_col = WIN
                    if mean_ai is not None and not np.isnan(mean_ai):
                        ai_col = WIN
            except Exception:
                human_col = ai_col = NEUTRAL

            # Draw lines separately so each can have its own color and style
            rk = dict(va="center", transform=ranking_ax.transAxes)
            t = ranking_ax.text(0.56, 0.80, "Classement", ha="center", fontsize=13, fontweight="bold",
                                color="#2d3047", **rk)
            _emoji_at(t, "\U0001F3C6", "left")

            has_h = mean_human is not None and not np.isnan(mean_human)
            has_ai = mean_ai is not None and not np.isnan(mean_ai)
            t = ranking_ax.text(0.24, 0.52, _fmt_score(mean_human) if has_h else "-", ha="left", fontsize=13,
                                fontweight="bold", color=human_col if has_h else NEUTRAL, **rk)
            _emoji_at(t, AVATAR, "left", scale=1.4)
            t = ranking_ax.text(0.68, 0.52, _fmt_score(mean_ai) if has_ai else "-", ha="left", fontsize=13,
                                fontweight="bold", color=ai_col if has_ai else NEUTRAL, **rk)
            _emoji_at(t, "\U0001F916", "left", scale=1.4)

            t = ranking_ax.text(0.55, 0.2, f"{count} manches jouées", ha="center", fontsize=10,
                                color="#555", **rk)
            _emoji_at(t, "\U0001F3AE", "left")
    except Exception:
        pass

    ax.figure.canvas.draw_idle()


def _generate_new_signal(index: int | None = None):
    """Sample a new signal from the dataset and update global partitions."""
    global x, y, x_obs, y_obs, x_future, y_future, n_obs, y_lim, current_signal_index, model_pred, mse_pred, current_stock_name, MAX_CTX

    if data is None or (not isinstance(data, dict) and getattr(data, 'size', 0) == 0):
        if not isinstance(data, dict):
            raise ValueError("Dataset is empty; cannot generate signal.")

    # Select a series depending on dataset type
    if isinstance(data, dict):
        keys = list(data.keys())
        if not keys:
            raise ValueError("No stocks in dictionary dataset.")
        key = random.choice(keys) if index is None else keys[index % len(keys)]
        series_full = np.asarray(data[key], dtype=float).reshape(-1)
        current_stock_name = str(key)
        series_len = len(series_full)
        ctx_len = max(2, min(MAX_CTX, max(0, series_len - n_future)))
        tail_len = ctx_len + n_future
        y_series = series_full[-tail_len:]
        current_signal_index = keys.index(key)
    else:
        # numpy array
        total_rows = int(data.shape[0])
        if index is None:
            index = random.randint(0, total_rows - 1)
        current_signal_index = index
        row = np.asarray(data[index], dtype=float).reshape(-1)
        series_len = len(row)
        ctx_len = max(2, min(MAX_CTX, max(0, series_len - n_future)))
        tail_len = ctx_len + n_future
        y_series = row[-tail_len:]
        current_stock_name = None

    # Normalize with the past only so the scale gives no hint about the future
    past = y_series[:-n_future]
    y_series = (y_series - np.mean(past)) / (np.std(past) + 0.0005)
    x_series = np.linspace(0, len(y_series), len(y_series))

    x = x_series
    y = y_series

    assert n_future < len(x), "n_future must be smaller than number of samples"
    n_obs = len(x) - n_future

    x_obs, y_obs = x[:n_obs], y[:n_obs]
    x_future, y_future = x[n_obs:], y[n_obs:]

    model_pred = None
    mse_pred = None
    # reset human validated prediction
    global human_pred, human_mse, hint_used
    human_pred = None
    human_mse = None
    hint_used = False
    global view_start_idx
    view_start_idx = None

# --- Points drawn by the child ---
drawn_x = []
drawn_y = []
drawn_line = None

def _update_drawn_line():
    """Show the drawn points as one continuous line (sorted left to right)."""
    if drawn_line is None:
        return
    if drawn_x:
        order = np.argsort(drawn_x)
        drawn_line.set_data(np.asarray(drawn_x)[order], np.asarray(drawn_y)[order])
    else:
        drawn_line.set_data([], [])

def _add_drawn_point(xd, yd):
    global hint_artist
    drawn_x.append(xd)
    drawn_y.append(yd)
    if hint_artist is not None:
        try:
            if getattr(hint_artist, "_emoji", None) is not None:
                hint_artist._emoji.remove()
            hint_artist.remove()
        except Exception:
            pass
        hint_artist = None
    _update_drawn_line()
    if len(drawn_x) == 1:
        _robot_say("draw")
    _place_avatar((xd, yd))
    _add_sparkles(xd, yd)
    ax.figure.canvas.draw_idle()

# --- Utility: attempt to maximize the figure window (best-effort across backends) ---
def _maximize_current_figure():
    try:
        mng = plt.get_current_fig_manager()
        # Generic (TkAgg often supports this)
        if hasattr(mng, "full_screen_toggle"):
            try:
                mng.full_screen_toggle()
                return
            except Exception:
                pass
        # Qt / others expose a window attribute
        if hasattr(mng, "window"):
            try:
                # Qt
                mng.window.showMaximized()
                return
            except Exception:
                pass
            try:
                # Tk
                mng.window.state("zoomed")
                return
            except Exception:
                pass
    except Exception:
        # If the backend doesn't support maximizing, silently ignore
        pass


def _validate_and_plot():
    """Validate current drawn points, compute MSE, and plot without spawning new figures."""
    global ax
    if x_future is None or ax is None:
        return

    print("Points dessinés (x, y) :")
    for xx, yy in zip(drawn_x, drawn_y):
        print(f"{xx:.2f}, {yy:.2f}")

    if not drawn_x:
        print("Aucun point pour valider.")
        return

    x_arr = np.asarray(drawn_x, dtype=float)
    y_arr = np.asarray(drawn_y, dtype=float)
    valid = (
        np.isfinite(x_arr) & np.isfinite(y_arr) &
        (x_arr >= x_future[0]) & (x_arr <= x_future[-1])
    )
    x_arr, y_arr = x_arr[valid], y_arr[valid]

    if x_arr.size == 0:
        print("Les points sortent de la zone jaune.")
        return

    order = np.argsort(x_arr)
    x_sorted, y_sorted = x_arr[order], y_arr[order]
    uniq_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    y_sums = np.bincount(inverse, weights=y_sorted, minlength=uniq_x.size)
    uniq_y = y_sums / counts

    if uniq_x.size >= 2:
        f = interp1d(uniq_x, uniq_y, kind="linear", bounds_error=False, fill_value="extrapolate")
        y_pred = f(x_future).astype(float)
    else:
        y_pred = np.full_like(y_future, fill_value=float(uniq_y[0]))

    finite_mask = np.isfinite(y_future) & np.isfinite(y_pred)
    if finite_mask.any():
        mse = mean_squared_error(y_future[finite_mask], y_pred[finite_mask])
    else:
        mse = float("nan")
        print("Attention : pas assez de points pour calculer l'erreur.")

    # Calculate model MSE if model prediction exists
    global mse_pred
    if model_pred is not None:
        mp = _normalize_model_pred(model_pred)
        if mp is not None and len(mp) == len(y_future):
            finite_mask_model = np.isfinite(y_future) & np.isfinite(mp)
            if finite_mask_model.any():
                mse_pred = mean_squared_error(y_future[finite_mask_model], mp[finite_mask_model])
            else:
                mse_pred = float("nan")
        else:
            mse_pred = None
    

    # Store human prediction and loss, then refresh view to consistently display
    global human_pred, human_mse
    human_pred = y_pred
    human_mse = mse

    if mse_pred is not None and human_mse is not None:
        add_result(human_mse, mse_pred)
        print(f"Résultats sauvegardés dans results.json (Human loss: {human_mse:.4f}, AI loss: {mse_pred:.4f})")
    else:
        print(f"Résultats non sauvegardés (Human loss: {human_mse:.4f}, AI loss: {mse_pred})")

    _start_ai_anim()
    _update_validate_button_state()

    _update_legend(ax)
    ax.figure.canvas.draw_idle()

# --- Event functions ---
def on_press(event):
    global dragging_view
    if x_future is None:
        return
    if welcome_active:
        _welcome_click(event)
        return
    if _hide_winner_box():
        return
    if overview_ax is not None and event.inaxes is overview_ax and event.xdata is not None:
        dragging_view = True
        _move_view_to(event.xdata)
        return
    # Only allow drawing inside the future window and with finite coordinates
    if (
        event.inaxes is ax
        and event.xdata is not None and np.isfinite(event.xdata)
        and event.ydata is not None and np.isfinite(event.ydata)
        and x_future[0] <= event.xdata <= x_future[-1]
    ):
        if human_pred is not None:
            _show_popup("Clique sur « Nouvelle courbe » pour rejouer !", face_color="#6c5ce7")
            return
        _add_drawn_point(event.xdata, event.ydata)
    _update_validate_button_state()

def on_move(event):
    # While dragging with left button, constrain to the future window and finite values
    if x_future is None or welcome_active:
        return
    if dragging_view:
        if event.inaxes is overview_ax and event.xdata is not None:
            _move_view_to(event.xdata)
        return
    if (
        event.inaxes is ax and event.button == 1
        and event.xdata is not None and np.isfinite(event.xdata)
        and event.ydata is not None and np.isfinite(event.ydata)
        and x_future[0] <= event.xdata <= x_future[-1]
        and human_pred is None
    ):
        _add_drawn_point(event.xdata, event.ydata)
    _update_validate_button_state()

def on_release(event):
    global dragging_view
    dragging_view = False


def on_key(event):
    global drawn_x, drawn_y
    if welcome_active:
        if event.key == "enter":
            _close_welcome()
        return
    if event.key == "enter":
        _validate_and_plot()

    elif event.key == "r":
        # Reset to empty target window: clear points and restore observed-only view
        _clear_drawn_points()
        _refresh_main_axes()


def on_new_signal_button(event):
    """Button callback: sample a new signal and reset drawing area."""
    _generate_new_signal()
    _clear_drawn_points()
    _start_past_anim()
    _update_validate_button_state()
    _robot_say("new")
    print(f"Nouveau signal prêt (n°{current_signal_index}).")


def on_hint_button(event):
    """Button callback: reveal the start of the true continuation (costs one star)."""
    global hint_used
    if human_pred is not None:
        _show_popup("Clique sur « Nouvelle courbe » pour rejouer !", face_color="#6c5ce7")
        return
    if hint_used:
        _show_popup("Tu as déjà ton indice pour cette courbe !", face_color="#ff9f1c")
        return
    hint_used = True
    _refresh_main_axes()
    _update_validate_button_state()
    _robot_say("hint")
    _show_popup("Voici où arrive la courbe ! (une étoile en moins)", face_color="#ff9f1c")


def on_erase_button(event):
    """Button callback: erase current drawn points without changing signal."""
    _clear_drawn_points()
    _refresh_main_axes()
    _update_validate_button_state()
    print("Zone propre, tu peux recommencer !")


def on_validate_button(event):
    """Button callback: run validation and display prediction vs ground truth."""
    if human_pred is not None:
        _show_popup("Clique sur « Nouvelle courbe » pour rejouer !", face_color="#6c5ce7")
        return
    if not drawn_x:
        _show_popup("Dessine d'abord dans la zone à droite !", face_color="#ff9f1c")
        return
    # Ensure model prediction is available before validating so mse_pred can be saved.
    # If model is not currently shown, request it (this may block up to the model timeout).
    global model_pred
    if model_pred is None:
        on_model_button(event)
    _validate_and_plot()


def on_model_button(event):
    """Button callback: call the trained model and display its prediction (with 5s timeout)."""
    global model_pred, mse_pred
    if model_pred is not None:
        model_pred = None
        print("Courbe du modèle masquée.")
        _refresh_main_axes()
        return

    if y_obs is None:
        print("Pas de signal chargé.")
        return
    try:
        prev = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)
        try:
            x = np.asarray(y_obs, dtype=float).tolist()
            x = torch.tensor(x).unsqueeze(0)
            pred, _ = model(x, forecast_horizon=n_future, quantiles=[0.5])
            pred = pred[0].detach().cpu().numpy().tolist()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)
        pred = np.asarray(pred, dtype=float).reshape(-1)
    except TimeoutError:
        _show_popup("Le robot dort... réessaie !", duration_ms=2000, face_color="#ff6f6f")
        return
    except Exception as exc:
        print(f"Oups, le modèle a eu un souci : {exc}")
        _show_popup("Le robot dort... réessaie !", duration_ms=2000, face_color="#ff6f6f")
        return

    if pred.size != len(x_future):
        print("Le modèle n'a pas renvoyé la bonne taille de suite.")
        return

    model_pred = pred
    
    # Calculate model MSE against ground truth
    finite_mask_model = np.isfinite(y_future) & np.isfinite(model_pred)
    if finite_mask_model.any():
        mse_pred = mean_squared_error(y_future[finite_mask_model], model_pred[finite_mask_model])
    else:
        mse_pred = float("nan")
        
    print("Courbe du modèle affichée.")
    _refresh_main_axes()

def on_radio_dataset(label):
    """RadioButtons callback: switch dataset to the selected label and refresh."""
    global current_dataset_index
    # Map from displayed label back to actual filename
    target = _dataset_name_from_label(label)
    prev_index = current_dataset_index
    if _load_dataset_by_name(target):
        _generate_new_signal()
        _clear_drawn_points()
        _apply_theme()
        _start_past_anim()
        _update_validate_button_state()
        _robot_say("new")
        print(f"Jeu de données sélectionné: {target}")
    else:
        _show_popup(f"Fichier {target} introuvable", duration_ms=2200)
        # revert selection
        try:
            if radio_datasets is not None:
                radio_datasets.set_active(prev_index)
        except Exception:
            pass


def on_close_button(event):
    """Button callback: close the Matplotlib window."""
    if fig is not None:
        print("À bientôt !")
        plt.close(fig)

# --- Display setup ---
# Load initial dataset (try simple, then medium, then stocks dict)
if not _load_dataset_by_name("simple.npy"):
    if not _load_dataset_by_name("medium.npy"):
        if not _load_dataset_by_name("dic_stocks.pkl"):
            raise FileNotFoundError("Aucun dataset trouvé: simple.npy, medium.npy, ni dic_stocks.pkl")

_generate_new_signal()

fig, ax = plt.subplots(figsize=(14, 6))
plt.subplots_adjust(left=0.04, right=0.76, top=0.86, bottom=0.37)
overview_ax = fig.add_axes([0.04, 0.235, 0.72, 0.075])
_maximize_current_figure()

_clear_drawn_points()
_refresh_main_axes()

# --- Buttons (Pygame-style palette) ---
BUTTON_FACE = "#4d96ff"   # bleu vif
BUTTON_HOVER = "#7ab4ff"
BUTTON_FACE_IA = "#ff03d9"
BUTTON_HOVER_IA = "#ff6efc"
BUTTON_FACE_CHECK = "#2ecc71"  # vert
BUTTON_HOVER_CHECK = "#58d68d"
BUTTON_LABEL = "#ffffff"
CLOSE_FACE = "#ff4b5c"
CLOSE_HOVER = "#ff6f91"

def _style_button(btn, radius_px=18):
    btn._rpatch = _rounded_bg(btn.ax, btn.color, radius_px=radius_px)
    btn._hovered = False


def _paint_button(btn):
    if getattr(btn, "_rpatch", None) is not None:
        btn._rpatch.set_facecolor(btn.hovercolor if btn._hovered else btn.color)


def on_hover(event):
    changed = False
    for btn in (button_new_signal, button_erase, button_hint, button_validate, button_close):
        hovered = event.inaxes is btn.ax
        if hovered != getattr(btn, "_hovered", False):
            btn._hovered = hovered
            _paint_button(btn)
            changed = True
    if changed:
        fig.canvas.draw_idle()


button_new_ax = fig.add_axes([0.80, 0.40, 0.18, 0.10])
button_new_ax.set_zorder(5)
button_new_signal = Button(button_new_ax, "Nouvelle courbe", color=BUTTON_FACE, hovercolor=BUTTON_HOVER)
button_new_signal.label.set_color(BUTTON_LABEL)
button_new_signal.label.set_fontweight("bold")
button_new_signal.label.set_fontsize(14)
button_new_signal.on_clicked(on_new_signal_button)
button_new_signal.label.set_position((0.56, 0.5))
_emoji_at(button_new_signal.label, "\U0001F3B2", "left", scale=1.4)

button_erase_ax = fig.add_axes([0.80, 0.27, 0.18, 0.10])
button_erase_ax.set_zorder(5)
button_erase = Button(button_erase_ax, "Recommencer", color=BUTTON_FACE, hovercolor=BUTTON_HOVER)
button_erase.label.set_color(BUTTON_LABEL)
button_erase.label.set_fontweight("bold")
button_erase.label.set_fontsize(14)
button_erase.on_clicked(on_erase_button)
button_erase.label.set_position((0.56, 0.5))
_emoji_at(button_erase.label, "\U0001F9FD", "left", scale=1.4)

#button_model_ax = fig.add_axes([0.27, 0.06, 0.18, 0.09]) 
#button_model_ax.set_zorder(5)
#button_model = Button(button_model_ax, "Voir la prédiction de l'IA 🤖 ", color=BUTTON_FACE_IA, hovercolor=BUTTON_HOVER_IA)
#button_model.label.set_color(BUTTON_LABEL)
#button_model.label.set_fontweight("bold")
#button_model.on_clicked(on_model_button)

button_hint_ax = fig.add_axes([0.80, 0.14, 0.18, 0.10])
button_hint_ax.set_zorder(5)
button_hint = Button(button_hint_ax, "Indice", color="#ffb703", hovercolor="#ffc94d")
button_hint.label.set_color(BUTTON_LABEL)
button_hint.label.set_fontweight("bold")
button_hint.label.set_fontsize(14)
button_hint.on_clicked(on_hint_button)
button_hint.label.set_position((0.56, 0.5))
_emoji_at(button_hint.label, "\U0001F4A1", "left", scale=1.4, pad=0.6)

button_validate_ax = fig.add_axes([0.30, 0.05, 0.28, 0.13])
button_validate_ax.set_zorder(5)
button_validate = Button(button_validate_ax, "Valider !", color=BUTTON_FACE_CHECK, hovercolor=BUTTON_HOVER_CHECK)
button_validate.label.set_color(BUTTON_LABEL)
button_validate.label.set_fontweight("bold")
button_validate.label.set_fontsize(20)
button_validate.on_clicked(on_validate_button)
button_validate.label.set_position((0.55, 0.5))
_emoji_at(button_validate.label, "\u2705", "left", scale=1.3)

button_close_ax = fig.add_axes([0.01, 0.92, 0.04, 0.06])
button_close_ax.set_zorder(6)
button_close = Button(button_close_ax, "X", color=CLOSE_FACE, hovercolor=CLOSE_HOVER)
button_close.label.set_color("#ffffff")
button_close.label.set_fontsize(16)
button_close.label.set_fontweight("bold")
button_close.on_clicked(on_close_button)

def _update_validate_button_state():
    """Set 'Valider' button grey when no human points, green otherwise."""
    if button_validate is None:
        return
    if drawn_x or (human_pred is not None):
        face = "#2ecc71"  # green
        hover = "#58d68d"
    else:
        face = "#bcbec2"  # grey
        hover = "#d0d3d8"
    try:
        button_validate.color = face
        button_validate.hovercolor = hover
        _paint_button(button_validate)
        button_validate.label.set_color(BUTTON_LABEL)
        hint_off = hint_used or human_pred is not None
        button_hint.color = "#bcbec2" if hint_off else "#ffb703"
        button_hint.hovercolor = "#d0d3d8" if hint_off else "#ffc94d"
        _paint_button(button_hint)
        fig.canvas.draw_idle()
    except Exception:
        pass

# Dataset selector radio panel (right side)
radio_ax = fig.add_axes([0.80, 0.68, 0.18, 0.17])
radio_ax.set_zorder(6)
radio_ax.set_facecolor("#ffffff")
for spine in radio_ax.spines.values():
    spine.set_edgecolor("#0f1115")
    spine.set_linewidth(1.2)
radio_ax.set_title("Choisis ton niveau", color=TEXT_DARK, fontsize=13, fontweight="bold", pad=8)
radio_labels = [ _dataset_display_name(n) for n in DATASET_OPTIONS ]
radio_datasets = RadioButtons(radio_ax, radio_labels, active=current_dataset_index)
try:
    radio_datasets.activecolor = "#4d96ff"
except Exception:
    pass
for text in radio_datasets.labels:
    text.set_color("#111")
    text.set_fontsize(13)
    text.set_fontweight("bold")
radio_datasets.on_clicked(on_radio_dataset)
for text, emo in zip(radio_datasets.labels, ["\U0001F423", "\U0001F680", "\U0001F4C8"]):
    _emoji_at(text, emo, "right", scale=1.3)

# Ranking panel (external, bottom-right). Coordinates chosen to sit outside main axes.
try:
    ranking_ax = fig.add_axes([0.03, 0.03, 0.23, 0.17])
    ranking_ax.set_zorder(6)
    ranking_ax.set_facecolor("#ffffff")
    ranking_ax.set_xticks([])
    ranking_ax.set_yticks([])
    for spine in ranking_ax.spines.values():
        spine.set_edgecolor("#d7d7d7")
        spine.set_linewidth(1.2)
except Exception:
    ranking_ax = None

# Initialize validate button state
_update_validate_button_state()

fig.canvas.draw()  # resolve axes sizes so rounded corners come out circular
for btn in (button_new_signal, button_erase, button_hint, button_validate):
    _style_button(btn)
_style_button(button_close, radius_px=10)
_rounded_bg(radio_ax, "#ffffff", edge="#d6def5", lw=1.5)
if ranking_ax is not None:
    ranking_ax.set_frame_on(False)
# Robot corner: Robo and its speech bubble, under the "Indice" button
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
if ROBOT_IMG is not None:
    fig.add_artist(AnnotationBbox(OffsetImage(ROBOT_IMG, zoom=42 / ROBOT_IMG.shape[0]), (0.615, 0.085),
                                  xycoords="figure fraction", frameon=False, zorder=7))
robot_bubble = fig.text(0.645, 0.085, "", ha="left", va="center", fontsize=12, color="#2d3047", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.8", facecolor="#ffffff",
                                  edgecolor=COLOR_MODEL, linewidth=2))

_apply_theme()
_refresh_main_axes()  # ranking panel exists now, draw it from the start
_update_validate_button_state()


# --- Welcome screen: title, avatar choice and a big "Jouer !" button ---
AVATAR_SPOTS = [(0.30 + 0.10 * i, 0.42) for i in range(len(AVATARS))]
PLAY_BOX = (0.40, 0.22, 0.20, 0.10)   # x0, y0, w, h in figure fraction (kept clear of the widgets)
welcome_active = False
welcome_artists = []
avatar_discs = []


def _select_avatar(i: int):
    global AVATAR
    AVATAR = AVATARS[i]
    for j, disc in enumerate(avatar_discs):
        disc.set_edgecolor("#4d96ff" if j == i else "#d6def5")
        disc.set_linewidth(5 if j == i else 1.5)
    fig.canvas.draw_idle()


def _show_welcome():
    global welcome_active
    from matplotlib.patches import FancyBboxPatch, Ellipse
    welcome_active = True
    fw, fh = fig.get_size_inches()

    def add(a):
        fig.add_artist(a)
        welcome_artists.append(a)
        return a

    add(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=fig.transFigure,
                       boxstyle="round,pad=0,rounding_size=0.02", mutation_aspect=fw / fh,
                       facecolor=FIG_FACE, edgecolor="#4d96ff", linewidth=4, zorder=400))
    if ROBOT_IMG is not None:
        add(AnnotationBbox(OffsetImage(ROBOT_IMG, zoom=70 / ROBOT_IMG.shape[0]), (0.5, 0.86),
                           xycoords="figure fraction", frameon=False, zorder=410))
    kw = dict(ha="center", va="center", transform=fig.transFigure, zorder=410)
    welcome_artists.append(fig.text(0.5, 0.695, "Bats le robot !", fontsize=44, fontweight="bold",
                                    color=COLOR_MODEL, **kw))
    welcome_artists.append(fig.text(0.5, 0.605, "Devine la suite des courbes mieux que Robo !",
                                    fontsize=18, color=TEXT_DARK, **kw))
    welcome_artists.append(fig.text(0.5, 0.525, "Choisis ton personnage :", fontsize=16,
                                    fontweight="bold", color=TEXT_DARK, **kw))
    avatar_discs.clear()
    for (ax_x, ax_y), emo in zip(AVATAR_SPOTS, AVATARS):
        avatar_discs.append(add(Ellipse((ax_x, ax_y), 0.075, 0.075 * fw / fh, transform=fig.transFigure,
                                        facecolor="#ffffff", edgecolor="#d6def5", linewidth=1.5, zorder=405)))
        img = _emoji_img(emo)
        if img is not None:
            add(AnnotationBbox(OffsetImage(img, zoom=48 / img.shape[0]), (ax_x, ax_y),
                               xycoords="figure fraction", frameon=False, zorder=410))
    _select_avatar(AVATARS.index(AVATAR))
    px, py, pw, ph = PLAY_BOX
    add(FancyBboxPatch((px, py), pw, ph, transform=fig.transFigure,
                       boxstyle="round,pad=0,rounding_size=0.025", mutation_aspect=fw / fh,
                       facecolor="#2ecc71", edgecolor="none", zorder=405))
    t = fig.text(px + pw / 2 + 0.012, py + ph / 2, "Jouer !", fontsize=28, fontweight="bold",
                 color="#ffffff", **kw)
    welcome_artists.append(t)
    welcome_artists.append(_emoji_at(t, "\U0001F3AE", "left", scale=1.2))
    welcome_artists.append(fig.text(0.5, 0.17, "(ou appuie sur Entrée)", fontsize=11, color=TEXT_MUTED, **kw))
    fig.canvas.draw_idle()


def _close_welcome():
    global welcome_active
    for a in welcome_artists:
        try:
            if a is not None:
                a.remove()
        except Exception:
            pass
    welcome_artists.clear()
    welcome_active = False
    _refresh_main_axes()
    _start_past_anim()
    _robot_say("hello")


def _welcome_click(event):
    fx, fy = fig.transFigure.inverted().transform((event.x, event.y))
    for i, (ax_x, ax_y) in enumerate(AVATAR_SPOTS):
        if abs(fx - ax_x) < 0.04 and abs(fy - ax_y) < 0.08:
            _select_avatar(i)
            return
    px, py, pw, ph = PLAY_BOX
    if px <= fx <= px + pw and py <= fy <= py + ph:
        _close_welcome()


_show_welcome()

# --- Event bindings ---
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("motion_notify_event", on_hover)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()