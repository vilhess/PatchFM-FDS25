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

plt.rcParams["font.family"] = ["Arial Rounded MT Bold", "DejaVu Sans"]

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
        if name in DATASET_OPTIONS:
            current_dataset_index = DATASET_OPTIONS.index(name)
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

MAX_CTX = 512
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
hint_artist = None

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


def _show_result_overlay(human_loss: float | None, ai_loss: float | None, duration_ms: int = 6000):
    """
    Display a big centered box announcing if the human beats the AI, in French.
    The header is green on human win, red on human loss, grey on tie.
    Also shows both losses. Auto-hides after duration_ms.
    """
    global fig

    if fig is None:
        # Fallback to console output
        try:
            if human_loss is None or ai_loss is None or not np.isfinite(human_loss) or not np.isfinite(ai_loss):
                print("Résultat indisponible.")
            else:
                if human_loss < ai_loss:
                    print(f"Tu as gagné !\nPerte humaine: {human_loss:.4f}\nPerte IA: {ai_loss:.4f}")
                elif human_loss > ai_loss:
                    print(f"Tu as perdu.\nPerte humaine: {human_loss:.4f}\nPerte IA: {ai_loss:.4f}")
                else:
                    print(f"Égalité !\nPerte humaine: {human_loss:.4f}\nPerte IA: {ai_loss:.4f}")
        except Exception:
            pass
        return

    # Clear any previous overlay
    try:
        prev_timer = getattr(_show_result_overlay, "_timer", None)
        if prev_timer is not None:
            try:
                prev_timer.stop()
            except Exception:
                pass
        for a in getattr(_show_result_overlay, "_artists", []):
            try:
                a.remove()
            except Exception:
                pass
    except Exception:
        pass

    if human_loss is None or ai_loss is None or not np.isfinite(human_loss) or not np.isfinite(ai_loss):
        return

    # Decide outcome
    WIN = "#2ecc71"
    LOSE = "#ff6b6b"
    TIE = "#6c757d"
    if human_loss < ai_loss:
        title = "Bravo, tu as battu l'IA !"
        color = WIN
    elif human_loss > ai_loss:
        title = "L'IA gagne cette fois..."
        color = LOSE
    else:
        title = "Égalité !"
        color = TIE

    # Panel geometry (figure coordinates)
    x0, y0, w, h = 0.3, 0.24, 0.4, 0.48

    artists = []
    try:
        from matplotlib.patches import FancyBboxPatch
        panel = FancyBboxPatch((x0, y0), w, h,
                               transform=fig.transFigure,
                               boxstyle="round,pad=0.015",
                               facecolor="#ffffff",
                               edgecolor=color,
                               linewidth=3.0,
                               zorder=200,
                               alpha=0.98)
        fig.add_artist(panel)
        artists.append(panel)
    except Exception:
        panel = None

    # Title and losses
    t_title = fig.text(x0 + w/2, y0 + h*0.78, title,
                       ha="center", va="center",
                       fontsize=28, fontweight="bold",
                       color=color, transform=fig.transFigure,
                       zorder=210)
    artists.append(t_title)

    # Loss lines with winner in green, loser in red
    human_col = WIN if human_loss <= ai_loss else LOSE
    ai_col = WIN if ai_loss < human_loss else (LOSE if ai_loss > human_loss else TIE)

    t_h = fig.text(x0 + w/2, y0 + h*0.52,
                   f"Toi : {_fmt_score(human_loss)}",
                   ha="center", va="center",
                   fontsize=22, fontweight="bold",
                   color=human_col, transform=fig.transFigure,
                   zorder=210)
    t_ai = fig.text(x0 + w/2, y0 + h*0.36,
                    f"IA : {_fmt_score(ai_loss)}",
                    ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color=ai_col, transform=fig.transFigure,
                    zorder=210)
    artists.extend([t_h, t_ai])

    # Optional hint
    t_hint = fig.text(x0 + w/2, y0 + h*0.15,
                      "Plus le score est petit, plus tu es proche de la vraie courbe.\nClique sur « Nouvelle courbe » pour rejouer !",
                      ha="center", va="center",
                      fontsize=13, color=TEXT_DARK,
                      transform=fig.transFigure, zorder=210)
    artists.append(t_hint)

    fig.canvas.draw_idle()

    # schedule removal
    timer = fig.canvas.new_timer(interval=int(duration_ms))

    def _remove_overlay():
        try:
            for a in artists:
                try:
                    a.remove()
                except Exception:
                    pass
            fig.canvas.draw_idle()
        except Exception:
            pass

    timer.add_callback(_remove_overlay)
    timer.start()

    _show_result_overlay._timer = timer
    _show_result_overlay._artists = artists

SCORE_SCALE = 100  # raw MSE values are tiny, show them x100 to kids


def _fmt_score(v) -> str:
    return f"{v * SCORE_SCALE:.2f}"


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
ai_anim_step = None   # None = AI curve fully shown; int = number of points revealed so far
ai_anim_timer = None
ai_line = None
robot_artist = None


_EMOJI_CACHE = {}


def _emoji_img(chars: str):
    """Render emoji to an RGBA array (matplotlib can't draw colour emoji itself). None if unavailable."""
    if chars not in _EMOJI_CACHE:
        try:
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 160)
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
    """Number of future points revealed by the hint (first quarter)."""
    return max(2, len(x_future) // 4)


def _stars_for(score_display: float) -> int:
    if score_display < 1:
        stars = 3
    elif score_display < 10:
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
        if human_mse is not None and mse_pred is not None and human_mse < mse_pred:
            _launch_confetti()
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
    ai_anim_step = 1
    _refresh_main_axes()
    ai_anim_timer = fig.canvas.new_timer(interval=AI_ANIM_INTERVAL_MS)
    ai_anim_timer.add_callback(_ai_anim_tick)
    ai_anim_timer.start()


def _clear_drawn_points():
    _stop_ai_anim()
    _stop_confetti()
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


def _refresh_main_axes():
    """Render the observed signal and prediction window on the main axes."""
    global ax
    if ax is None or x_obs is None:
        return
    ax.clear()
    ax.set_facecolor(AX_FACE)
    if ax.figure is not None:
        ax.figure.set_facecolor(FIG_FACE)

    ax.plot(x_obs, y_obs, label="Le passé", color=COLOR_OBS, linewidth=3)
    ax.axvline(x_obs[-1], color=COLOR_OBS, linestyle="--", linewidth=1.5)
    ax.axvspan(x_future[0], x_future[-1], color=COLOR_WINDOW, alpha=0.3)
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
                               label="Le robot", solid_capstyle="round")
            robot_artist = _make_robot((x_future[n - 1], mp[n - 1]))

    # Human validated prediction, with the gap to the truth shaded
    if human_pred is not None and len(human_pred) == len(x_future):
        ax.fill_between(x_future, human_pred, y_future, color=COLOR_PRED, alpha=0.18, linewidth=0)
        ax.plot(x_future, y_future, color=COLOR_FUTURE, linewidth=3, label="La vraie suite")
        ax.plot(x_future, human_pred, color=COLOR_PRED, linewidth=3, label="Ta courbe")
    elif hint_used:
        k = _hint_len()
        ax.plot(x_future[:k], y_future[:k], color=COLOR_FUTURE, linewidth=3, linestyle=(0, (2, 2)),
                label="Indice")
        img = _emoji_img("\U0001F4A1")
        if img is not None:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=22 / img.shape[0]),
                                         (x_future[k - 1], y_future[k - 1]), xybox=(0, 18),
                                         boxcoords="offset points", frameon=False, zorder=9))

    # Drawn points (pre/post validation)
    global drawn_line, hint_artist
    drawn_line, = ax.plot([], [], color=COLOR_DRAWN, linewidth=4, marker="o", markersize=5,
                          markeredgecolor="white", solid_capstyle="round",
                          label="Ta courbe" if human_pred is None else "_nolegend_")
    _update_drawn_line()
    drawn_line.set_visible(human_pred is None)
    hint_artist = None
    if not drawn_x and human_pred is None:
        hint_artist = ax.text((x_future[0] + x_future[-1]) / 2, 0.45, "Dessine\nla suite\nici !",
                              transform=ax.get_xaxis_transform(), ha="center", va="center",
                              fontsize=17, fontweight="bold", color="#e07a00", linespacing=1.4,
                              bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.8", facecolor="#ffffff",
                                        edgecolor=COLOR_WINDOW, linewidth=2.5, alpha=0.9))
        hint_emoji = _emoji_at(hint_artist, "\u270F\uFE0F", side="top", scale=1.8, pad=0.9)
        hint_artist._emoji = hint_emoji

    ax.set_xlim(x[max(0, n_obs - 2 * n_future)], x[-1])
    ax.set_ylim(*y_lim)
    ax.set_autoscale_on(False)
    # Title: instructions while playing, gentle result banner once validated
    ds_base = _dataset_display_name(current_dataset_name or "?")
    stock_part = f" | Action : {current_stock_name}" if current_stock_name else ""
    _update_legend(ax)

    human_val = float(human_mse) if human_mse is not None and np.isfinite(human_mse) else None
    ai_val = float(mse_pred) if mse_pred is not None and np.isfinite(mse_pred) else None

    if human_pred is not None and ai_anim_step is not None:
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
        _emoji_at(t, "\U0001F9D2", "left")
        ax.text(0.5, 1.045, "vs", ha="center", fontsize=14, color="#b0b7c9", **kw)
        t = ax.text(0.57, 1.045, f"Robot : {_fmt_score(ai_val) if ai_val is not None else '-'}", ha="left",
                    fontsize=16, fontweight="bold", color=ai_col, **kw)
        _emoji_at(t, "\U0001F916", "left")
        t = ax.text(0.01, 1.045, "Ton dessin :", ha="left", fontsize=12, color="#8a90a6", **kw)
        _emoji_at(t, "\u2B50" * _stars_for(human_val * SCORE_SCALE), "right", scale=1.3, pad=0.08)
        ax.text(0.99, 1.045, "zone rose = ton écart", ha="right", fontsize=11, color="#8a90a6", **kw)

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
                                color=TEXT_DARK, **rk)
            _emoji_at(t, "\U0001F3C6", "left")

            has_h = mean_human is not None and not np.isnan(mean_human)
            has_ai = mean_ai is not None and not np.isnan(mean_ai)
            t = ranking_ax.text(0.24, 0.52, _fmt_score(mean_human) if has_h else "-", ha="left", fontsize=13,
                                fontweight="bold", color=human_col if has_h else NEUTRAL, **rk)
            _emoji_at(t, "\U0001F9D2", "left", scale=1.4)
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

    y_series = y_series - np.mean(y_series)  # Center the signal
    y_series = y_series / (np.std(y_series) + 0.0005)  # Normalize the signal
    x_series = np.linspace(0, len(y_series), len(y_series))

    x = x_series
    y = y_series

    assert n_future < len(x), "n_future must be smaller than number of samples"
    n_obs = len(x) - n_future

    x_obs, y_obs = x[:n_obs], y[:n_obs]
    x_future, y_future = x[n_obs:], y[n_obs:]

    y_min, y_max = np.nanmin(y[max(0, n_obs - 2 * n_future):]), np.nanmax(y[max(0, n_obs - 2 * n_future):])
    pad = 0.05 * (y_max - y_min) if (y_max - y_min) > 0 else 0.05
    y_lim = (y_min - pad, y_max + pad)
    model_pred = None
    mse_pred = None
    # reset human validated prediction
    global human_pred, human_mse, hint_used
    human_pred = None
    human_mse = None
    hint_used = False

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
    # Only allow drawing inside the future window and with finite coordinates
    if x_future is None:
        return
    if (
        event.inaxes
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
    if x_future is None:
        return
    if (
        event.inaxes and event.button == 1
        and event.xdata is not None and np.isfinite(event.xdata)
        and event.ydata is not None and np.isfinite(event.ydata)
        and x_future[0] <= event.xdata <= x_future[-1]
        and human_pred is None
    ):
        _add_drawn_point(event.xdata, event.ydata)
    _update_validate_button_state()

def on_key(event):
    global drawn_x, drawn_y
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
    _refresh_main_axes()
    _update_validate_button_state()
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
    _show_popup("Voici le début de la vraie suite ! (une étoile en moins)", face_color="#ff9f1c")


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
        _show_popup("Dessine d'abord dans la zone jaune !", face_color="#ff9f1c")
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
        _refresh_main_axes()
        _update_validate_button_state()
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
plt.subplots_adjust(left=0.04, right=0.76, top=0.86, bottom=0.26)
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
_update_validate_button_state()

# --- Event bindings ---
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("motion_notify_event", on_hover)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()