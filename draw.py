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

# --- Model setup ---
config = PatchFMConfig(compile=True)
model = Forecaster(config)

# --- Dataset management ---
DATASET_OPTIONS = ["simple.npy", "medium.npy", "dic_stocks.pkl"]
DATASET_LABELS = {
    "simple.npy": "Simple",
    "medium.npy": "Moyen",
    "dic_stocks.pkl": "Cours de bourse",
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
    base = DATASET_LABELS.get(current_dataset_name or "", os.path.splitext(os.path.basename(current_dataset_name or "?"))[0])
    return f"Données: {base}"

def _dataset_display_name(name: str) -> str:
    return DATASET_LABELS.get(name, os.path.splitext(os.path.basename(name))[0])

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
AX_FACE = "#fff7eb"
FIG_FACE = "#fce8f5"

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
        0.5, 0.95, message,
        ha="center", va="top", fontsize=12, color="#ffffff",
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


def _show_result_overlay(human_loss: float | None, ai_loss: float | None, duration_ms: int = 3500):
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
        title = "Bravo, tu as gagné !"
        color = WIN
    elif human_loss > ai_loss:
        title = "Tu as perdu."
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
                       fontsize=24, fontweight="bold",
                       color=color, transform=fig.transFigure,
                       zorder=210)
    artists.append(t_title)

    # Loss lines with winner in green, loser in red
    human_col = WIN if human_loss <= ai_loss else LOSE
    ai_col = WIN if ai_loss < human_loss else (LOSE if ai_loss > human_loss else TIE)

    t_h = fig.text(x0 + w/2, y0 + h*0.52,
                   f"Perte humaine: {human_loss:.4f}",
                   ha="center", va="center",
                   fontsize=18, fontweight="bold",
                   color=human_col, transform=fig.transFigure,
                   zorder=210)
    t_ai = fig.text(x0 + w/2, y0 + h*0.36,
                    f"Perte IA: {ai_loss:.4f}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color=ai_col, transform=fig.transFigure,
                    zorder=210)
    artists.extend([t_h, t_ai])

    # Optional hint
    t_hint = fig.text(x0 + w/2, y0 + h*0.14,
                      "La boîte disparaîtra bientôt…",
                      ha="center", va="center",
                      fontsize=11, color="#FFA500",
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

def _timeout_handler(signum, frame):
    raise TimeoutError("operation timed out")

def _update_legend(axis: plt.Axes):
    """Place a larger legend box just outside the plot with pretty styling."""
    handles, labels = axis.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique and label:
            unique[label] = handle

    if not unique:
        axis.legend_.remove() if axis.legend_ else None
        return

    axis.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="center left",
        bbox_to_anchor=(1.08, 0.5),
        borderaxespad=0.9,
        frameon=True,
        facecolor="white",
        edgecolor="#d7d7d7",
        fontsize=12,
        borderpad=1.3,
        labelspacing=1.1,
        handlelength=2.8,
        fancybox=True
    )


def _clear_drawn_points():
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

    ax.plot(x_obs, y_obs, label="Courbe connue", color=COLOR_OBS, linewidth=2)
    ax.axvline(x_obs[-1], color=COLOR_OBS, linestyle="--", linewidth=1.5, label="Stop")
    ax.axvspan(x_future[0], x_future[-1], color=COLOR_WINDOW, alpha=0.25, label="Zone à dessiner")

    # Model prediction (use normalized array)
    if model_pred is not None:
        mp = _normalize_model_pred(model_pred)
        if mp is not None and len(mp) == len(x_future):
            ax.plot(x_future, mp, color=COLOR_MODEL, linewidth=2.4, label="Prédiction modèle")

    # Human validated prediction
    if human_pred is not None and len(human_pred) == len(x_future):
        ax.plot(x_future, y_future, color=COLOR_FUTURE, linewidth=2.4, label="Vraie suite")
        ax.plot(x_future, human_pred, color=COLOR_PRED, linewidth=2.4, label="Ta prédiction")

    # Drawn points (pre/post validation)
    if drawn_x:
        ax.scatter(drawn_x, drawn_y, c=COLOR_DRAWN, edgecolors="white", linewidths=0.6, s=60, alpha=0.9, label="Tes points")

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*y_lim)
    ax.set_autoscale_on(False)
    # Title with dataset and optional stock name
    ds_base = _dataset_display_name(current_dataset_name or "?")
    stock_part = f" | Stock: {current_stock_name}" if current_stock_name else ""
    # Build title with losses if available
    title_text = (
        "Dessine dans la zone jaune la suite\n"
        f"Jeu de données: {ds_base}{stock_part}"
    )
    # Keep title only for dataset/stock; render losses below with colors
    ax.set_title(title_text, fontsize=14)

    _update_legend(ax)
    # Remove previous loss text artists if any
    try:
        prev_loss = getattr(_refresh_main_axes, "_loss_artists", [])
        for a in prev_loss:
            try:
                a.remove()
            except Exception:
                pass
    except Exception:
        pass
    loss_artists = []

    # Render colored loss text (best green, worst red) centered under the title
    try:
        human_val = None
        ai_val = None
        try:
            if human_mse is not None and np.isfinite(human_mse):
                human_val = float(human_mse)
        except Exception:
            human_val = None
        try:
            if mse_pred is not None and np.isfinite(mse_pred):
                ai_val = float(mse_pred)
        except Exception:
            ai_val = None

        if human_val is not None or ai_val is not None:
            # Decide colors: lower loss is better
            human_col = "#6c757d"
            ai_col = "#6c757d"
            if human_val is not None and ai_val is not None:
                if human_val < ai_val:
                    human_col, ai_col = "#2ecc71", "#ff6b6b"
                elif ai_val < human_val:
                    human_col, ai_col = "#ff6b6b", "#2ecc71"
                else:
                    human_col = ai_col = "#6c757d"
            else:
                if human_val is not None:
                    human_col = "#2ecc71"
                if ai_val is not None:
                    ai_col = "#2ecc71"

            # Build formatted strings
            human_str = f"Humain: {human_val:.4f}" if human_val is not None else "Humain: -"
            ai_str = f"IA: {ai_val:.4f}" if ai_val is not None else "IA: -"

            # Draw a rounded background panel to make losses stand out
            try:
                from matplotlib.patches import FancyBboxPatch
                panel = FancyBboxPatch((0.3, 0.9), 0.36, 0.06, transform=ax.transAxes,
                                       boxstyle="round,pad=0.02", facecolor="#ffffff", edgecolor="#d7d7d7",
                                       linewidth=1.2, alpha=0.98)
                ax.add_patch(panel)
                loss_artists.append(panel)
            except Exception:
                pass

            # Larger, bolder text centered inside the panel
            fontsize_big = 15
            t_h = ax.text(0.45, 0.93, human_str, ha="right", va="center", fontsize=fontsize_big, fontweight="bold",
                          color=human_col, transform=ax.transAxes)
            t_sep = ax.text(0.5, 0.93, "  |  ", ha="center", va="center", fontsize=fontsize_big, fontweight="bold",
                            color="#333", transform=ax.transAxes)
            t_ai = ax.text(0.55, 0.93, ai_str, ha="left", va="center", fontsize=fontsize_big, fontweight="bold",
                           color=ai_col, transform=ax.transAxes)
            loss_artists.extend([t_h, t_sep, t_ai])
    except Exception:
        pass

    _refresh_main_axes._loss_artists = loss_artists
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
                from matplotlib.patches import FancyBboxPatch
                panel = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ranking_ax.transAxes,
                                       boxstyle="round,pad=0.02", facecolor=PANEL_BG,
                                       edgecolor=PANEL_EDGE, linewidth=1.2)
                ranking_ax.add_patch(panel)
            except Exception:
                # fallback: rely on text bbox if FancyBboxPatch isn't available
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
            ranking_ax.text(0.5, 0.78, "Classement", ha="center", va="center",
                            fontsize=11, fontweight="bold", color="#111", transform=ranking_ax.transAxes)

            if mean_human is not None and not np.isnan(mean_human):
                ranking_ax.text(0.5, 0.56, f"Humain: {mean_human:.4f}", ha="center", va="center",
                                fontsize=10, color=human_col, transform=ranking_ax.transAxes)
            else:
                ranking_ax.text(0.5, 0.56, "Humain: -", ha="center", va="center",
                                fontsize=10, color=NEUTRAL, transform=ranking_ax.transAxes)

            if mean_ai is not None and not np.isnan(mean_ai):
                ranking_ax.text(0.5, 0.36, f"IA: {mean_ai:.4f}", ha="center", va="center",
                                fontsize=10, color=ai_col, transform=ranking_ax.transAxes)
            else:
                ranking_ax.text(0.5, 0.36, "IA: -", ha="center", va="center",
                                fontsize=10, color=NEUTRAL, transform=ranking_ax.transAxes)

            ranking_ax.text(0.5, 0.16, f"Parties: {count}", ha="center", va="center",
                            fontsize=9, color="#333", transform=ranking_ax.transAxes)

            # make sure frame looks like a panel
            for spine in ranking_ax.spines.values():
                spine.set_edgecolor(PANEL_EDGE)
                spine.set_linewidth(1.2)
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

    y_min, y_max = np.nanmin(y), np.nanmax(y)
    pad = 0.05 * (y_max - y_min) if (y_max - y_min) > 0 else 0.05
    y_lim = (y_min - pad, y_max + pad)
    model_pred = None
    mse_pred = None
    # reset human validated prediction
    global human_pred, human_mse
    human_pred = None
    human_mse = None

# --- Points drawn by the child ---
drawn_x = []
drawn_y = []

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
        # Show big overlay with outcome
        try:
            _show_result_overlay(human_mse, mse_pred)
        except Exception:
            pass
    else:
        print(f"Résultats non sauvegardés (Human loss: {human_mse:.4f}, AI loss: {mse_pred})")

    _refresh_main_axes()
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
        drawn_x.append(event.xdata)
        drawn_y.append(event.ydata)
        ax.scatter(event.xdata, event.ydata, c=COLOR_DRAWN, edgecolors="white", linewidths=0.4, s=45)
        plt.draw()
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
    ):
        drawn_x.append(event.xdata)
        drawn_y.append(event.ydata)
        ax.scatter(event.xdata, event.ydata, c=COLOR_DRAWN, edgecolors="white", linewidths=0.4, s=40)
        plt.draw()
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


def on_erase_button(event):
    """Button callback: erase current drawn points without changing signal."""
    _clear_drawn_points()
    _refresh_main_axes()
    _update_validate_button_state()
    print("Zone propre, tu peux recommencer !")


def on_validate_button(event):
    """Button callback: run validation and display prediction vs ground truth."""
    if not drawn_x:
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
        _show_popup("modèle non connecté", duration_ms=2000, face_color="#ff6f6f")
        return
    except Exception as exc:
        print(f"Oups, le modèle a eu un souci : {exc}")
        _show_popup("modèle non connecté", duration_ms=2000, face_color="#ff6f6f")
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
plt.subplots_adjust(left=0.08, right=0.78, top=0.9, bottom=0.24)
_maximize_current_figure()

_clear_drawn_points()
_refresh_main_axes()

# --- Buttons (Pygame-style palette) ---
BUTTON_FACE = "#6c5ce7"   # violet doux
BUTTON_HOVER = "#a29bfe"
BUTTON_FACE_IA = "#ff03d9"
BUTTON_HOVER_IA = "#ff6efc"
BUTTON_FACE_CHECK = "#2ecc71"  # vert
BUTTON_HOVER_CHECK = "#58d68d"
BUTTON_LABEL = "#ffffff"
CLOSE_FACE = "#ff4b5c"
CLOSE_HOVER = "#ff6f91"

button_new_ax = fig.add_axes([0.81, 0.29, 0.18, 0.09]) 
button_new_ax.set_zorder(5)
button_new_signal = Button(button_new_ax, "Trouver un nouveau signal 📈", color=BUTTON_FACE, hovercolor=BUTTON_HOVER)
button_new_signal.label.set_color(BUTTON_LABEL)
button_new_signal.label.set_fontweight("bold")
button_new_signal.on_clicked(on_new_signal_button)

button_erase_ax = fig.add_axes([0.81, 0.18, 0.18, 0.09])  
button_erase_ax.set_zorder(5)
button_erase = Button(button_erase_ax, "Effacer ma prédiction 🗑️", color=BUTTON_FACE, hovercolor=BUTTON_HOVER)
button_erase.label.set_color(BUTTON_LABEL)
button_erase.label.set_fontweight("bold")
button_erase.on_clicked(on_erase_button)

#button_model_ax = fig.add_axes([0.27, 0.06, 0.18, 0.09]) 
#button_model_ax.set_zorder(5)
#button_model = Button(button_model_ax, "Voir la prédiction de l'IA 🤖 ", color=BUTTON_FACE_IA, hovercolor=BUTTON_HOVER_IA)
#button_model.label.set_color(BUTTON_LABEL)
#button_model.label.set_fontweight("bold")
#button_model.on_clicked(on_model_button)

button_validate_ax = fig.add_axes([0.38, 0.06, 0.18, 0.09]) 
button_validate_ax.set_zorder(5)
button_validate = Button(button_validate_ax, "Valider ma prédiction ✅", color=BUTTON_FACE_CHECK, hovercolor=BUTTON_HOVER_CHECK)
button_validate.label.set_color(BUTTON_LABEL)
button_validate.label.set_fontweight("bold")
button_validate.on_clicked(on_validate_button)

button_close_ax = fig.add_axes([0.01, 0.92, 0.07, 0.07])
button_close_ax.set_zorder(6)
button_close = Button(button_close_ax, "❌", color=CLOSE_FACE, hovercolor=CLOSE_HOVER)
button_close.label.set_color("#ffffff")
button_close.label.set_fontsize(16)
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
        button_validate.ax.set_facecolor(face)
        button_validate.label.set_color(BUTTON_LABEL)
        fig.canvas.draw_idle()
    except Exception:
        pass

# Dataset selector radio panel (right side)
radio_ax = fig.add_axes([0.85, 0.75, 0.10, 0.12])
radio_ax.set_zorder(6)
radio_ax.set_facecolor("#ffffff")
for spine in radio_ax.spines.values():
    spine.set_edgecolor("#0f1115")
    spine.set_linewidth(1.2)
radio_ax.set_title("Jeu de données", color="#222", fontsize=11, pad=8)
radio_labels = [ _dataset_display_name(n) for n in DATASET_OPTIONS ]
radio_datasets = RadioButtons(radio_ax, radio_labels, active=current_dataset_index)
try:
    radio_datasets.activecolor = "#6c5ce7"
except Exception:
    pass
for text in radio_datasets.labels:
    text.set_color("#111")
    text.set_fontsize(11)
    text.set_fontweight("bold")
radio_datasets.on_clicked(on_radio_dataset)

# Ranking panel (external, bottom-right). Coordinates chosen to sit outside main axes.
try:
    ranking_ax = fig.add_axes([0.02, 0.06, 0.18, 0.09])
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

#for btn_ax in (button_new_ax, button_model_ax, button_validate_ax, button_erase_ax, button_close_ax, radio_ax):
for btn_ax in (button_new_ax, button_validate_ax, button_erase_ax, button_close_ax, radio_ax):
    for spine in btn_ax.spines.values():
        if btn_ax is button_close_ax:
            spine.set_edgecolor(CLOSE_HOVER)
        else:
            spine.set_edgecolor("#0f1115")
        spine.set_linewidth(2)

# --- Event bindings ---
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()