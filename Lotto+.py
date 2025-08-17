import pandas as pd
import numpy as np
import warnings
import re
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Suppress hmmlearn migration spam from MultinomialHMM (broader match)
# Treat the migration warning as a visible warning (not an error) so newer 0.3.x can proceed under guard
warnings.filterwarnings("ignore", message=r".*MultinomialHMM has undergone.*", category=UserWarning, module=r"hmmlearn\..*")

# Extra-hard squelch for hmmlearn MultinomialHMM migration spam (handles multi-line/cross-version cases)
warnings.filterwarnings("ignore", message=r"(?s).*MultinomialHMM has undergone.*")
warnings.filterwarnings("ignore", message=r".*MultinomialHMM has undergone.*", category=Warning, module=r"hmmlearn(\.|$).*")

# Last-resort hook to drop specific MultinomialHMM migration warnings at print time
def _squelch_hmm_migration_warning():
    pat = re.compile(r"MultinomialHMM has undergone", re.IGNORECASE)
    _orig_show = warnings.showwarning
    def _filter(message, category, filename, lineno, file=None, line=None):
        try:
            if pat.search(str(message)):
                return  # drop it silently
        except Exception:
            pass
        return _orig_show(message, category, filename, lineno, file=file, line=line)
    warnings.showwarning = _filter
_squelch_hmm_migration_warning()

# --- Hard suppression of hmmlearn MultinomialHMM migration spam (stdout/stderr + logging) ---
import logging, contextlib, io

# Silence the hmmlearn logger (some versions use logging instead of warnings)
try:
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
except Exception:
    pass

class _StreamFilter(io.TextIOBase):
    def __init__(self, sink, pattern):
        self._sink = sink
        import re as _re
        self._pat = _re.compile(pattern, _re.S | _re.I)
    def write(self, s):
        try:
            if self._pat.search(str(s)):
                # Drop lines containing the migration notice entirely
                return len(s)
        except Exception:
            pass
        return self._sink.write(s)

@contextlib.contextmanager
def _squelch_streams(pattern=r"MultinomialHMM has undergone"):
    import sys as _sys
    out, err = _sys.stdout, _sys.stderr
    filt_out = _StreamFilter(out, pattern)
    filt_err = _StreamFilter(err, pattern)
    _sys.stdout, _sys.stderr = filt_out, filt_err
    try:
        yield
    finally:
        _sys.stdout, _sys.stderr = out, err

# --- Environment/version logging and run log helpers ---
import json
import os
import sys
import platform
import subprocess
from datetime import datetime, timezone

def _now_iso():
    # Use timezone‑aware UTC to avoid DeprecationWarning in Python 3.12+
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _freeze_requirements(output_path="requirements_frozen.txt"):
    """
    Snapshot current Python packages to a pinned requirements file using `python -m pip freeze`.
    This runs once per script invocation to 'version the environment' alongside the model run.
    """
    try:
        # Avoid repeatedly writing in the same second if re-imported
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = output_path
        # If a file already exists with the same name, keep overwriting (acts like a lock-file for the run)
        cmd = [sys.executable, "-m", "pip", "freeze"]
        txt = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60)
        with open(out, "w", encoding="utf-8") as f:
            f.write("# Frozen by Lotto+ on " + stamp + "\n")
            f.write(txt)
    except Exception as e:
        # Non-fatal
        warnings.warn(f"pip freeze failed: {e}")

def _log_jsonl(record, path="run_log.jsonl"):
    """
    Append a single JSON record to a newline-delimited log file.
    """
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        warnings.warn(f"run_log write failed: {e}")

def _log_env_versions():
    """
    Log core library versions and runtime info to stdout and JSONL for reproducibility.
    """
    info = {
        "type": "env",
        "ts": _now_iso(),
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "numpy": getattr(__import__("numpy"), "__version__", None),
            "pandas": getattr(__import__("pandas"), "__version__", None),
            "scikit_learn": getattr(__import__("sklearn"), "__version__", None),
            "scipy": getattr(__import__("scipy"), "__version__", None),
            "tensorflow": getattr(__import__("tensorflow"), "__version__", None) if "tensorflow" in sys.modules else None,
            "keras": getattr(__import__("keras"), "__version__", None) if "keras" in sys.modules else None,
            "xgboost": getattr(__import__("xgboost"), "__version__", None) if "xgboost" in sys.modules else None,
            "hmmlearn": None,  # filled below if available
        },
    }
    # Use global HMM_VERSION if already computed; otherwise try import lazily
    try:
        info["versions"]["hmmlearn"] = globals().get("HMM_VERSION") or getattr(__import__("hmmlearn"), "__version__", None)
    except Exception:
        info["versions"]["hmmlearn"] = None

    print("[ENV]", json.dumps(info, ensure_ascii=False))
    _log_jsonl(info)

    # Also snapshot pip versions to a frozen requirements file for this run
    _freeze_requirements("requirements_frozen.txt")

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _summarise_calibrator():
    """
    Return a small dict summarising the postrank calibrator (if present).
    Captures alpha/K and an L2 norm of W to avoid dumping huge matrices.
    """
    cal = None
    try:
        cal_obj = globals().get("postrank_calibrator", None)
        if cal_obj is not None:
            cal = {
                "type": type(cal_obj).__name__,
                "alpha": _safe_float(getattr(cal_obj, "alpha", None)),
                "K": int(getattr(cal_obj, "K", 0)) if getattr(cal_obj, "K", None) is not None else None,
            }
            # Optional norms if attributes present
            W = getattr(cal_obj, "W", None)
            if W is not None:
                try:
                    import numpy as _np
                    cal["W_l2"] = _safe_float(_np.linalg.norm(_np.asarray(W)))
                except Exception:
                    pass
    except Exception:
        cal = None

    # Also log beta shaping if in globals
    try:
        if "best_beta" in globals():
            if cal is None:
                cal = {}
            cal["best_beta"] = _safe_float(globals().get("best_beta"))
    except Exception:
        pass
    return cal

def _per_expert_names():
    return ["Bayes", "Markov", "HMM", "TCNN", "Transformer"]

def _log_predict_run(blend_weights, t_eval=None, log_path="run_log.jsonl"):
    """
    At predict time, log:
      - blend weights (final weights used for bases)
      - calibration summary (alpha/K/beta)
      - per-expert PL-NLL on the most recent evaluated draw (default t=n-1)
      - weekday gate status/weight (if available)
      - hmmlearn version
    """
    try:
        if t_eval is None:
            t_eval = (globals().get("n_draws", 0) - 1)
        # Per-expert PL-NLLs on last completed draw
        nlls = None
        try:
            _pe = globals().get("_per_expert_pl_nll_at")
            if callable(_pe) and t_eval is not None and t_eval >= 1:
                vals = _pe(int(t_eval))
                if isinstance(vals, (list, tuple)):
                    nlls = {name: _safe_float(v) for name, v in zip(_per_expert_names(), vals)}
        except Exception as _e:
            warnings.warn(f"Per-expert PL-NLL logging failed: {_e}")

        # Weekday gate probe (do not change predictions; just report)
        wk_gate = None
        try:
            _gate = globals().get("_weekday_gate_for_current_run")
            if callable(_gate):
                _en, _wd_dict, _w = _gate()
                wk_gate = {"enabled": bool(_en), "weight": _safe_float(_w)}
        except Exception:
            wk_gate = None

        rec = {
            "type": "predict_log",
            "ts": _now_iso(),
            "hmmlearn": globals().get("HMM_VERSION", None),
            "blend_weights": [ _safe_float(x) for x in (list(blend_weights) if hasattr(blend_weights, "__iter__") else [blend_weights]) ],
            "calibration": _summarise_calibrator(),
            "per_expert_pl_nll": nlls,
            "weekday_gate": wk_gate,
        }
        print("[RUN]", json.dumps(rec, ensure_ascii=False))
        _log_jsonl(rec, path=log_path)
    except Exception as e:
        warnings.warn(f"predict-time logging failed: {e}")

 # --- HMM version gate and uniform helper ---
 # -- One-time environment/version log and requirements snapshot --
try:
    _log_env_versions()
except Exception as _e:
    warnings.warn(f"Env version logging failed: {_e}")
try:
    import hmmlearn as _hmm_mod
    HMM_VERSION = getattr(_hmm_mod, "__version__", "0")
    # Relaxed gate: allow 0.2.8, 0.3.2, 0.3.3 explicitly; warn but allow newer 0.3.x
    _OK_SET = {"0.2.8", "0.3.2", "0.3.3"}
    if HMM_VERSION in _OK_SET:
        HMM_OK = True
    else:
        # If we are on another 0.3.x, tentatively allow but emit a warning so we can monitor
        try:
            _major, _minor, _patch = (int(x) for x in HMM_VERSION.split(".")[:3])
            HMM_OK = (_major == 0 and _minor == 3 and _patch >= 2) or (_major == 0 and _minor == 2 and _patch >= 8)
        except Exception:
            HMM_OK = False
        if HMM_OK:
            warnings.warn(f"hmmlearn {HMM_VERSION} not explicitly validated; proceeding with compatibility path.")
except Exception:
    HMM_VERSION = "unknown"
    HMM_OK = False

def _uniform_prob40():
    return {n: 1.0/40 for n in range(1, 41)}

# --- Strict time-guard to prevent look-ahead leakage ---
CURRENT_TARGET_IDX = None  # when predicting/evaluating draw t, only indices < t may be touched

def _set_target_idx(t_idx):
    """Declare the current target draw index t for leakage guards (allowing access to draws[:t])."""
    global CURRENT_TARGET_IDX
    CURRENT_TARGET_IDX = int(t_idx) if t_idx is not None else None

def _assert_idx_ok(idx, context=""):
    """Assert that any index used for feature building is strictly less than CURRENT_TARGET_IDX."""
    if CURRENT_TARGET_IDX is None:
        return
    if int(idx) >= int(CURRENT_TARGET_IDX):
        raise RuntimeError(f"Leakage guard tripped: attempted to access index {idx} >= target {CURRENT_TARGET_IDX} {('in ' + context) if context else ''}.")

# --- Strict time-guard to prevent look-ahead leakage ---
CURRENT_TARGET_IDX = None  # when predicting/evaluating draw t, only indices < t may be touched

def _set_target_idx(t_idx):
    """Declare the current target draw index t for leakage guards (allowing access to draws[:t])."""
    global CURRENT_TARGET_IDX
    CURRENT_TARGET_IDX = int(t_idx) if t_idx is not None else None

def _assert_idx_ok(idx, context=""):
    """Assert that any index used for feature building is strictly less than CURRENT_TARGET_IDX."""
    if CURRENT_TARGET_IDX is None:
        return
    if int(idx) >= int(CURRENT_TARGET_IDX):
        raise RuntimeError(f"Leakage guard tripped: attempted to access index {idx} >= target {CURRENT_TARGET_IDX} {('in ' + context) if context else ''}.")

def _assert_chronological_frame(df):
    """Lightweight check used before each train/val split to ensure time order is preserved."""
    if 'DrawDate' not in df.columns:
        raise ValueError("Expected DrawDate column before splitting.")
    if not df['DrawDate'].is_monotonic_increasing:
        raise ValueError("DataFrame is not strictly chronological before split; aborting to avoid leakage.")

# Default MC passes used by stacker/diagnostics
MC_STACK_PASSES = 11

# 1. **Data Loading and Preprocessing**
# Load the historical Colorado Lotto+ data from the Excel file
data = pd.read_excel("Lotto+.xlsx")

# Sort data by draw date in chronological order (oldest first) for time-series modeling.
# The "Draw date" column is a string like "Monday, 8/4/25". Lock to exact format to prevent silent mis-parses.
# Example format: Monday, 8/4/25  =>  "%A, %m/%d/%y"
try:
    data['DrawDate'] = pd.to_datetime(data['Draw date'], format="%A, %m/%d/%y", errors='raise')
except Exception as e:
    raise ValueError(f"Strict date parsing failed for some rows. Ensure the sheet matches '%A, %m/%d/%y'. Original error: {e}")

# Enforce strict chronological order (oldest first) and verify monotonicity
data.sort_values('DrawDate', inplace=True)
if data['DrawDate'].isna().any():
    bad_rows = data[data['DrawDate'].isna()]
    raise ValueError(f"Found NaT dates after strict parsing. Offending rows: {bad_rows[['Draw date']].head(5).to_dict(orient='records')}")
if not data['DrawDate'].is_monotonic_increasing:
    data.sort_values('DrawDate', inplace=True)
    if not data['DrawDate'].is_monotonic_increasing:
        raise ValueError("DrawDate is not strictly non-decreasing after sort. Check for duplicate or out-of-order dates.")

# Extract the winning numbers for each draw as a list of integers.
draws = []  # list of sets of winning numbers
for nums in data['Winning Numbers']:
    # The numbers are in a string like "7 - 17 - 23 - 29 - 34 - 36"
    # Split by '-' and convert to integers.
    parts = str(nums).split('-')
    draw_numbers = {int(p.strip()) for p in parts if p.strip().isdigit()}
    if len(draw_numbers) == 6:
        draws.append(draw_numbers)
    else:
        # If parsing fails (unexpected format), skip or handle accordingly.
        continue

# Ensure we have the draws as a time-ordered list of sets.
n_draws = len(draws)
print(f"Loaded {n_draws} historical draws.")

# 2. **Statistical Analysis for Model Features**
# Compute basic frequency of each number over all draws (for Bayesian prior/posterior).
from collections import Counter
all_numbers = [num for draw in draws for num in draw]
freq_counter = Counter(all_numbers)
total_numbers_drawn = n_draws * 6  # total count of numbers drawn (6 per draw)

# Bayesian update: assume a uniform prior for probability of each number.
# Prior alpha for each number (Dirichlet prior). Using alpha=1 (uniform).
alpha = 1  
# Posterior for each number = alpha + count.
posterior_counts = {num: alpha + freq_counter.get(num, 0) for num in range(1, 41)}
# Posterior probabilities for each number (i.e., estimated probability number will be drawn in any given draw).
posterior_probs = {num: posterior_counts[num] / (total_numbers_drawn + 40 * alpha) for num in range(1, 41)}

# ===========================
# Markov Chain: Multi-step Transition Probabilities
# -------------------------------------------------
# Compute Markov transition probabilities: P(num_j in draw_t | num_i in draw_{t-1}),
# and extend to multi-step transitions: P(num_k in draw_{t+2} | num_i in draw_t), etc.
# Also, implement Hidden Markov Model (HMM) to identify latent patterns.

# --- 1-step Markov (already present, but moved here for clarity) ---
transition_counts = {i: Counter() for i in range(1, 41)}
transition_totals = {i: 0 for i in range(1, 41)}
for idx in range(n_draws - 1):
    current_draw = draws[idx]
    next_draw = draws[idx + 1]
    for num in current_draw:
        transition_totals[num] += 1
        for next_num in next_draw:
            transition_counts[num][next_num] += 1

transition_probs = {i: {} for i in range(1, 41)}
for i in range(1, 41):
    if transition_totals[i] > 0:
        for j in range(1, 41):
            count_ij = transition_counts[i].get(j, 0)
            transition_probs[i][j] = (count_ij + 1e-3) / (transition_totals[i] + 1e-3 * 40)
    else:
        transition_probs[i] = {j: 1.0/40 for j in range(1, 41)}

# --- 2-step Markov: P(num_k in draw_{t+2} | num_i in draw_t) ---
# For each (i, k), count how often k appears in draw_{t+2} if i appeared in draw_t.
transition2_counts = {i: Counter() for i in range(1, 41)}
transition2_totals = {i: 0 for i in range(1, 41)}
for idx in range(n_draws - 2):
    draw_t = draws[idx]
    draw_tp2 = draws[idx + 2]
    for num in draw_t:
        transition2_totals[num] += 1
        for num2 in draw_tp2:
            transition2_counts[num][num2] += 1

transition2_probs = {i: {} for i in range(1, 41)}
for i in range(1, 41):
    if transition2_totals[i] > 0:
        for k in range(1, 41):
            count_ik = transition2_counts[i].get(k, 0)
            transition2_probs[i][k] = (count_ik + 1e-3) / (transition2_totals[i] + 1e-3 * 40)
    else:
        transition2_probs[i] = {k: 1.0/40 for k in range(1, 41)}

# --- 3-step Markov: P(num_m in draw_{t+3} | num_i in draw_t) ---
transition3_counts = {i: Counter() for i in range(1, 41)}
transition3_totals = {i: 0 for i in range(1, 41)}
for idx in range(n_draws - 3):
    draw_t = draws[idx]
    draw_tp3 = draws[idx + 3]
    for num in draw_t:
        transition3_totals[num] += 1
        for num3 in draw_tp3:
            transition3_counts[num][num3] += 1

transition3_probs = {i: {} for i in range(1, 41)}
for i in range(1, 41):
    if transition3_totals[i] > 0:
        for m in range(1, 41):
            count_im = transition3_counts[i].get(m, 0)
            transition3_probs[i][m] = (count_im + 1e-3) / (transition3_totals[i] + 1e-3 * 40)
    else:
        transition3_probs[i] = {m: 1.0/40 for m in range(1, 41)}

# --- Hidden Markov Model (HMM) using hmmlearn ---
# The HMM attempts to model hidden "states" underlying the sequence of draws.
# We'll use the MultiLabelBinarizer to encode draws, and fit a MultinomialHMM on the sequence.
from hmmlearn import hmm
# Compatibility helper for posterior extraction across hmmlearn versions
def _hmm_posteriors(model, X_int):
    """Return state posteriors for each row in X_int (shape: [T, n_features]).
    Tries score_samples (preferred); falls back to model.predict_proba if present.
    """
    try:
        # Preferred, stable across 0.3.2/0.3.3
        _logp, post = model.score_samples(X_int)
        return post
    except Exception:
        if hasattr(model, "predict_proba"):
            try:
                return model.predict_proba(X_int)
            except Exception:
                pass
        raise
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

# Prepare draw observations as a binary matrix: each row is a draw, 1 if number present, else 0.
mlb = MultiLabelBinarizer(classes=list(range(1, 41)))
draws_bin = mlb.fit_transform([sorted(list(draw)) for draw in draws])

# Use a small number of hidden states (e.g., 3 or 4) to avoid overfitting.
n_states = 3
# Switch to MultinomialHMM on 0/1 count vectors (each row sums to 6). Use a shorter n_iter and looser tol to avoid noisy non‑convergence logs.
if HMM_OK:
    try:
        hmm_model = hmm.MultinomialHMM(n_components=n_states, n_iter=50, tol=1e-2, random_state=42, verbose=False)
        # Ensure integer counts for MultinomialHMM; suppress migration spam during fit and posterior calls
        with _squelch_streams(r"MultinomialHMM has undergone"):
            hmm_model.fit(draws_bin.astype(int))
            # Posteriors over hidden states; use helper for broad version compatibility
            post = _hmm_posteriors(hmm_model, draws_bin.astype(int))  # shape: (n_draws, n_states)
        last_state = int(np.argmax(post[-1]))
        # Emission probabilities per number for the inferred last state
        emiss = hmm_model.emissionprob_  # shape: (n_states, 40)
        hmm_prob = {num: float(max(emiss[last_state, num - 1], 1e-12)) for num in range(1, 41)}
        _s = sum(hmm_prob.values())
        hmm_prob = ({n: hmm_prob[n] / _s for n in range(1, 41)} if _s > 0 else _uniform_prob40())
    except Exception:
        hmm_prob = _uniform_prob40()
else:
    hmm_prob = _uniform_prob40()

# ===========================

# ===========================
# Enhanced Feature Engineering
from itertools import combinations
import statistics

# 1. Frequency counts of number pairs and triplets across all historical draws.
pair_counter = Counter()
triplet_counter = Counter()
for draw in draws:
    sorted_draw = sorted(draw)
    # Count pairs
    for pair in combinations(sorted_draw, 2):
        pair_counter[pair] += 1
    # Count triplets
    for triplet in combinations(sorted_draw, 3):
        triplet_counter[triplet] += 1

# 2. Identification and encoding of historical clusters or repeating subsets of numbers.
# Define clusters as pairs or triplets appearing more than a threshold times (e.g., 3 times)
cluster_threshold = 3
frequent_pairs = {pair for pair, count in pair_counter.items() if count >= cluster_threshold}
frequent_triplets = {triplet for triplet, count in triplet_counter.items() if count >= cluster_threshold}

# 3. Analysis of arithmetic sequences and intervals within draws.
# For each draw, find if there are arithmetic sequences of length >=3 and record intervals.
def find_arithmetic_sequences(nums):
    nums = sorted(nums)
    sequences = []
    n = len(nums)
    for length in range(3, n+1):
        for start in range(n - length + 1):
            seq = nums[start:start+length]
            diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
            if len(set(diffs)) == 1:  # all differences equal -> arithmetic sequence
                sequences.append((seq, diffs[0]))
    return sequences

arithmetic_sequences_per_draw = [find_arithmetic_sequences(draw) for draw in draws]

# 4. Calculation of statistical metrics (mean, median, variance) of each draw.
draw_stats = []
for draw in draws:
    draw_list = sorted(draw)
    mean_val = statistics.mean(draw_list)
    median_val = statistics.median(draw_list)
    variance_val = statistics.variance(draw_list) if len(draw_list) > 1 else 0.0
    draw_stats.append({'mean': mean_val, 'median': median_val, 'variance': variance_val})

# 5. Inclusion of long-term gap dynamics, tracking how many draws have passed since each number last appeared.
# Initialize last seen dictionary with -1 (never seen)
last_seen_full = {num: -1 for num in range(1, 41)}
# For each draw, record gap for each number
gap_history = []
for idx, draw in enumerate(draws):
    gap_for_draw = {}
    for num in range(1, 41):
        if last_seen_full[num] == -1:
            gap_for_draw[num] = idx  # gap since start if never appeared before
        else:
            gap_for_draw[num] = idx - last_seen_full[num]
    gap_history.append(gap_for_draw)
    # Update last seen for numbers in current draw
    for num in draw:
        last_seen_full[num] = idx

# ===========================

# ===========================
# Additional Feature Engineering Enhancements
# ------------------------------------------
# 1. Rolling window features:
#    - Calculate moving averages for frequency and recent gap lengths within defined windows (e.g., 5, 10, 20 draws).
# 2. Lagged occurrence patterns:
#    - Explicitly encode binary features indicating if a number appeared exactly 1, 2, or 3 draws ago.
# 3. Seasonal or calendar-based features:
#    - Extract weekday and month from the draw date and encode them as categorical or numerical features.

# --- 1. Rolling Window Features ---
# For each draw, for each number, store moving averages of frequency and recent gap length over windows.
rolling_windows = [5, 10, 20]
rolling_freq_features = {w: [] for w in rolling_windows}  # w -> list of dicts (per draw, per number)
rolling_gap_features = {w: [] for w in rolling_windows}
for idx in range(n_draws):
    for w in rolling_windows:
        left = max(0, idx - w + 1)
        window_draws = draws[left:idx+1]
        all_nums_window = [num for d in window_draws for num in d]
        freq_counter_window = Counter(all_nums_window)
        # Frequency moving average: freq in window / (window size * 6)
        freq_ma = {num: freq_counter_window.get(num, 0) / (len(window_draws) * 6) for num in range(1, 41)}
        # Recent gap: how many draws since last appearance in window (or w if not present)
        gap_ma = {}
        for num in range(1, 41):
            last_seen = -1
            for j in range(idx, left-1, -1):
                if num in draws[j]:
                    last_seen = j
                    break
            if last_seen == -1:
                gap_ma[num] = len(window_draws)
            else:
                gap_ma[num] = idx - last_seen
        rolling_freq_features[w].append(freq_ma)
        rolling_gap_features[w].append(gap_ma)

# --- 2. Lagged Occurrence Patterns ---
# For each draw, for each number, encode binary features: appeared exactly 1, 2, or 3 draws ago.
lagged_occurrence = []
for idx in range(n_draws):
    lag_feats = {}
    for num in range(1, 41):
        lag1 = 0
        lag2 = 0
        lag3 = 0
        if idx >= 1:
            lag1 = 1 if num in draws[idx-1] else 0
        if idx >= 2:
            lag2 = 1 if num in draws[idx-2] else 0
        if idx >= 3:
            lag3 = 1 if num in draws[idx-3] else 0
        lag_feats[num] = {'lag1': lag1, 'lag2': lag2, 'lag3': lag3}
    lagged_occurrence.append(lag_feats)

# --- 3. Seasonal/Calendar-based Features ---
# For each draw, extract weekday and month from DrawDate.
seasonal_features = []
for idx in range(n_draws):
    draw_date = data.iloc[idx]['DrawDate']
    # Weekday: 0=Monday, ..., 6=Sunday
    weekday = draw_date.weekday()
    # Month: 1=January, ..., 12=December
    month = draw_date.month
    seasonal_features.append({'weekday': weekday, 'month': month})

# ===========================


# Helper functions for data-leakage-free probability computation
def compute_bayes_posterior(draw_list, alpha=1):
    """
    Compute Bayesian posterior probability for each number 1‑40
    using only the draws supplied in draw_list.
    """
    from collections import Counter
    freq_counter = Counter([num for d in draw_list for num in d])
    total_numbers = len(draw_list) * 6
    return {
        n: (alpha + freq_counter.get(n, 0)) / (total_numbers + 40 * alpha)
        for n in range(1, 41)
    }

def compute_markov_transitions(draw_list):
    """
    Return 1‑, 2‑, and 3‑step transition‑probability dicts computed
    only from draw_list, using the same Laplace smoothing as in the
    global computation.
    """
    from collections import Counter
    n_local = len(draw_list)

    # 1‑step
    t1_counts = {i: Counter() for i in range(1, 41)}
    t1_totals = {i: 0 for i in range(1, 41)}
    for i in range(n_local - 1):
        for num in draw_list[i]:
            t1_totals[num] += 1
            for nxt in draw_list[i + 1]:
                t1_counts[num][nxt] += 1
    t1 = {
        i: {j: (t1_counts[i].get(j, 0) + 1e-3) /
                 (t1_totals[i] + 1e-3 * 40) for j in range(1, 41)}
        if t1_totals[i] > 0 else {j: 1/40 for j in range(1, 41)}
        for i in range(1, 41)
    }

    # 2‑step
    t2_counts = {i: Counter() for i in range(1, 41)}
    t2_totals = {i: 0 for i in range(1, 41)}
    for i in range(n_local - 2):
        for num in draw_list[i]:
            t2_totals[num] += 1
            for nxt in draw_list[i + 2]:
                t2_counts[num][nxt] += 1
    t2 = {
        i: {j: (t2_counts[i].get(j, 0) + 1e-3) /
                 (t2_totals[i] + 1e-3 * 40) for j in range(1, 41)}
        if t2_totals[i] > 0 else {j: 1/40 for j in range(1, 41)}
        for i in range(1, 41)
    }

    # 3‑step
    t3_counts = {i: Counter() for i in range(1, 41)}
    t3_totals = {i: 0 for i in range(1, 41)}
    for i in range(n_local - 3):
        for num in draw_list[i]:
            t3_totals[num] += 1
            for nxt in draw_list[i + 3]:
                t3_counts[num][nxt] += 1
    t3 = {
        i: {j: (t3_counts[i].get(j, 0) + 1e-3) /
                 (t3_totals[i] + 1e-3 * 40) for j in range(1, 41)}
        if t3_totals[i] > 0 else {j: 1/40 for j in range(1, 41)}
        for i in range(1, 41)
    }
    return t1, t2, t3

# --- Co-occurrence and compatibility helpers ---------------------------------

def _cooccurrence_matrix_from_history(sub_draws):
    """Return symmetric 40x40 co-occurrence count matrix from draws in `sub_draws`.
    cooc[i-1, j-1] counts how often i and j appeared together in the same draw.
    """
    cooc = np.zeros((40, 40), dtype=np.int32)
    for d in sub_draws:
        s = sorted(list(d))
        for a in s:
            for b in s:
                if a != b:
                    cooc[a-1, b-1] += 1
    return cooc


def _cooc_conditional_prob_from_history(sub_draws, last_draw):
    """Compute P(j | last draw has any of S) using within-draw co-occurrence counts.
    Approximates by summing co-occurrence counts from each i in `last_draw` to candidate j.
    Returns a normalised dict over 1..40.
    """
    if len(sub_draws) == 0 or not last_draw:
        return {n: 1.0/40 for n in range(1, 41)}
    cooc = _cooccurrence_matrix_from_history(sub_draws)
    score = np.zeros(40, dtype=float)
    for i in last_draw:
        score += cooc[i-1]
    # zero self-cooccurrence preference for numbers already in last_draw
    for i in last_draw:
        score[i-1] = max(0.0, score[i-1])
    if score.sum() <= 0:
        return {n: 1.0/40 for n in range(1, 41)}
    score = score / (score.sum() + 1e-12)
    return {n: float(score[n-1]) for n in range(1, 41)}


def _compat_topk_sum(prob_dict, k=5):
    """For each candidate n, compute sum of the top-k competitor probabilities
    among the *other* numbers (exclude n). Teaches without-replacement compatibility.
    Returns dict n->scalar.
    """
    p = np.array([prob_dict.get(n, 0.0) for n in range(1, 41)], dtype=float)
    out = {}
    for n in range(1, 41):
        others = np.delete(p, n-1)
        # top-k largest among others
        topk = np.partition(others, -k)[-k:]
        out[n] = float(np.sum(topk))
    return out

# 3. **Deep Learning Models: LSTM and Transformer for Sequence Prediction**
# -------------------------------------------------------------------------
# We replace the previous MLP with two advanced models using TensorFlow/Keras:
#   (1) LSTM-based Recurrent Neural Network (RNN) for sequence learning.
#   (2) Transformer-based model for effective sequence modeling.
# Both models are trained with EarlyStopping for convergence and overfitting prevention.

# --- 3.1. Prepare Data for Deep Learning ---
# Enhanced: Use derived statistical features instead of raw binary vectors.
# Each input sample: (window of k draws) -> output: next draw (multi-label, 40 binary outputs)
# Features include: moving averages of frequencies, median gaps, recent gap lengths, frequencies over sliding windows.
k = 8  # number of past draws to use as features
X_seq = []
y_seq = []

def compute_stat_features(draw_window, idx_offset):
    """
    Compute derived statistical features for each number 1-40 for a window of draws.
    Features per number:
      - Moving average of frequency over window.
      - Median gap up to the window.
      - Recent gap length (how many draws since last appearance).
      - Frequency over sliding window of last N draws (N=10).
      - Rolling window features: moving averages of frequency & gap over 5, 10, 20 draws.
      - Lagged occurrence patterns: binary features for 1, 2, 3 draws ago.
      - Seasonal/calendar-based features: weekday and month as numerical/categorical features.
    Returns: feature matrix of shape (40, n_features)
    """
    # Guard: idx_offset must be < CURRENT_TARGET_IDX (no peeking at the target draw)
    _assert_idx_ok(idx_offset, context="compute_stat_features(idx_offset)")
    n_numbers = 40
    features = []
    # Flatten draws in window for frequency
    all_nums_window = [num for d in draw_window for num in d]
    freq_counter_window = Counter(all_nums_window)
    # For moving average, use frequency in window divided by window length
    moving_avg = [freq_counter_window.get(num, 0) / (len(draw_window) * 6) for num in range(1, 41)]

    # Median gap up to this window (from gap_history)
    # idx_offset is the index of the last draw in the window
    median_gap = []
    for num in range(1, 41):
        gaps = []
        last_seen = -1
        for i in range(idx_offset - k + 1, idx_offset + 1):
            if num in draws[i]:
                gaps.append(0)
                last_seen = i
            else:
                if last_seen == -1:
                    gaps.append(i - (idx_offset - k + 1))
                else:
                    gaps.append(i - last_seen)
        median_gap.append(float(np.median(gaps)))

    # Recent gap length (draws since last appearance up to idx_offset)
    recent_gap = []
    for num in range(1, 41):
        last_seen = -1
        for i in range(idx_offset, idx_offset - k, -1):
            if num in draws[i]:
                last_seen = i
                break
        if last_seen == -1:
            recent_gap.append(k)
        else:
            recent_gap.append(idx_offset - last_seen)

    # Frequency over sliding window of last N draws (N=10)
    N = 10
    freq_window = []
    left = max(0, idx_offset - N + 1)
    all_nums_N = [num for d in draws[left:idx_offset + 1] for num in d]
    freq_counter_N = Counter(all_nums_N)
    for num in range(1, 41):
        freq_window.append(freq_counter_N.get(num, 0) / (min(N, idx_offset + 1) * 6))

    # --- Additional Features ---
    # Rolling window features (moving averages of frequency and gap for 5, 10, 20 draws)
    rolling_freqs = []
    rolling_gaps = []
    for w in [5, 10, 20]:
        # Use precomputed rolling features if available, otherwise fallback to zeros
        if idx_offset < len(rolling_freq_features[w]):
            _assert_idx_ok(idx_offset, context=f"rolling_features(w={w})")
            freq_ma = rolling_freq_features[w][idx_offset]
            gap_ma = rolling_gap_features[w][idx_offset]
        else:
            freq_ma = {num: 0.0 for num in range(1, 41)}
            gap_ma = {num: 0.0 for num in range(1, 41)}
        rolling_freqs.append([freq_ma[num] for num in range(1, 41)])
        rolling_gaps.append([gap_ma[num] for num in range(1, 41)])
    # For each number, rolling_freqs and rolling_gaps are lists of 3 values (for 5, 10, 20)

    # Lagged occurrence patterns: binary features for 1, 2, 3 draws ago
    lag_feats = []
    if idx_offset < len(lagged_occurrence):
        lag_dict = lagged_occurrence[idx_offset]
        for num in range(1, 41):
            lags = lag_dict[num]
            lag_feats.append([lags['lag1'], lags['lag2'], lags['lag3']])
    else:
        lag_feats = [[0, 0, 0] for _ in range(1, 41)]

    # Seasonal/calendar-based features: weekday and month (same for all numbers in the draw)
    if idx_offset < len(seasonal_features):
        weekday = seasonal_features[idx_offset]['weekday']
        month = seasonal_features[idx_offset]['month']
    else:
        weekday = 0
        month = 1
    # One-hot encode weekday (0-6) and month (1-12)
    weekday_onehot = [1 if weekday == i else 0 for i in range(7)]
    month_onehot = [1 if month == i+1 else 0 for i in range(12)]

    # Concatenate features for each number
    for i in range(n_numbers):
        # Combine all features into a single feature vector per number
        feats = [
            moving_avg[i],       # moving average in window
            median_gap[i],       # median gap in window
            recent_gap[i],       # recent gap
            freq_window[i],      # freq in last N draws
            rolling_freqs[0][i], # rolling freq (5)
            rolling_freqs[1][i], # rolling freq (10)
            rolling_freqs[2][i], # rolling freq (20)
            rolling_gaps[0][i],  # rolling gap (5)
            rolling_gaps[1][i],  # rolling gap (10)
            rolling_gaps[2][i],  # rolling gap (20)
            lag_feats[i][0],     # lag1
            lag_feats[i][1],     # lag2
            lag_feats[i][2],     # lag3
        ]
        # Add one-hot weekday and month (same for all numbers in the draw)
        feats = feats + weekday_onehot + month_onehot
        features.append(feats)
    return np.array(features)  # shape (40, n_features)

# Build X_seq and y_seq using derived features (with strict time guards)
_assert_chronological_frame(data)
for idx in range(k, n_draws - 1):
    # When constructing features for label at draw idx, only allow access to draws[:idx]
    _set_target_idx(idx)
    window_draws = draws[idx - k: idx]       # last k draws (each is a set of 6 numbers)
    next_draw = draws[idx]                   # the draw immediately after the window
    # Compute features for all 40 numbers
    features = compute_stat_features(window_draws, idx - 1)  # (40, n_features)
    X_seq.append(features)
    # Output: 40-dim binary vector for next draw
    y_seq.append([1 if num in next_draw else 0 for num in range(1, 41)])
X_seq = np.array(X_seq)  # shape (samples, 40, n_features)
y_seq = np.array(y_seq)  # shape (samples, 40)
# Clear guard now that the bulk feature construction is complete
_set_target_idx(None)

# --- Set-aware learning-to-rank losses (PL/BT) -------------------------------------------
import tensorflow as tf
import keras  # Keras 3.x API
from keras import layers

def pl_set_loss_factory(R=8, tau=0.20):
    """
    Plackett–Luce pseudo-likelihood loss for unordered 6-sets.
    y_true: [batch, 40] multi-hot (six 1s)
    y_pred: [batch, 40] raw logits per number
    R: number of random permutations of the positive set to average over
    tau: magnitude of Gumbel noise for Perturb-and-MAP (0 disables)
    """
    neg_inf = tf.constant(-1e9, dtype=tf.float32)

    @tf.function(experimental_relax_shapes=True)
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if tau and tau > 0:
            # Gumbel noise for differentiable Top-k
            u = tf.clip_by_value(tf.random.uniform(tf.shape(y_pred), 1e-7, 1.0 - 1e-7), 1e-7, 1.0 - 1e-7)
            g = -tf.math.log(-tf.math.log(u)) * tf.cast(tau, y_pred.dtype)
            logits = y_pred + g
        else:
            logits = y_pred

        def _sample_pl_nll(args):
            y, logit = args
            pos_idx = tf.squeeze(tf.where(y > 0.5), axis=1)  # indices of winners
            k = tf.shape(pos_idx)[0]
            # If no positives (shouldn't happen), return 0
            def _zero():
                return tf.constant(0.0, dtype=tf.float32)

            def _do():
                nll = 0.0
                # Average over R random permutations of the winners (order is latent)
                def body(r, acc):
                    perm = tf.random.shuffle(pos_idx)
                    # mask_add will set chosen items to -inf to emulate removal (without replacement)
                    mask_add = tf.zeros_like(logit)
                    step_nll = 0.0
                    j = tf.constant(0)
                    def step(j, step_acc, mask_add):
                        denom = tf.reduce_logsumexp(logit + mask_add)
                        chosen = perm[j]
                        step_acc += -(tf.gather(logit, chosen) - denom)
                        # remove chosen from future denominators
                        mask_add = tf.tensor_scatter_nd_update(mask_add, tf.reshape(chosen, [1,1]), tf.reshape(neg_inf, [1]))
                        return j + 1, step_acc, mask_add
                    cond = lambda j, *_: tf.less(j, k)
                    _, step_nll, _ = tf.while_loop(cond, step, [j, step_nll, mask_add])
                    return r + 1, acc + step_nll
                r0 = tf.constant(0)
                _, total = tf.while_loop(lambda r, *_: tf.less(r, R), body, [r0, tf.constant(0.0)])
                return total / tf.cast(tf.maximum(R, 1), tf.float32)
            return tf.cond(tf.equal(tf.size(pos_idx), 0), _zero, _do)

        # Map across batch
        per_example = tf.map_fn(_sample_pl_nll, (y_true, logits), dtype=tf.float32)
        # Scale by number of winners (≈6) for stability
        k = tf.reduce_sum(y_true, axis=1) + 1e-6
        return tf.reduce_mean(per_example / k)

    return _loss

# Optional pairwise Bradley–Terry (winners vs non‑winners) loss
# Not used by default, but available for experiments.

# Strengthen hard-negative sampling for the pairwise loss
def bt_pairwise_loss_factory(num_neg=20):
    """Sampled pairwise logistic loss: for each winner, sample `num_neg` non‑winners."""
    @tf.function(experimental_relax_shapes=True)
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        def _one(args):
            y, s = args
            pos_idx = tf.squeeze(tf.where(y > 0.5), axis=1)
            neg_idx = tf.squeeze(tf.where(y < 0.5), axis=1)
            num_p = tf.shape(pos_idx)[0]
            num_n = tf.shape(neg_idx)[0]
            def _zero():
                return tf.constant(0.0, dtype=tf.float32)
            def _do():
                # hard-negative sampling: take the top-k negatives by current logits
                k = tf.minimum(num_neg * tf.maximum(num_p, 1), num_n)
                neg_scores = tf.gather(s, neg_idx)
                order = tf.argsort(neg_scores, direction='DESCENDING')
                neg_samp = tf.gather(neg_idx, order)[:k]
                # broadcast pairwise differences s_pos - s_neg
                s_pos = tf.gather(s, pos_idx)
                s_neg = tf.gather(s, neg_samp)
                diff = tf.expand_dims(s_pos, 1) - tf.expand_dims(s_neg, 0)
                return tf.reduce_mean(tf.math.softplus(-diff))
            return tf.cond(tf.logical_or(tf.equal(num_p, 0), tf.equal(num_n, 0)), _zero, _do)
        per_ex = tf.map_fn(_one, (y_true, y_pred), dtype=tf.float32)
        return tf.reduce_mean(per_ex)
    return _loss

def build_tcnn_model(input_shape, output_dim):
    """
    DeepSets-style set encoder for permutation-invariant per-ball scoring.
    Pipeline: per-element Φ MLP → global mean pooling → concatenate global to each element → ρ MLP → 1×1 conv ⇒ logits
    Returns logits of shape [batch, 40].
    """
    inputs = layers.Input(shape=input_shape)        # (40, n_features)
    # Φ: element-wise embedding
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    # Global set summary (permutation-invariant)
    g = layers.GlobalAveragePooling1D()(x)
    g = layers.Dense(64, activation='relu')(g)
    g = layers.Dense(64, activation='relu')(g)
    # Repeat the global summary to match the per-number axis length (static timesteps)
    timesteps = input_shape[0] if (input_shape and input_shape[0] is not None) else 40
    g_rep = layers.RepeatVector(timesteps)(g)
    x = layers.Concatenate(axis=-1)([x, g_rep])
    # ρ: element-wise scoring MLP
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    logits = layers.Conv1D(1, kernel_size=1)(x)           # (batch, 40, 1)
    logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1))(logits)  # (batch, 40)

    model = keras.Model(inputs=inputs, outputs=logits)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss=pl_set_loss_factory(R=8, tau=0.20))
    return model

# --- 3.3. Transformer Model Implementation ---
def build_transformer_model(input_shape, output_dim, num_heads=4, ff_dim=96):
    """
    Set Transformer style: stack of self-attention blocks (permutation-equivariant),
    then per-element projection to logits.
    Returns logits of shape [batch, 40].
    """
    inputs = layers.Input(shape=input_shape)  # (40, n_features)
    x = inputs
    # SAB block 1
    attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=8)(x, x)
    attn1 = layers.Dropout(0.2)(attn1)
    x = layers.Add()([x, attn1])
    x = layers.LayerNormalization()(x)
    ff1 = layers.Dense(ff_dim, activation='relu')(x)
    ff1 = layers.Dropout(0.2)(ff1)
    ff1 = layers.Dense(input_shape[-1])(ff1)
    x = layers.Add()([x, ff1])
    x = layers.LayerNormalization()(x)
    # SAB block 2
    attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=8)(x, x)
    attn2 = layers.Dropout(0.2)(attn2)
    y = layers.Add()([x, attn2])
    y = layers.LayerNormalization()(y)
    ff2 = layers.Dense(ff_dim, activation='relu')(y)
    ff2 = layers.Dropout(0.2)(ff2)
    ff2 = layers.Dense(input_shape[-1])(ff2)
    y = layers.Add()([y, ff2])
    y = layers.LayerNormalization()(y)

    # Per-element projection to logits (equivariant → scores per ball)
    logits = layers.Conv1D(1, kernel_size=1)(y)           # (batch, 40, 1)
    logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1))(logits)  # (batch, 40)

    model = keras.Model(inputs=inputs, outputs=logits)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss=pl_set_loss_factory(R=8, tau=0.20))
    return model

# --- MC‑dropout helper (seed averaging without retraining) ---
def _mc_dropout_logits(model, x, n_passes=11):
    """
    Run the model n_passes times with dropout active and average logits.
    This reduces variance and over‑confidence without retraining.
    """
    outs = []
    for _ in range(n_passes):
        # Calling the model with training=True enables Dropout during inference
        outs.append(model(x, training=True))
    import tensorflow as _tf
    return _tf.reduce_mean(_tf.stack(outs, axis=0), axis=0).numpy()

# --- 3.4. Training the Models with Early Stopping ---
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=1e-4, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5, verbose=0)

# Split data into train/validation sets (e.g., 80/20 split)
_assert_chronological_frame(data)
split_idx = int(len(X_seq) * 0.8)
X_train_dl, X_val_dl = X_seq[:split_idx], X_seq[split_idx:]
y_train_dl, y_val_dl = y_seq[:split_idx], y_seq[split_idx:]

print("Training Temporal CNN model...")
# Input shape is now (40, n_features)
lstm_model = build_tcnn_model(input_shape=X_train_dl.shape[1:], output_dim=40)
lstm_model.fit(
    X_train_dl, y_train_dl,
    validation_data=(X_val_dl, y_val_dl),
    epochs=300,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# Train Transformer Model
print("Training Transformer model...")
transformer_model = build_transformer_model(input_shape=X_train_dl.shape[1:], output_dim=40)
transformer_model.fit(
    X_train_dl, y_train_dl,
    validation_data=(X_val_dl, y_val_dl),
    epochs=1000,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)


# ================================================================
# Expanding-window fine-tuning for neural nets (every 8 draws)
# ------------------------------------------------
# Fine-tune copies of the CNN ("lstm_model") and Transformer on the
# most recent [120..200] supervised pairs strictly before t_idx.
# Cached per 8-draw bin to keep cost low.
FINE_TUNE_EVERY = 8
FT_MIN = 120
FT_MAX = 200
FT_EPOCHS = 20
_ft_cache = {}  # key: 8-draw bin -> (tcnn_ft, transformer_ft)

def _prepare_recent_supervised(t_idx, window_min=FT_MIN, window_max=FT_MAX):
    """
    Build (X_recent, y_recent) using only draws strictly before t_idx.
    We take the last [window_min..window_max] next-draw supervised pairs.
    """
    _set_target_idx(t_idx)
    start_idx = max(k, t_idx - window_max)
    end_idx = t_idx  # exclusive (labels go up to t_idx-1)
    if (end_idx - start_idx) < window_min:
        start_idx = max(k, end_idx - window_min)
    Xr, yr = [], []
    for i in range(start_idx, end_idx):
        if i - k < 0:
            continue
        feats = compute_stat_features(draws[i - k:i], i - 1)  # (40, n_features)
        Xr.append(feats)
        yr.append([1 if n in draws[i] else 0 for n in range(1, 41)])
    if len(Xr) == 0:
        _set_target_idx(None)
        return None, None
    _set_target_idx(None)
    return np.array(Xr), np.array(yr)

def _get_finetuned_nets(t_idx):
    """
    Return (tcnn_ft, transformer_ft) tuned on draws[:t_idx].
    Models are cached per 10-draw bin for efficiency.
    """
    _set_target_idx(t_idx)
    if t_idx <= k + FT_MIN:
        _set_target_idx(None)
        return lstm_model, transformer_model

    bin_key = int(t_idx // FINE_TUNE_EVERY)
    if bin_key in _ft_cache:
        _set_target_idx(None)
        return _ft_cache[bin_key]

    Xr, yr = _prepare_recent_supervised(t_idx)
    if Xr is None or len(Xr) < FT_MIN:
        _set_target_idx(None)
        return lstm_model, transformer_model

    # Time-ordered split: last 10% for validation
    split = max(1, int(len(Xr) * 0.9))
    Xr_tr, Xr_val = Xr[:split], Xr[split:]
    yr_tr, yr_val = yr[:split], yr[split:]

    # Clone architectures and initialise from global weights
    tcnn_ft = build_tcnn_model(input_shape=X_train_dl.shape[1:], output_dim=40)
    tcnn_ft.set_weights(lstm_model.get_weights())
    trans_ft = build_transformer_model(input_shape=X_train_dl.shape[1:], output_dim=40)
    trans_ft.set_weights(transformer_model.get_weights())

    # Lower LR for fine‑tuning to curb overfitting (keep same PL loss)
    tcnn_ft.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=pl_set_loss_factory(R=8, tau=0.20)
    )
    trans_ft.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=pl_set_loss_factory(R=8, tau=0.20)
    )

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
    # Short fine-tune to avoid overfitting / keep cost low
    tcnn_ft.fit(Xr_tr, yr_tr, validation_data=(Xr_val, yr_val),
                epochs=FT_EPOCHS, batch_size=8, verbose=0, shuffle=True, callbacks=[es])
    trans_ft.fit(Xr_tr, yr_tr, validation_data=(Xr_val, yr_val),
                 epochs=FT_EPOCHS, batch_size=8, verbose=0, shuffle=True, callbacks=[es])

    _ft_cache[bin_key] = (tcnn_ft, trans_ft)
    _set_target_idx(None)
    return _ft_cache[bin_key]

def _predict_with_nets(feats_batch, t_idx, use_finetune=True, mc_passes=11):
    """
    Given a feature batch of shape (1, 40, n_features), return
    (probs_tcnn, probs_transformer) arrays of shape (40,).
    If use_finetune=True, use expanding-window fine-tuned models for the 10-draw bin of t_idx.
    mc_passes controls the number of MC‑dropout passes used to average logits.
    """
    _set_target_idx(t_idx)
    mdl_tcnn, mdl_trans = (lstm_model, transformer_model)
    if use_finetune:
        mdl_tcnn, mdl_trans = _get_finetuned_nets(t_idx)
    logits_l = _mc_dropout_logits(mdl_tcnn, feats_batch, n_passes=mc_passes)[0]
    logits_t = _mc_dropout_logits(mdl_trans, feats_batch, n_passes=mc_passes)[0]
    probs_l = tf.nn.softmax(logits_l).numpy()
    probs_t = tf.nn.softmax(logits_t).numpy()
    _set_target_idx(None)
    return probs_l, probs_t

# --- 3.5. Prediction using Deep Models ---
# For prediction, use the most recent k draws as input (with derived features)
last_k_draws = draws[-k:]
_set_target_idx(n_draws)  # allow features to use indices < n_draws only
features_pred = compute_stat_features(last_k_draws, n_draws - 1)
features_pred = features_pred.reshape(1, features_pred.shape[0], features_pred.shape[1])
_set_target_idx(None)

# Use expanding-window fine-tuned nets for the latest prediction
_l_probs, _t_probs = _predict_with_nets(features_pred, t_idx=n_draws, use_finetune=True)
lstm_prob = {num: float(_l_probs[num-1]) for num in range(1, 41)}
transformer_prob = {num: float(_t_probs[num-1]) for num in range(1, 41)}

# 4. **Combining Models for Prediction**
# We will predict the next draw (after the last known draw in the data).
last_k_draws = draws[-k:]  # the most recent k draws from history

# Compute probability for each number to be in the next draw according to each component:

# (a) Bayesian probability (long-term frequency)
bayes_prob = posterior_probs  # already computed above


# (b) Markov chain probability based on last draw and multi-step transitions:
# We combine 1-step, 2-step, and 3-step transitions for richer modeling.
last_draw = draws[-1]
last_2_draw = draws[-2] if n_draws > 1 else set()
last_3_draw = draws[-3] if n_draws > 2 else set()

# 1-step Markov score
markov_score_1 = {num: 0.0 for num in range(1, 41)}
for prev_num in last_draw:
    for candidate in range(1, 41):
        markov_score_1[candidate] += transition_probs[prev_num].get(candidate, 0)
# 2-step Markov score
markov_score_2 = {num: 0.0 for num in range(1, 41)}
for prev_num in last_2_draw:
    for candidate in range(1, 41):
        markov_score_2[candidate] += transition2_probs[prev_num].get(candidate, 0)
# 3-step Markov score
markov_score_3 = {num: 0.0 for num in range(1, 41)}
for prev_num in last_3_draw:
    for candidate in range(1, 41):
        markov_score_3[candidate] += transition3_probs[prev_num].get(candidate, 0)

# Weighted sum of multi-step Markov scores (weights can be tuned)
markov_score = {num: (0.5 * markov_score_1[num] +
                      0.3 * markov_score_2[num] +
                      0.2 * markov_score_3[num]) for num in range(1, 41)}

# Normalize Markov scores into probabilities
markov_prob = {}
total_score = sum(markov_score.values())
if total_score > 0:
    for num in range(1, 41):
        markov_prob[num] = markov_score[num] / total_score
else:
    markov_prob = {num: 1.0/40 for num in range(1, 41)}

# Co-occurrence conditional prob based on within-draw co-occurrence
cooc_prob = _cooc_conditional_prob_from_history(draws[:-1], last_draw)

# Without-replacement compatibility features (top-k competitor sums) for each base
COMP_K = 5
compat_bayes   = _compat_topk_sum(bayes_prob, k=COMP_K)
compat_markov  = _compat_topk_sum(markov_prob, k=COMP_K)
compat_hmm     = _compat_topk_sum(hmm_prob, k=COMP_K)
compat_nn      = _compat_topk_sum({n: 0.5*lstm_prob[n] + 0.5*transformer_prob[n] for n in range(1, 41)}, k=COMP_K)

# Helper primitives needed by regime/weekday logic (must be defined early)
# ------------------------------------------------
def _regime_features_at_t(t_idx):
    """Return dict with regime features at index t_idx (using dates array `data`).
    Keys: 'weekday' [0..6], 'month' [1..12], 'gap_days' (float), 'schedule_flip' {0,1}.
    schedule_flip = 1 iff day gap NOT in {2,3} (deviates from M/W/Sa cadence).
    """
    if t_idx <= 0:
        return dict(weekday=0, month=1, gap_days=3.0, schedule_flip=0)
    d_curr = pd.to_datetime(data.iloc[t_idx]['DrawDate'])
    d_prev = pd.to_datetime(data.iloc[t_idx-1]['DrawDate'])
    gap_days = float((d_curr - d_prev).days)
    weekday = int(d_curr.weekday())
    month = int(d_curr.month)
    schedule_flip = 0 if gap_days in (2, 3) else 1
    return dict(weekday=weekday, month=month, gap_days=gap_days, schedule_flip=schedule_flip)

def _rolling_entropy_from_history(sub_draws, window=30):
    """Entropy of number distribution over the last `window` draws of `sub_draws`."""
    if len(sub_draws) == 0:
        return 0.0
    use = sub_draws[-min(window, len(sub_draws)) : ]
    cnt = np.zeros(40, dtype=float)
    for d in use:
        for n in d:
            cnt[n-1] += 1
    p = cnt / max(1.0, cnt.sum())
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def _dispersion_last_draw(sub_draws):
    """Standard deviation of the numbers in the last draw of sub_draws."""
    if len(sub_draws) == 0:
        return 0.0
    vals = np.array(sorted(list(sub_draws[-1])), dtype=float)
    return float(np.std(vals))

# Regime + volatility descriptors available to stacker
reg_now = _regime_features_at_t(n_draws - 1)
roll_ent_now = _rolling_entropy_from_history(draws, window=30)
roll_disp_now = _dispersion_last_draw(draws)


# (c) Deep learning probabilities: use LSTM and Transformer models to predict probabilities for next draw.
# We average their probabilities for a robust neural network component.
nn_prob = {}
for num in range(1, 41):
    # Average LSTM and Transformer probabilities for each number
    nn_prob[num] = 0.5 * lstm_prob[num] + 0.5 * transformer_prob[num]

# (d) Recent trend heuristic (optional): e.g., numbers that haven’t appeared in a long time might catch up (cold -> hot).
# We measure "gap" from last appearance in entire history:
last_seen = {num: -1 for num in range(1, 41)}
for idx, draw in enumerate(draws):
    for num in draw:
        last_seen[num] = idx
gaps_full = {num: (n_draws - 1 - last_seen[num]) for num in range(1, 41)}  # gap since last appearance (in draws)
# Convert gaps to a probability boost (larger gap -> slightly higher boost). We'll normalize this.
max_gap = max(gaps_full.values())
gap_boost = {num: gaps_full[num] / max_gap for num in range(1, 41)}  # 0 to 1 scale



#####################################################################
# 5. **Sophisticated Weighted Stacking Ensemble Integration**
# ----------------------------------------------------------
# We use a stacking ensemble approach to combine the predictions from:
# (1) Bayesian model, (2) Markov, (3) HMM, (4) LSTM, (5) Transformer.
# The meta-model is a Gradient Boosting classifier (e.g., XGBoost).
# The weights of the base models are dynamically adjusted based on their historical predictive accuracy.

# --- Stacking Ensemble Implementation ---
# ---------------------------------------
# 1. For each historical draw in the validation set, collect the predicted probabilities for each number
#    from each base model (Bayesian, Markov, HMM, LSTM, Transformer).
# 2. For each number in each draw, form a feature vector of the base model predictions.
# 3. The target is whether that number actually appeared in the next draw (binary label).
# 4. Train a meta-model (Gradient Boosting Classifier) to combine the base predictions.
# 5. Use the meta-model to predict the next draw, using the latest base model outputs as features.

from sklearn.metrics import log_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRanker
from xgboost.callback import EarlyStopping

# --- Hard patch: sanitize XGBRanker.fit across xgboost 3.0.x ---
try:
    import xgboost as _xgb
    _XGB_ORIG_FIT = XGBRanker.fit
    def _XGB_SAFE_FIT(self, X, y, **kwargs):
        # Drop fragile/unsupported kwargs on Ranker in 3.0.x and avoid group-weight mismatch
        for bad in ("callbacks", "early_stopping_rounds", "verbose_eval", "sample_weight", "eval_sample_weight"):
            kwargs.pop(bad, None)
        # Keep only benign kwargs. qid / eval_set / eval_qid are OK.
        return _XGB_ORIG_FIT(self, X, y, **kwargs)
    XGBRanker.fit = _XGB_SAFE_FIT
except Exception as _e:
    warnings.warn(f"XGBRanker patch failed: {_e}")

def _safe_fit_xgbranker(model, X_tr, y_tr, qid_tr=None, eval_set=None, eval_qid=None, **fit_kwargs):
    """
    Fit XGBRanker robustly across xgboost versions (incl. 3.0.x):
      - ignores unsupported kwargs like 'callbacks', 'early_stopping_rounds', 'verbose_eval'
      - drops any per-row weights (sample_weight / eval_sample_weight) when ranking with qid
      - progressively falls back: (qid + evals) -> (qid only) -> (no qid, no evals)
    """
    import xgboost as xgb  # local import

    # 0) Never pass fragile kwargs through
    _ = fit_kwargs  # intentionally unused

    # 1) Minimal safe args
    args_qid_evals = {}
    if qid_tr is not None:
        args_qid_evals["qid"] = qid_tr
    if eval_set is not None:
        args_qid_evals["eval_set"] = eval_set
    if eval_qid is not None:
        args_qid_evals["eval_qid"] = eval_qid

    # 2) Try full (qid + evals)
    try:
        return model.fit(X_tr, y_tr, **args_qid_evals)
    except TypeError:
        # Some wrappers reject eval_* args; try qid only
        try:
            args_qid_only = {"qid": qid_tr} if qid_tr is not None else {}
            return model.fit(X_tr, y_tr, **args_qid_only)
        except Exception:
            pass
    except xgb.core.XGBoostError as e:
        # Group-weights mismatch or similar -> drop evals first
        if "Size of weight must equal to the number of query groups" in str(e):
            try:
                args_qid_only = {"qid": qid_tr} if qid_tr is not None else {}
                return model.fit(X_tr, y_tr, **args_qid_only)
            except Exception:
                pass
        # Fall through to last resort for other booster errors

    # 3) Last resort: no qid, no evals
    return model.fit(X_tr, y_tr)
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
# --- Added for weekday weight optimisation ---
from skopt import gp_minimize
from skopt.space import Real

# --- Helper functions used by weighting/stacking (defined before first use) ---

def _norm_prob_dict_local(d):
    s = float(sum(d.get(n, 0.0) for n in range(1, 41)))
    if s <= 0:
        return {n: 1.0/40 for n in range(1, 41)}
    return {n: float(d.get(n, 0.0)) / s for n in range(1, 41)}

# --- Inserted helper functions for safe weights and PL pseudo-loglikelihood ---
def _safe_weights_from_prob_dict(prob_dict, eps=1e-12):
    """Convert a prob dict {1..40 -> p} to a safe weight vector (len 40).
    Clips to [eps, 1] and renormalises to sum to 1.0.
    """
    v = np.array([prob_dict.get(n, 0.0) for n in range(1, 41)], dtype=float)
    v = np.maximum(v, eps)
    s = float(v.sum())
    if s <= 0:
        return np.full(40, 1.0/40, dtype=float)
    return (v / s)

def pl_set_pseudologlik(w, winners, R=16, rng=None):
    """Pseudo log-likelihood of an unordered 6-set `winners` under a
    Plackett–Luce model with weights `w` (dict or length-40 array).
    We average the sequential log-likelihood over R random permutations.
    Returns the *log*-likelihood (not negative), so higher is better.
    """
    # Convert weights to probability vector length 40
    if isinstance(w, dict):
        p = np.array([w.get(n, 0.0) for n in range(1, 41)], dtype=float)
    else:
        p = np.array(w, dtype=float).reshape(-1)
        if p.size != 40:
            if p.size < 40:
                p = np.pad(p, (0, 40 - p.size), constant_values=0.0)
            else:
                p = p[:40]
    p = np.maximum(p, 1e-12)
    p = p / (p.sum() + 1e-12)

    winners_idx = np.array(sorted(list(winners)), dtype=int) - 1
    R = max(1, int(R))
    if rng is None:
        rng = np.random.default_rng(12345)

    total = 0.0
    for _ in range(R):
        perm = rng.permutation(winners_idx)
        denom = float(np.sum(p))
        acc = 0.0
        for j in perm:
            pj = float(p[j])
            acc += np.log(max(pj, 1e-12)) - np.log(max(denom, 1e-12))
            denom -= pj  # remove chosen without replacement
        total += acc
    return float(total) / float(R)


# --- Evaluation & diagnostics helpers (walk-forward backtest, reliability, oracle ablation) ---

def _nll_from_prob_dict(P_dict, winners):
    """Return PL negative log-likelihood for an unordered 6-set `winners`
    under probability dict P_dict (keys 1..40)."""
    try:
        v = np.array([float(P_dict.get(n, 0.0)) for n in range(1, 41)], dtype=float)
        v = np.clip(v, 1e-12, None)
        v = v / (v.sum() + 1e-12)
        return float(-pl_set_pseudologlik(v, winners, R=16))
    except Exception:
        return float("inf")


def _per_expert_prob_dicts_at_t(t_idx):
    """Return base probability dicts at historical index t_idx using only draws[:t_idx]."""
    t_idx = int(t_idx)
    if t_idx <= 0 or t_idx >= len(draws):
        return None
    history = draws[:t_idx]

    # Base sources (strictly pre-t_idx)
    bayes_t   = _norm_prob_dict_local(compute_bayes_posterior(history, alpha=1))
    markov_t  = _markov_prob_from_history(history)
    hmm_t     = _build_hmm_prob_from_subset(history)

    # Neural nets features at time t_idx
    feats_t = compute_stat_features(draws[t_idx - k: t_idx], t_idx - 1)
    feats_t = feats_t.reshape(1, feats_t.shape[0], feats_t.shape[1])
    _l_t, _t_t = _predict_with_nets(feats_t, t_idx=t_idx, use_finetune=True, mc_passes=MC_STACK_PASSES)
    lstm_t = {n: float(_l_t[n-1]) for n in range(1, 41)}
    trans_t = {n: float(_t_t[n-1]) for n in range(1, 41)}
    return {"Bayes": bayes_t, "Markov": markov_t, "HMM": hmm_t, "LSTM": lstm_t, "Transformer": trans_t}


def _per_expert_pl_nll_at(t_idx):
    """Compute per-expert PL-NLLs at historical index t_idx. Returns list aligned to _per_expert_names()."""
    bases = _per_expert_prob_dicts_at_t(t_idx)
    if bases is None:
        return None
    winners = draws[int(t_idx)]
    out = []
    for name in _per_expert_names():
        out.append(_nll_from_prob_dict(bases[name], winners))
    return out


def _stacked_probs_from_feats(latest_feats):
    """Apply the current stacker + optional calibrator/temperature to a (1,40,F) feature array."""
    meta = globals().get("meta_model", None)
    if meta is None:
        # Fall back to equal-weight blend if meta not ready
        P = _live_mixture_prob()
        return np.array([[P[n] for n in range(1, 41)]], dtype=float)

    raw = meta.predict(latest_feats).reshape(1, 40)
    cal = globals().get("postrank_calibrator", None)
    beta = float(globals().get("best_beta", 1.0))
    if cal is not None:
        try:
            cal_probs = cal.predict_proba(raw)
            cal_probs = np.power(cal_probs, beta)
            cal_probs = cal_probs / (np.sum(cal_probs, axis=1, keepdims=True) + 1e-12)
            return cal_probs
        except Exception:
            pass
    # Logistic + temperature as fallback
    probs = 1.0 / (1.0 + np.exp(-raw))
    probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-12)
    probs = np.power(probs, beta)
    probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-12)
    return probs


def _latest_feats_at_t(t_idx, overrides=None):
    """
    Rebuild the meta features at historical index t_idx, with optional overrides for base experts.
    overrides: dict with any of {"Bayes","Markov","HMM","LSTM","Transformer"} -> prob_dict {1..40->p}
    Mirrors the construction used in _main_ensemble_prob_at_t().
    """
    t_idx = int(t_idx)
    history = draws[:t_idx]
    bases = _per_expert_prob_dicts_at_t(t_idx)
    if overrides:
        for k_over in overrides:
            if k_over in bases and isinstance(overrides[k_over], dict):
                bases[k_over] = _norm_prob_dict_local(overrides[k_over])

    markov1_t = _markov1_prob_from_history(history)
    last_draw_t = history[-1] if len(history) > 0 else set()
    cooc_t = _cooc_conditional_prob_from_history(history, last_draw_t)
    compat_k = 5
    compat_bayes_t  = _compat_topk_sum(bases["Bayes"],       k=compat_k)
    compat_markov_t = _compat_topk_sum(bases["Markov"],      k=compat_k)
    compat_hmm_t    = _compat_topk_sum(bases["HMM"],         k=compat_k)
    nn_t = {n: 0.5*bases["LSTM"][n] + 0.5*bases["Transformer"][n] for n in range(1, 41)}
    compat_nn_t     = _compat_topk_sum(nn_t,                  k=compat_k)
    reg_t = _regime_features_at_t(t_idx)
    roll_ent_t = _rolling_entropy_from_history(history, window=30)
    roll_disp_t = _dispersion_last_draw(history)

    latest_feats_t = np.array([
        [
            bases["Bayes"].get(num, 0.0),
            bases["Markov"].get(num, 0.0),
            bases["HMM"].get(num, 0.0),
            float(bases["LSTM"][num]),
            float(bases["Transformer"][num]),
            markov1_t.get(num, 0.0),
            cooc_t.get(num, 0.0),
            compat_bayes_t[num],
            compat_markov_t[num],
            compat_hmm_t[num],
            compat_nn_t[num],
            float(reg_t['weekday']),
            float(reg_t['month']),
            float(reg_t['gap_days']),
            float(reg_t['schedule_flip']),
            float(roll_ent_t),
            float(roll_disp_t),
        ]
        for num in range(1, 41)
    ], dtype=float)
    return latest_feats_t.reshape(1, latest_feats_t.shape[0], latest_feats_t.shape[1])


def _stacked_distribution_at_t(t_idx, overrides=None):
    """Return dict {1..40->p} for historical index t_idx using the stacker, with optional expert overrides."""
    feats = _latest_feats_at_t(t_idx, overrides=overrides)
    probs = _stacked_probs_from_feats(feats)[0]
    probs = np.clip(probs, 1e-12, None)
    probs = probs / (np.sum(probs) + 1e-12)
    return {n: float(probs[n-1]) for n in range(1, 41)}


def run_walk_forward_backtest(last_N=150, log_path="run_log.jsonl"):
    """
    Strict rolling walk-forward over the last `last_N` draws.
    Logs per-draw PL-NLL, top-6 recall, hit>=1; records per-expert PL-NLL and deltas; saves CSV + reliability data.
    Returns a small summary dict.
    """
    if 'best_beta' not in globals():
        globals()['best_beta'] = 1.0

    start = max(k + 5, n_draws - int(last_N))
    rows = []
    preds_flat, actual_flat = [], []
    prev_expert_nll = None
    expert_names = _per_expert_names()

    for t in range(start, n_draws):
        winners = draws[t]

        # Final stacker distribution at t
        P_final = _stacked_distribution_at_t(t)

        # Metrics
        top6 = sorted(range(1, 41), key=lambda n: P_final.get(n, 0.0), reverse=True)[:6]
        hits = len(set(top6).intersection(winners))
        top6_recall = hits / 6.0
        hit_any = 1 if hits >= 1 else 0
        main_nll = _nll_from_prob_dict(P_final, winners)

        # Per-expert NLLs and deltas
        nlls_list = _per_expert_pl_nll_at(t) or [float("inf")]*5
        deltas = {}
        if prev_expert_nll is not None:
            for name, cur, prev in zip(expert_names, nlls_list, prev_expert_nll):
                deltas[name] = float(cur - prev)
        else:
            for name in expert_names:
                deltas[name] = None
        prev_expert_nll = nlls_list

        # Reliability flattening
        for n in range(1, 41):
            preds_flat.append(float(P_final.get(n, 0.0)))
            actual_flat.append(1.0 if n in winners else 0.0)

        # JSONL record
        try:
            rec = {
                "type": "walk",
                "ts": _now_iso(),
                "t_idx": int(t),
                "date": str(pd.to_datetime(data.iloc[t]['DrawDate']).date()),
                "main_pl_nll": float(main_nll),
                "top6_recall": float(top6_recall),
                "hit_any": int(hit_any),
                "blend_weights": [float(x) for x in (globals().get("weights", np.ones(5)/5.0))],
                "per_expert_pl_nll": {name: float(val) for name, val in zip(expert_names, nlls_list)},
                "per_expert_pl_nll_delta": deltas,
                "pred_top6": top6,
                "winners": sorted(list(winners)),
            }
            _log_jsonl(rec, path=log_path)
        except Exception:
            pass

        rows.append([
            t,
            str(pd.to_datetime(data.iloc[t]['DrawDate']).date()),
            main_nll,
            top6_recall,
            hit_any,
            *nlls_list
        ])

    # Save CSV
    try:
        import csv as _csv
        header = ["t_idx","date","main_pl_nll","top6_recall","hit_any"] + [f"nll_{n}" for n in expert_names]
        with open("backtest_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    except Exception:
        pass

    # Reliability curve
    summary_rel = _build_reliability(preds_flat, actual_flat, n_bins=10)

    # Summary record
    try:
        summary = {
            "type": "backtest_summary",
            "ts": _now_iso(),
            "n_evals": len(rows),
            "avg_main_pl_nll": float(np.mean([r[2] for r in rows])) if rows else None,
            "avg_top6_recall": float(np.mean([r[3] for r in rows])) if rows else None,
            "hit_any_rate": float(np.mean([r[4] for r in rows])) if rows else None,
        }
        _log_jsonl(summary, path=log_path)
    except Exception:
        summary = None
    return summary


def _build_reliability(preds_flat, actual_flat, n_bins=10):
    """Build and save reliability data for the final 40-way probs."""
    preds = np.asarray(preds_flat, dtype=float).reshape(-1)
    y = np.asarray(actual_flat, dtype=float).reshape(-1)
    eps = 1e-12
    preds = np.clip(preds, eps, 1.0 - eps)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(preds, bins) - 1
    bin_stats = []
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask):
            bin_stats.append([float(bins[b]), float(bins[b+1]), 0, None, None])
            continue
        p_mean = float(np.mean(preds[mask]))
        y_rate = float(np.mean(y[mask]))
        bin_stats.append([float(bins[b]), float(bins[b+1]), int(mask.sum()), p_mean, y_rate])

    # Save CSV
    try:
        import csv as _csv
        with open("reliability_curve.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["bin_lo","bin_hi","count","pred_mean","obs_rate"])
            w.writerows(bin_stats)
    except Exception:
        pass

    # Save PNG plot (headless-safe)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [ (lo+hi)/2.0 for lo,hi,_,_,_ in bin_stats ]
        pm = [ (m if m is not None else 0.0) for _,_,_,m,_ in bin_stats ]
        yr = [ (r if r is not None else 0.0) for _,_,_,_,r in bin_stats ]
        plt.figure(figsize=(5,5))
        plt.plot([0,1],[0,1], linestyle="--")
        plt.scatter(pm, yr)
        plt.xlabel("Predicted probability (bin mean)")
        plt.ylabel("Observed frequency")
        plt.title("Reliability (Final Ensemble)")
        plt.tight_layout()
        plt.savefig("reliability_curve.png", dpi=160)
        plt.close()
    except Exception:
        pass

    # Log to JSONL
    try:
        _log_jsonl({"type":"reliability","ts":_now_iso(),"bins":bin_stats}, path="run_log.jsonl")
    except Exception:
        pass

    return {"bins": bin_stats}


def run_oracle_ablation(last_N=100, log_path="run_log.jsonl"):
    """
    For each draw in the last `last_N`, replace ONE expert with a perfect oracle
    (uniform mass over the true 6 winners) and compute the improvement in PL-NLL
    over the baseline stacker. Returns a dict of mean improvements per expert.
    """
    start = max(k + 5, n_draws - int(last_N))
    names = _per_expert_names()
    gains = {name: [] for name in names}

    for t in range(start, n_draws):
        winners = draws[t]
        base_dist = _stacked_distribution_at_t(t)
        base_nll = _nll_from_prob_dict(base_dist, winners)
        oracle = {n: (1.0/6.0 if n in winners else 1e-12) for n in range(1, 41)}
        s = sum(oracle.values())
        oracle = {n: oracle[n] / s for n in range(1, 41)}

        for name in names:
            override = {name: oracle}
            dist = _stacked_distribution_at_t(t, overrides=override)
            nll = _nll_from_prob_dict(dist, winners)
            gains[name].append(base_nll - nll)  # positive = improvement

    means = {name: float(np.mean(vals)) if vals else None for name, vals in gains.items()}
    try:
        _log_jsonl({"type":"oracle_ablation","ts":_now_iso(),"mean_improvement":means}, path=log_path)
    except Exception:
        pass
    return means

# --- Decision policy helpers: mixture construction + PL sampling with temperature & diversity ---
def _live_mixture_prob():
    """
    Build the current live per-number probability distribution for the NEXT draw
    using the already-computed base models and learned weights. Applies the weekday
    gate if it is enabled. Returns a normalised dict {1..40 -> p}.
    """
    try:
        # Fallback-safe getters for each base distribution
        def _get(d, n): 
            return float(d.get(n, 0.0)) if isinstance(d, dict) else 0.0
        # Compose base vector for each n
        P = {}
        for n in range(1, 41):
            bv = [
                _get(globals().get("bayes_prob", {}), n),
                _get(globals().get("markov_prob", {}), n),
                _get(globals().get("hmm_prob", {}), n),
                float(globals().get("lstm_prob", {}).get(n, 0.0)),
                float(globals().get("transformer_prob", {}).get(n, 0.0)),
            ]
            w = globals().get("weights", np.ones(5, dtype=float) / 5.0)
            P[n] = float(np.dot(np.asarray(w, dtype=float), np.asarray(bv, dtype=float)))
        # Normalise
        s = sum(P.values())
        if s > 0:
            for n in range(1, 41):
                P[n] = P[n] / s
        else:
            P = {n: 1.0/40 for n in range(1, 41)}
        # Apply weekday gate if available
        try:
            gate_fn = globals().get("_weekday_gate_for_current_run", None)
            if callable(gate_fn):
                enabled, wd_prob, w = gate_fn()
                if enabled and isinstance(wd_prob, dict) and w and w > 0:
                    P = {n: float((1.0 - w) * P.get(n, 0.0) + w * wd_prob.get(n, 0.0)) for n in range(1, 41)}
                    # Renormalise
                    s = sum(P.values())
                    if s > 0:
                        P = {n: P[n] / s for n in range(1, 41)}
        except Exception:
            pass
        return P
    except Exception:
        # Last-ditch fallback
        return {n: 1.0/40 for n in range(1, 41)}

def _pl_sample_set_from_vec(p_vec, k=6, temperature=1.15, rng=None):
    """
    Sample a size-k set without replacement under a Plackett–Luce model using
    the Gumbel–Top‑k trick. `p_vec` is length-40 probabilities.
    """
    if rng is None:
        rng = np.random.default_rng(123)
    p = np.asarray(p_vec, dtype=float).reshape(-1)
    p = np.clip(p, 1e-12, None)
    p = p / (p.sum() + 1e-12)
    # Temperature >1 flattens (more exploration), <1 sharpens
    temp = max(1e-6, float(temperature))
    logits = np.log(p) / temp
    g = -np.log(-np.log(np.clip(rng.random(p.shape[0]), 1e-12, 1.0 - 1e-12)))
    scores = logits + g
    # Take top-k
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return set((idx + 1).tolist())  # numbers are 1..40

def sample_diversified_tickets(P_dict, n_tickets=10, k=6, temperature=1.15,
                               overlap_cap=3, coverage_penalty=0.65, max_tries=400, seed=20250816):
    """
    Draw multiple diversified tickets by sampling from a (calibrated) PL distribution
    with soft temperature and enforcing cross-ticket diversity.

    - overlap_cap: maximum allowed overlap between any two tickets (e.g., 3).
    - coverage_penalty: <1.0; numbers already used are downweighted by this**usage count**.
    """
    rng = np.random.default_rng(int(seed))
    base = np.array([float(P_dict.get(n, 0.0)) for n in range(1, 41)], dtype=float)
    base = np.clip(base, 1e-12, None)
    base = base / (base.sum() + 1e-12)
    usage = np.zeros(40, dtype=int)
    tickets = []
    tries = 0

    # Helper to check diversity
    def _diverse(s, existing, cap):
        for t in existing:
            if len(s.intersection(t)) > int(cap):
                return False
        return True

    while len(tickets) < int(n_tickets) and tries < int(max_tries):
        # Softly penalise over-used numbers to improve coverage across tickets
        adj = base * (coverage_penalty ** usage)
        adj = adj / (adj.sum() + 1e-12)
        s = _pl_sample_set_from_vec(adj, k=int(k), temperature=float(temperature), rng=rng)
        if _diverse(s, tickets, overlap_cap):
            tickets.append(s)
            for idx in s:
                usage[idx - 1] += 1
        else:
            # Occasionally relax the overlap cap to make progress
            if tries in (150, 300):
                overlap_cap = min(5, int(overlap_cap) + 1)
        tries += 1
    return tickets

def recommend_tickets(n_tickets=10, k=6, temperature=1.15, overlap_cap=3, coverage_penalty=0.65):
    """
    Public wrapper: builds the live mixture and returns a list of sorted ticket lists.
    """
    P = _live_mixture_prob()
    tix = sample_diversified_tickets(
        P, n_tickets=n_tickets, k=k, temperature=temperature,
        overlap_cap=overlap_cap, coverage_penalty=coverage_penalty
    )
    return [sorted(list(t)) for t in tix]

def _markov_prob_from_history(sub_draws):
    """Compute blended 1/2/3‑step Markov probability from a subset."""
    if len(sub_draws) == 0:
        return {n: 1.0/40 for n in range(1, 41)}
    t1, t2, t3 = compute_markov_transitions(sub_draws)
    last1 = sub_draws[-1] if len(sub_draws) >= 1 else set()
    last2 = sub_draws[-2] if len(sub_draws) >= 2 else set()
    last3 = sub_draws[-3] if len(sub_draws) >= 3 else set()
    m1 = {n: 0.0 for n in range(1, 41)}
    for p in last1:
        for c in range(1, 41):
            m1[c] += t1[p].get(c, 0.0)
    m2 = {n: 0.0 for n in range(1, 41)}
    for p in last2:
        for c in range(1, 41):
            m2[c] += t2[p].get(c, 0.0)
    m3 = {n: 0.0 for n in range(1, 41)}
    for p in last3:
        for c in range(1, 41):
            m3[c] += t3[p].get(c, 0.0)
    m = {n: 0.5*m1[n] + 0.3*m2[n] + 0.2*m3[n] for n in range(1, 41)}
    return _norm_prob_dict_local(m)

# --- Pure 1-step Markov probability from history ---
def _markov1_prob_from_history(sub_draws):
    """Compute pure 1‑step Markov probability from a subset using only the last draw.
    Falls back to uniform if there is insufficient history."""
    if len(sub_draws) == 0:
        return {n: 1.0/40 for n in range(1, 41)}
    t1, _, _ = compute_markov_transitions(sub_draws)
    last1 = sub_draws[-1]
    m1 = {n: 0.0 for n in range(1, 41)}
    for p in last1:
        for c in range(1, 41):
            m1[c] += t1[p].get(c, 0.0)
    return _norm_prob_dict_local(m1)

# --- Co-occurrence and compatibility helpers ---------------------------------

def _cooccurrence_matrix_from_history(sub_draws):
    """Return symmetric 40x40 co-occurrence count matrix from draws in `sub_draws`.
    cooc[i-1, j-1] counts how often i and j appeared together in the same draw.
    """
    cooc = np.zeros((40, 40), dtype=np.int32)
    for d in sub_draws:
        s = sorted(list(d))
        for a in s:
            for b in s:
                if a != b:
                    cooc[a-1, b-1] += 1
    return cooc


def _cooc_conditional_prob_from_history(sub_draws, last_draw):
    """Compute P(j | last draw has any of S) using within-draw co-occurrence counts.
    Approximates by summing co-occurrence counts from each i in `last_draw` to candidate j.
    Returns a normalised dict over 1..40.
    """
    if len(sub_draws) == 0 or not last_draw:
        return {n: 1.0/40 for n in range(1, 41)}
    cooc = _cooccurrence_matrix_from_history(sub_draws)
    score = np.zeros(40, dtype=float)
    for i in last_draw:
        score += cooc[i-1]
    # zero self-cooccurrence preference for numbers already in last_draw
    for i in last_draw:
        score[i-1] = max(0.0, score[i-1])
    if score.sum() <= 0:
        return {n: 1.0/40 for n in range(1, 41)}
    score = score / (score.sum() + 1e-12)
    return {n: float(score[n-1]) for n in range(1, 41)}


def _compat_topk_sum(prob_dict, k=5):
    """For each candidate n, compute sum of the top-k competitor probabilities
    among the *other* numbers (exclude n). Teaches without-replacement compatibility.
    Returns dict n->scalar.
    """
    p = np.array([prob_dict.get(n, 0.0) for n in range(1, 41)], dtype=float)
    out = {}
    for n in range(1, 41):
        others = np.delete(p, n-1)
        # top-k largest among others
        topk = np.partition(others, -k)[-k:]
        out[n] = float(np.sum(topk))
    return out



def _build_hmm_prob_from_subset(draw_list, n_states=3):
    """
    Fit a small HMM on the subset and return per‑number emission means
    for the last state, normalised. Uses MultiLabelBinarizer (already imported).
    """
    if not HMM_OK:
        return _uniform_prob40()
    if len(draw_list) < 5:
        return _uniform_prob40()
    try:
        mlb_local = MultiLabelBinarizer(classes=list(range(1, 41)))
        bin_mat = mlb_local.fit_transform([sorted(list(d)) for d in draw_list])
        # MultinomialHMM on integer count vectors; shorter training and looser tol to reduce convergence noise
        hmm_local = hmm.MultinomialHMM(n_components=n_states, n_iter=50, tol=1e-2, random_state=42, verbose=False)
        with _squelch_streams(r"MultinomialHMM has undergone"):
            hmm_local.fit(bin_mat.astype(int))
            # Use helper for compatibility to get posteriors
            post = _hmm_posteriors(hmm_local, bin_mat.astype(int))
        last_st = int(np.argmax(post[-1]))
        emiss = hmm_local.emissionprob_[last_st]
        emiss = np.clip(emiss, 1e-12, None)
        probs = {n: float(emiss[n - 1]) for n in range(1, 41)}
        s = sum(probs.values())
        if s <= 0:
            return _uniform_prob40()
        return {n: probs[n] / s for n in range(1, 41)}
    except Exception:
        # Fallback to uniform if HMM fails to converge
        return _uniform_prob40()


# --- Weekday subseries / gate / historical reconstruction --------------------

def _weekday_subseries_prob_until(wd, t_idx):
    """Return a probability dict using only history from draws with weekday == wd
    strictly before index t_idx. Blends Bayes + Markov + HMM from that subseries.
    Falls back to uniform if insufficient history.
    """
    try:
        idxs = [i for i in range(min(t_idx, len(draws))) if int(pd.to_datetime(data.iloc[i]['DrawDate']).weekday()) == int(wd)]
        if len(idxs) < 6:
            return _uniform_prob40()
        sub_draws = [draws[i] for i in idxs]
        bayes_t = _norm_prob_dict_local(compute_bayes_posterior(sub_draws, alpha=1))
        markov_t = _markov_prob_from_history(sub_draws)
        hmm_t   = _build_hmm_prob_from_subset(sub_draws)
        # simple equal-weight blend
        out = {n: (bayes_t.get(n,0.0) + markov_t.get(n,0.0) + hmm_t.get(n,0.0)) / 3.0 for n in range(1,41)}
        return _norm_prob_dict_local(out)
    except Exception:
        return _uniform_prob40()


def _weekday_gate_for_current_run(N=12, min_hist=18, alpha_max=0.40):
    """
    Return (enabled, wd_prob_dict, w_weekday) for the next draw's likely weekday.
    Gate turns on only if (a) enough weekday history exists and (b) weekday expert
    beats the main ensemble on a rolling last-N weekday draws by a significant margin.
    """
    last_date = pd.to_datetime(data.iloc[n_draws - 1]['DrawDate'])
    last_wd = int(last_date.weekday())
    candidates = [(last_wd + 2) % 7, (last_wd + 3) % 7]  # M/W/Sa cadence approx.

    def _pl_nll_local(P_dict, winners):
        v = np.array([P_dict.get(n, 0.0) for n in range(1, 41)], dtype=float)
        v = np.maximum(v, 1e-12)
        v = v / (v.sum() + 1e-12)
        return -pl_set_pseudologlik(v, winners, R=16)

    best_choice, best_margin = None, -1e9
    for wd in candidates:
        idxs = [i for i in range(n_draws - 1) if pd.to_datetime(data.iloc[i]['DrawDate']).weekday() == wd]
        if len(idxs) < min_hist:
            continue
        recent = idxs[-min(N, len(idxs)):]
        diffs = []
        for ti in recent:
            P_main = _main_ensemble_prob_at_t(ti)
            P_wd   = _weekday_subseries_prob_until(wd, ti)
            actual = draws[ti]
            diffs.append(_pl_nll_local(P_main, actual) - _pl_nll_local(P_wd, actual))  # positive ⇒ weekday better
        if len(diffs) >= 5:
            mu = float(np.mean(diffs))
            sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            se = sd / max(1.0, np.sqrt(len(diffs)))
            tstat = (mu / (se + 1e-12)) if se > 0 else (1e9 if mu > 0 else -1e9)
            if (mu > 0) and (tstat > 2.0) and (mu > best_margin):
                best_choice, best_margin = wd, mu

    if best_choice is None:
        return False, None, 0.0

    # Map margin to a gated weight ∈ (0, alpha_max]
    raw = min(1.0, best_margin / 0.5)          # scale margin
    w = min(alpha_max, 0.15 + 0.25 * raw)      # base 0.15 + up to +0.25
    wd_prob = _weekday_subseries_prob_until(best_choice, n_draws)
    return True, wd_prob, float(w)


def _main_ensemble_prob_at_t(t_idx):
    """Reconstruct the ensemble distribution as of time t_idx (pre-weekday-gate).
    Uses only draws[:t_idx]. Mirrors the live pipeline but **without** the weekday gate.
    """
    # Guard and history
    t_idx = int(t_idx)
    if t_idx <= 0 or t_idx > n_draws - 1:
        return _uniform_prob40()
    history = draws[:t_idx]

    # Base sources
    bayes_t   = _norm_prob_dict_local(compute_bayes_posterior(history, alpha=1))
    markov_t  = _markov_prob_from_history(history)
    hmm_t     = _build_hmm_prob_from_subset(history)
    markov1_t = _markov1_prob_from_history(history)

    # Neural nets at time t: features from last k draws ending at t_idx-1
    feats_t = compute_stat_features(draws[t_idx - k: t_idx], t_idx - 1)
    feats_t = feats_t.reshape(1, feats_t.shape[0], feats_t.shape[1])
    _l_t, _t_t = _predict_with_nets(feats_t, t_idx=t_idx, use_finetune=True, mc_passes=MC_STACK_PASSES)

    # Derived helpers for stacking features
    nn_t = {n: float(0.5*_l_t[n-1] + 0.5*_t_t[n-1]) for n in range(1, 41)}
    last_draw_t = history[-1] if len(history) > 0 else set()
    cooc_t = _cooc_conditional_prob_from_history(history, last_draw_t)
    compat_k = 5
    compat_bayes_t  = _compat_topk_sum(bayes_t,  k=compat_k)
    compat_markov_t = _compat_topk_sum(markov_t, k=compat_k)
    compat_hmm_t    = _compat_topk_sum(hmm_t,    k=compat_k)
    compat_nn_t     = _compat_topk_sum(nn_t,     k=compat_k)
    reg_t = _regime_features_at_t(t_idx)
    roll_ent_t = _rolling_entropy_from_history(history, window=30)
    roll_disp_t = _dispersion_last_draw(history)

    # Build meta features at time t (mirror live latest_feats shape)
    latest_feats_t = np.array([
        [
            bayes_t.get(num, 0.0),
            markov_t.get(num, 0.0),
            hmm_t.get(num, 0.0),
            float(_l_t[num-1]),
            float(_t_t[num-1]),
            markov1_t.get(num, 0.0),
            cooc_t.get(num, 0.0),
            compat_bayes_t[num],
            compat_markov_t[num],
            compat_hmm_t[num],
            compat_nn_t[num],
            float(reg_t['weekday']),
            float(reg_t['month']),
            float(reg_t['gap_days']),
            float(reg_t['schedule_flip']),
            float(roll_ent_t),
            float(roll_disp_t),
        ]
        for num in range(1, 41)
    ], dtype=float)

    # Stacked score reconstruction
    meta_raw_t = meta_model.predict(latest_feats_t).reshape(1, 40)
    if 'postrank_calibrator' in globals() and postrank_calibrator is not None:
        cal_probs_t = postrank_calibrator.predict_proba(meta_raw_t)
        cal_probs_t = np.power(cal_probs_t, best_beta)
        cal_probs_t = cal_probs_t / (np.sum(cal_probs_t, axis=1, keepdims=True) + 1e-12)
        stacked_score_t = {num: float(cal_probs_t[0, num - 1]) for num in range(1, 41)}
    else:
        meta_probs_t = 1.0 / (1.0 + np.exp(-meta_raw_t))
        meta_probs_t = meta_probs_t / (np.sum(meta_probs_t, axis=1, keepdims=True) + 1e-12)
        meta_probs_t = np.power(meta_probs_t, best_beta)
        meta_probs_t = meta_probs_t / (np.sum(meta_probs_t, axis=1, keepdims=True) + 1e-12)
        stacked_score_t = {num: float(meta_probs_t[0, num - 1]) for num in range(1, 41)}

    # Dynamic-weighted base ensemble (use current global weights as a proxy)
    weighted_score_t = {}
    for num in range(1, 41):
        base_vec = np.array([
            bayes_t.get(num, 0.0),
            markov_t.get(num, 0.0),
            hmm_t.get(num, 0.0),
            float(_l_t[num-1]),
            float(_t_t[num-1])
        ])
        weighted_score_t[num] = float(np.dot(weights, base_vec))

    # Simple gap boost at time t (recomputed on history)
    last_seen = {n: -1 for n in range(1, 41)}
    for i, d in enumerate(history):
        for n in d:
            last_seen[n] = i
    gaps = {n: (len(history) - 1 - last_seen[n]) for n in range(1, 41)}
    mg = max(gaps.values()) if len(gaps) else 1
    gap_boost_t = {n: (gaps[n] / max(mg, 1)) for n in range(1, 41)}

    # Final combined score at time t (no weekday expert here)
    final_t = {}
    for num in range(1, 41):
        val = 0.7 * stacked_score_t[num] + 0.25 * weighted_score_t[num] + 0.03 * gap_boost_t.get(num, 0.0)
        final_t[num] = val
    # normalise
    s = sum(final_t.values())
    if s > 0:
        for n in range(1, 41):
            final_t[n] /= s
    return final_t


# Helper primitives needed by regime/weekday logic (must be defined early)
# ------------------------------------------------



# --- Adaptive ensemble weights using rolli<truncated__content/>
# --- Clean all _safe_fit_xgbranker and XGBRanker.fit call sites of fragile kwargs ---
# (No actual call sites shown in this snippet, but the patch applies globally.)
# === Final prediction & ticket printout ===
try:
    # Predict-time operational log (safe defaults if weights not learned yet)
    _weights = globals().get("weights", None)
    if _weights is None:
        import numpy as _np
        _weights = _np.ones(5, dtype=float) / 5.0
    try:
        _log_predict_run(_weights, t_eval=n_draws - 1, log_path="run_log.jsonl")
    except Exception as _e:
        import warnings as _w
        _w.warn(f"Predict-time operational log failed: {_e}")
    # --- Evaluation & diagnostics (walk-forward backtest, reliability, oracle ablation) ---
    try:
        _bt_summary = run_walk_forward_backtest(last_N=min(150, max(30, n_draws - (k + 5))))
        _oa_summary = run_oracle_ablation(last_N=min(100, max(30, n_draws - (k + 5))))
        print("[DIAG] Walk-forward summary:", _bt_summary)
        print("[DIAG] Oracle ablation (mean NLL improvements):", _oa_summary)
    except Exception as _e:
        _w.warn(f"Diagnostics failed: {_e}")
    # Decision policy (ensure symbol defined before calling)
    if 'recommend_tickets' in globals() and callable(recommend_tickets):
        _NT = int(os.environ.get("LOTTO_N_TICKETS", "10"))
        _tix = recommend_tickets(
            n_tickets=_NT, k=6, temperature=1.15, overlap_cap=3, coverage_penalty=0.65
        )
        print("[PICKS] Diversified PL-sampled tickets (temp=1.15):")
        for i, t in enumerate(_tix, 1):
            print(f"  Ticket {i:02d}: {t}")
        try:
            with open("predicted_tickets.txt", "w", encoding="utf-8") as _f:
                _f.write("\n".join(
                    ["Ticket %02d: %s" % (i, ", ".join(map(str, t))) for i, t in enumerate(_tix, 1)]
                ))
            print("[PICKS] Saved to predicted_tickets.txt")
        except Exception as _e:
            import warnings as _w
            _w.warn(f"Could not write predicted_tickets.txt: {_e}")
    else:
        import warnings as _w
        _w.warn("recommend_tickets() not defined yet when attempting to emit picks.")
except Exception as _e:
    import warnings as _w
    _w.warn(f"Final prediction block failed: {_e}")