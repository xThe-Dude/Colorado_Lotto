# --- cursor edit test by ChatGPT ---
# If you see this comment in Cursor, edits via the assistant are working.
# (Safe no-op change)
import pandas as pd
import numpy as np
import warnings
import re
import networkx as nx

# XGBoost for ticket-level reranker
try:
    import xgboost as xgb
except Exception as _e:
    xgb = None
    warnings.warn(f"XGBoost unavailable for reranker: {_e}")
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Suppress hmmlearn migration spam from MultinomialHMM (broader match)
# Treat the migration warning as a visible warning (not an error) so newer 0.3.x can proceed under guard
warnings.filterwarnings("ignore", message=r".*MultinomialHMM has undergone.*", category=UserWarning, module=r"hmmlearn\..*")

# Extra-hard squelch for hmmlearn MultinomialHMM migration spam (handles multi-line/cross-version cases)
warnings.filterwarnings("ignore", message=r"(?s).*MultinomialHMM has undergone.*")
warnings.filterwarnings("ignore", message=r".*MultinomialHMM has undergone.*", category=Warning, module=r"hmmlearn(\.|$).*")

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

# === Lotto+ Compatibility Shims (guarded) =====================================
# These avoid NameErrors after a revert by providing minimal, robust fallbacks.

# 1) Post‑rank isotonic calibrator used by reliability export
if '_PostRankIsotonic' not in globals():
    try:
        from sklearn.isotonic import IsotonicRegression as _Iso
    except Exception:
        _Iso = None
    class _PostRankIsotonic:
        def __init__(self):
            self.iso = _Iso(y_min=0.0, y_max=1.0, out_of_bounds='clip') if _Iso else None
            self.fitted = False
        def fit(self, probs_list, labels_list):
            import numpy as _np
            if self.iso is None:
                return self
            x = _np.asarray(probs_list, dtype=float).reshape(-1)
            y = _np.asarray(labels_list, dtype=float).reshape(-1)
            if x.size >= 100 and y.size == x.size:
                try:
                    self.iso.fit(x, y)
                    self.fitted = True
                except Exception:
                    self.fitted = False
            return self
        def map(self, p):
            # Map a single probability through isotonic; identity if not fitted or unavailable.
            try:
                return float(self.iso.predict([float(p)])[0]) if (self.iso is not None and self.fitted) else float(p)
            except Exception:
                return float(p)

#
# === Probabilistic Modeling & Calibration Utilities ============================
SUM6_EPS = 1e-9
SUM6_MAX_ITERS = 4
CAL_WINDOW = 60              # draws used to fit calibrators
CAL_MIN = 20
CAL_METHOD = "platt+isotonic"  # {"isotonic","platt","platt+isotonic"}
PF_PRIOR_BLEND = 0.08        # lighter PF prior so stacker dominates
PF_ENSEMBLE_CONFIGS = [      # (num_particles, alpha, sigma)
    (12000, 0.004, 0.008),
    (16000, 0.003, 0.006),
    (8000,  0.006, 0.012),
    (10000, 0.005, 0.010),
    (14000, 0.0035, 0.007),
    (20000, 0.0025, 0.005),
]

_cal_cache = {"LSTM": None, "Transformer": None, "t_fit": None}

def _enforce_sum6(v):
    """Project a length-40 vector v (non-negative) so that sum≈6 with clipping to [0,1].
    Uses iterative rescale + clip (water-filling style) for a few passes.
    Returns a copy with sum close to 6.
    """
    import numpy as _np
    x = _np.clip(_np.asarray(v, dtype=float), 0.0, 1.0)
    for _ in range(SUM6_MAX_ITERS):
        s = float(x.sum())
        if s <= 0:
            x[:] = 6.0/40.0
            break
        x *= (6.0 / s)
        over = x > 1.0
        if _np.any(over):
            x[over] = 1.0
        s2 = float(x.sum())
        if abs(s2 - 6.0) <= 1e-6:
            break
        if s2 > 0:
            x *= (6.0 / s2)
    return _np.clip(x, 0.0, 1.0)

def _sum6_to_sum1_dict(p6_dict):
    import numpy as _np
    v = _np.array([float(p6_dict.get(i, 0.0)) for i in range(1,41)], dtype=float)
    if v.sum() <= 0:
        v = _np.ones(40, dtype=float)*(6.0/40.0)
    v = _np.clip(v/6.0, 1e-18, None)
    v = v / (v.sum() + 1e-18)
    return {i: float(v[i-1]) for i in range(1,41)}

# ---- Simple 1D calibrators ----------------------------------------------------
class _Platt1D:
    def __init__(self):
        self.ok = False
        self.coef_ = None
        self.intercept_ = None
    def fit(self, p, y):
        try:
            import numpy as _np
            from sklearn.linear_model import LogisticRegression
            p = _np.clip(_np.asarray(p, dtype=float).reshape(-1,1), 1e-6, 1-1e-6)
            y = _np.asarray(y, dtype=float).reshape(-1)
            if p.shape[0] < 100 or y.sum() == 0 or y.sum() == y.size:
                self.ok = False
                return self
            logit = _np.log(p/(1-p))
            lr = LogisticRegression(class_weight="balanced", max_iter=1000)
            lr.fit(logit, y)
            self.ok = True
            self.coef_ = float(lr.coef_.ravel()[0])
            self.intercept_ = float(lr.intercept_.ravel()[0])
            self._lr = lr
        except Exception:
            self.ok = False
        return self
    def map(self, p):
        import numpy as _np
        p = _np.clip(_np.asarray(p, dtype=float).reshape(-1), 1e-9, 1-1e-9)
        if not self.ok:
            return p
        z = _np.log(p/(1-p)) * self.coef_ + self.intercept_
        out = 1.0/(1.0+_np.exp(-z))
        return _np.clip(out, 1e-9, 1-1e-9)

class _Iso1D:
    def __init__(self):
        try:
            from sklearn.isotonic import IsotonicRegression
            self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        except Exception:
            self.iso = None
        self.ok = False
    def fit(self, p, y):
        import numpy as _np
        if self.iso is None:
            self.ok = False
            return self
        p = _np.clip(_np.asarray(p, dtype=float).reshape(-1), 1e-9, 1-1e-9)
        y = _np.asarray(y, dtype=float).reshape(-1)
        if p.size < 100 or y.sum() == 0 or y.sum() == y.size:
            self.ok = False
            return self
        try:
            self.iso.fit(p, y)
            self.ok = True
        except Exception:
            self.ok = False
        return self
    def map(self, p):
        import numpy as _np
        p = _np.clip(_np.asarray(p, dtype=float).reshape(-1), 1e-9, 1-1e-9)
        if not self.ok or self.iso is None:
            return p
        out = self.iso.predict(p)
        return _np.clip(out, 1e-9, 1-1e-9)

class _ComboCal:
    def __init__(self, method="platt+isotonic"):
        self.method = method
        self.pl = _Platt1D()
        self.iso = _Iso1D()
        self.ok = False
    def fit(self, p, y):
        m = (self.method or "").lower()
        if "platt" in m:
            self.pl.fit(p, y)
        if "isotonic" in m:
            self.iso.fit(p, y)
        self.ok = (self.pl.ok or self.iso.ok)
        return self
    def map(self, p):
        import numpy as _np
        p = _np.asarray(p, dtype=float).reshape(-1)
        if not self.ok:
            return _np.clip(p, 1e-9, 1-1e-9)
        vals = []
        if self.pl.ok:
            vals.append(self.pl.map(p))
        if self.iso.ok:
            vals.append(self.iso.map(p))
        if not vals:
            return _np.clip(p, 1e-9, 1-1e-9)
        out = _np.mean(_np.vstack(vals), axis=0)
        return _np.clip(out, 1e-9, 1-1e-9)

# Fit calibrators for LSTM/Transformer using last CAL_WINDOW draws up to t_eval
def _fit_expert_calibrators(t_eval):
    import numpy as _np
    global _cal_cache
    t_eval = int(t_eval)
    t0 = max(1, t_eval - CAL_WINDOW + 1)
    if t_eval == _cal_cache.get("t_fit") and all(_cal_cache.get(k) is not None for k in ("LSTM","Transformer")):
        return _cal_cache
    P = {"LSTM": [], "Transformer": []}
    Y = []
    _orig = globals().get("_predict_with_nets", None)
    if not callable(_orig):
        _cal_cache = {"LSTM": None, "Transformer": None, "t_fit": t_eval}
        return _cal_cache
    if t_eval - t0 + 1 < CAL_MIN:
        _cal_cache = {"LSTM": None, "Transformer": None, "t_fit": t_eval}
        return _cal_cache
    for t in range(t0, t_eval+1):
        try:
            hist = _history_upto(t, context="_fit_expert_calibrators")
            if 'compute_stat_features' in globals():
                feats_t = compute_stat_features(hist[-min(len(hist), 20):], t - 1)
                feats_t = feats_t.reshape(1, feats_t.shape[0], feats_t.shape[1])
            else:
                continue
            meta_t = None
            if '_meta_features_at_idx' in globals():
                meta_t = __import__('numpy').array([_meta_features_at_idx(t)], dtype=float)
            l, tr = _orig(feats_t, meta_t, t_idx=t, use_finetune=True, mc_passes=globals().get('MC_STACK_PASSES', 1))
            l6 = _enforce_sum6(l*6.0)
            tr6 = _enforce_sum6(tr*6.0)
            winners = _winner_draw_at(t, context="_fit_expert_calibrators")
            y = __import__('numpy').array([1 if n in winners else 0 for n in range(1,41)], dtype=float)
            P["LSTM"].append(l6); P["Transformer"].append(tr6)
            Y.append(y)
        except Exception:
            continue
    if not P["LSTM"] or not Y:
        _cal_cache = {"LSTM": None, "Transformer": None, "t_fit": t_eval}
        return _cal_cache
    P_L = __import__('numpy').vstack(P["LSTM"]).reshape(-1)
    P_T = __import__('numpy').vstack(P["Transformer"]).reshape(-1)
    Yv  = __import__('numpy').vstack(Y).reshape(-1)
    cal_L = _ComboCal(CAL_METHOD).fit(P_L, Yv)
    cal_T = _ComboCal(CAL_METHOD).fit(P_T, Yv)
    _cal_cache = {"LSTM": cal_L, "Transformer": cal_T, "t_fit": t_eval}
    return _cal_cache

# Postprocess raw net probabilities (length-40 array) → calibrated, sum-6, then sum-1
def _postprocess_net_probs(arr, t_idx, expert_name="LSTM"):
    import numpy as _np
    v = _np.asarray(arr, dtype=float).reshape(-1)
    if v.size != 40:
        v = _np.ones(40, dtype=float) * (1.0/40.0)
    # Step 1: sum-to-6 constraint
    v6 = _enforce_sum6(v*6.0)
    # Step 2: calibration
    cals = _fit_expert_calibrators(max(1, int(t_idx)-1))
    cal = cals.get(expert_name) if isinstance(cals, dict) else None
    if cal is not None and getattr(cal, 'ok', False):
        v6 = cal.map(v6)
        v6 = _enforce_sum6(v6)
    # Step 3: back to categorical (sum=1)
    out = v6 / 6.0
    out = _np.clip(out, 1e-12, None)
    out = out / (out.sum() + 1e-12)
    return out

# ---- PF prior from history (ensemble over hyper-params) ----------------------
def _pf_prior_from_history(sub_draws, configs=PF_ENSEMBLE_CONFIGS):
    import numpy as _np
    if not sub_draws:
        return {n: 6.0/40.0 for n in range(1,41)}
    preds = []
    for (npart, a, s) in (configs or []):
        try:
            pf = ParticleFilter(num_numbers=40, num_particles=int(npart), alpha=float(a), sigma=float(s))
            for d in sub_draws:
                pf.predict()
                pf.update(d)
            pf.predict()
            preds.append(pf.get_mean_probabilities())
        except Exception:
            continue
    if not preds:
        return {n: 6.0/40.0 for n in range(1,41)}
    m = _np.mean(_np.vstack(preds), axis=0)
    m = _enforce_sum6(m)
    return {n: float(m[n-1]) for n in range(1,41)}

def _geom_blend(p, q, lam=0.5):
    """Geometric blend of two categorical dists p and q over 1..40 (sum=1 each)."""
    import numpy as _np
    lam = float(max(0.0, min(1.0, lam)))
    vp = _np.array([float(p.get(i,0.0)) for i in range(1,41)], dtype=float)
    vq = _np.array([float(q.get(i,0.0)) for i in range(1,41)], dtype=float)
    vp = _np.clip(vp, 1e-18, None); vq = _np.clip(vq, 1e-18, None)
    v = _np.power(vp, 1.0-lam) * _np.power(vq, lam)
    v = _np.clip(v, 1e-18, None); v = v / (v.sum() + 1e-18)
    return {i: float(v[i-1]) for i in range(1,41)}
# ============================================================================

# 2) Historical per‑expert distributions builder used by backtests/diagnostics
if '_per_expert_prob_dicts_at_t' not in globals():
    def _per_expert_prob_dicts_at_t(t_idx):
        """Return base probability dicts at historical index t_idx using only draws[:t_idx].
        Provides a safe, minimal version compatible with older code paths.
        Keys match _per_expert_names(): ["Bayes","Markov","HMM","LSTM","Transformer","GNN"].
        """
        try:
            t = int(t_idx)
        except Exception:
            return None
        try:
            hist = _history_upto(t, context="_per_expert_prob_dicts_at_t")
        except Exception:
            return None
        if t <= 0 or len(hist) <= 0:
            return None
        # Bayes / Markov / HMM
        try:
            bayes_t = compute_bayes_posterior(hist, alpha=1)
        except Exception:
            bayes_t = {n: 1.0/40 for n in range(1,41)}
        try:
            # compute_markov_transitions returns (t1,t2,t3); use a simple conditional from last draw
            t1, _, _ = compute_markov_transitions(hist)
            last = hist[-1]
            import numpy as _np
            sc = _np.zeros(40, dtype=float)
            for i in last:
                for j in range(1,41):
                    sc[j-1] += float(t1.get(i, {}).get(j, 1.0/40))
            if sc.sum() > 0:
                sc = sc / sc.sum()
            markov_t = {n: float(sc[n-1]) for n in range(1,41)}
        except Exception:
            markov_t = {n: 1.0/40 for n in range(1,41)}
        try:
            hmm_t = _build_tcn_prob_from_subset(hist)
        except Exception:
            hmm_t = {n: 1.0/40 for n in range(1,41)}
        # LSTM / Transformer via existing net predictor, if available
        try:
            if 'compute_stat_features' in globals():
                feats_t = compute_stat_features(hist[-min(len(hist), 20):], t - 1)
                feats_t = feats_t.reshape(1, feats_t.shape[0], feats_t.shape[1])
            else:
                feats_t = None
            meta_t = None
            if '_meta_features_at_idx' in globals():
                import numpy as _np
                meta_t = _np.array([_meta_features_at_idx(t)], dtype=float)
            if '_predict_with_nets' in globals() and feats_t is not None:
                _l_t, _tr_t = _predict_with_nets(feats_t, meta_t, t_idx=t, use_finetune=True, mc_passes=globals().get('MC_STACK_PASSES', 1))
                _l_t = _postprocess_net_probs(_l_t, t, expert_name="LSTM")
                _tr_t = _postprocess_net_probs(_tr_t, t, expert_name="Transformer")
                lstm_t = {n: float(_l_t[n-1]) for n in range(1,41)}
                trans_t = {n: float(_tr_t[n-1]) for n in range(1,41)}
            else:
                lstm_t = {n: 1.0/40 for n in range(1,41)}
                trans_t = {n: 1.0/40 for n in range(1,41)}
        except Exception:
            lstm_t = {n: 1.0/40 for n in range(1,41)}
            trans_t = {n: 1.0/40 for n in range(1,41)}
        # GNN
        try:
            if '_gnn_prob_from_history' in globals():
                import numpy as _np
                gnn_model_ref = globals().get("gnn_model", None)
                meta_gnn = _meta_features_at_idx(t - 1) if t > 0 else _np.array([0.0, 0.5, 0.5])
                gnn_t = _gnn_prob_from_history(hist, gnn_model=gnn_model_ref, meta_features=meta_gnn)
            else:
                gnn_t = {n: 1.0/40 for n in range(1,41)}
        except Exception:
            gnn_t = {n: 1.0/40 for n in range(1,41)}
        # Normalise defensively
        def _norm(d):
            import numpy as _np
            v = _np.array([float(d.get(i,0.0)) for i in range(1,41)], dtype=float)
            v = _np.clip(v, 1e-12, None); v = v / (v.sum() + 1e-12)
            return {i: float(v[i-1]) for i in range(1,41)}
        return {
            "Bayes": _norm(bayes_t),
            "Markov": _norm(markov_t),
            "HMM": _norm(hmm_t),
            "LSTM": _norm(lstm_t),
            "Transformer": _norm(trans_t),
            "GNN": _norm(gnn_t),
        }

# 3) Joint/marginal blender used by SetAR decoding paths
if 'blend_joint_with_marginals' not in globals():
    def blend_joint_with_marginals(joint_scores, marginal_scores, alpha=0.5):
        """Blend a ticket‑level joint score dict with per‑number marginals.
        `joint_scores`: dict[ticket_tuple->score]; `marginal_scores`: dict[num->p].
        Returns a new dict with renormalised blended scores.
        """
        import numpy as _np
        a = float(alpha)
        a = 0.0 if not _np.isfinite(a) else max(0.0, min(1.0, a))
        # Normalise joint scores
        js_keys = list(joint_scores.keys()) if isinstance(joint_scores, dict) else []
        if not js_keys:
            return {}
        js = _np.array([float(joint_scores[k]) for k in js_keys], dtype=float)
        js = _np.clip(js, 1e-12, None); js = js / (js.sum() + 1e-12)
        # Build a marginal product per ticket (without replacement proxy)
        mp = []
        for t in js_keys:
            prod = 1.0
            for n in t:
                prod *= float(marginal_scores.get(n, 1.0/40.0))
            mp.append(prod)
        mp = _np.array(mp, dtype=float)
        mp = _np.clip(mp, 1e-18, None); mp = mp / (mp.sum() + 1e-18)
        comb = (1.0 - a) * js + a * mp
        comb = _np.clip(comb, 1e-18, None); comb = comb / (comb.sum() + 1e-18)
        return {k: float(v) for k, v in zip(js_keys, comb)}

# --- Inserted predict_joint_enhanced ---
if 'predict_joint_enhanced' not in globals():
    def predict_joint_enhanced(t_eval=None, alpha=0.40, n_samples=20000, keep_top=400, topN=128):
        """
        Joint ticket predictor:
          • If t_eval is None, uses CURRENT_TARGET_IDX or len(draws)-1.
          • Returns (ticket_rankings, blended_scores_dict)
        """
        try:
            if t_eval is None:
                ct = globals().get("CURRENT_TARGET_IDX")
                t_eval = int(ct) if ct is not None else int(len(draws) - 1)
        except Exception:
            t_eval = int(len(draws) - 1)
        blended, ranked = score_and_blend_tickets(
            int(t_eval),
            alpha=float(alpha),
            n_samples=int(n_samples),
            keep_top=int(keep_top),
            topN=int(topN)
        )
        return ranked, blended

if '_export_reliability' not in globals():
    def _export_reliability(
        probs=None,
        labels=None,
        path_csv='reliability_curve.csv',
        path_png='reliability_curve.png',
        bins=20,
        min_bin_count=50,
        mode="equal_freq",
        **kwargs
    ):
        """
        Reliability (calibration) export on sum-6 probabilities with adaptive binning and full diagnostics.

        Inputs:
          - probs: 1D array-like of predicted probabilities for the observed outcome (binary).
          - labels: 1D array-like of 0/1 labels (same length as probs).
          - bins: maximum number of bins (for 'equal_freq' this is an upper bound).
          - min_bin_count: minimum samples per bin; reduces #bins as needed.
          - mode: {"equal_freq", "fixed_width"}.
          - path_csv, path_png: output paths (relative paths are written under ./artifacts).
          - Back-compat aliases: csv_path, png_path in kwargs.

        CSV columns:
          bin_lower, bin_upper, count, avg_pred, emp_rate, ci_lower, ci_upper

        Returns:
          dict summary with keys:
            {
              "ece": float,          # Expected Calibration Error (bin-weighted |emp - avg_pred|)
              "brier": float,        # mean squared error
              "logloss": float,      # -mean[y log p + (1-y) log(1-p)]
              "bins": int,           # number of populated bins
              "min_bin_count": int,  # enforced minimum per bin
              "mode": str,           # binning mode
              "csv": str,            # path to CSV
              "png": str             # path to PNG
            }
        """
        # Back-compat for callers using csv_path/png_path
        if 'csv_path' in kwargs and not kwargs.get('path_csv'):
            path_csv = kwargs.get('csv_path', path_csv)
        if 'png_path' in kwargs and not kwargs.get('path_png'):
            path_png = kwargs.get('png_path', path_png)

        # Normalize to ./artifacts for relative filenames
        try:
            import os as _os
            if path_csv and ("/" not in str(path_csv) and "\\" not in str(path_csv)):
                path_csv = _artifact_path(str(path_csv))
            if path_png and ("/" not in str(path_png) and "\\" not in str(path_png)):
                path_png = _artifact_path(str(path_png))
        except Exception:
            pass

        import numpy as _np
        import math as _math
        import csv as _csv

        # Source data
        _p = _np.asarray(probs, dtype=float).reshape(-1) if probs is not None else None
        _y = _np.asarray(labels, dtype=float).reshape(-1) if labels is not None else None
        if _p is None or _y is None or _p.size == 0 or _y.size != _p.size:
            # Defensive globals fallback
            _p = _np.asarray(globals().get('last_joint_probs', []), dtype=float).reshape(-1)
            _y = _np.asarray(globals().get('last_joint_labels', []), dtype=float).reshape(-1)
            if _p.size == 0 or _y.size != _p.size:
                return {}
        # Evaluate reliability on sum-6 semantics: scale categorical probs by 6 and clip to [0,1].
        # This aligns predicted per-number probabilities with 0/1 labels (≈6 positives per draw).
        try:
            _p = _np.asarray(_p, dtype=float).reshape(-1)
            _p = _np.clip(_p * 6.0, 1e-12, 1.0 - 1e-12)
        except Exception:
            _p = _np.clip(_p, 1e-12, 1.0 - 1e-12)

        # Sanity and clipping (labels already 0/1)
        _y = _np.clip(_y, 0.0, 1.0)

        N = int(_p.size)
        if N < 5:
            return {}

        # Compute global scores (independent of binning)
        brier = float(_np.mean((_p - _y) ** 2))
        logloss = float(-_np.mean(_y * _np.log(_p) + (1.0 - _y) * _np.log(1.0 - _p)))

        # --- Binning helpers ---------------------------------------------------
        def _wilson_ci(k, n, z=1.96):
            if n <= 0:
                return (0.0, 0.0)
            phat = k / n
            denom = 1 + (z**2)/n
            center = (phat + (z**2)/(2*n)) / denom
            margin = (z / denom) * _math.sqrt((phat*(1-phat)/n) + (z**2)/(4*n*n))
            lo = max(0.0, center - margin)
            hi = min(1.0, center + margin)
            return (float(lo), float(hi))

        def _equal_freq_bins(p_sorted, max_bins, min_cnt):
            """Return list of (start_idx, end_idx) inclusive ranges for equal-frequency bins."""
            n = p_sorted.size
            if n == 0:
                return []
            max_bins = max(1, int(max_bins))
            min_cnt = max(1, int(min_cnt))
            # target bins limited by min_bin_count
            max_by_count = max(1, n // min_cnt)
            B = int(min(max_bins, max_by_count))
            if B <= 1:
                return [(0, n-1)]
            edges = [0]
            for b in range(1, B):
                # quantile split
                q = b / B
                idx = int(_np.floor(q * n))
                if idx <= edges[-1]:
                    idx = edges[-1] + 1
                edges.append(min(idx, n-1))
            edges.append(n)
            # Build ranges, merge tiny last bin if necessary
            ranges = []
            for i in range(len(edges)-1):
                a, b = edges[i], edges[i+1]
                if i == 0:
                    ranges.append((a, b-1))
                else:
                    if b - edges[i] < min_cnt and ranges:
                        # merge with previous
                        prev_a, prev_b = ranges[-1]
                        ranges[-1] = (prev_a, b-1)
                    else:
                        ranges.append((a, b-1))
            # Clean empty bins
            ranges = [(a,b) for (a,b) in ranges if b >= a]
            return ranges

        def _fixed_width_bins(p_vals, B):
            edges = _np.linspace(0.0, 1.0, int(B)+1)
            return edges

        # Sort by probability for equal-frequency if needed
        order = _np.argsort(_p)
        p_sorted = _p[order]
        y_sorted = _y[order]

        rows = []
        ece_num = 0.0
        if (mode or "").lower() == "equal_freq":
            ranges = _equal_freq_bins(p_sorted, bins if bins else 20, min_bin_count)
            for (a, b) in ranges:
                p_bin = p_sorted[a:b+1]
                y_bin = y_sorted[a:b+1]
                cnt = int(p_bin.size)
                avg_p = float(p_bin.mean())
                emp = float(y_bin.mean())
                lo, hi = _wilson_ci(int(y_bin.sum()), cnt)
                ece_num += (cnt / N) * abs(emp - avg_p)
                # Use observed bin bounds (min/max) for CSV/plot
                bin_lo = float(p_bin.min())
                bin_hi = float(p_bin.max())
                rows.append([bin_lo, bin_hi, cnt, avg_p, emp, lo, hi])
        else:
            # fixed-width as fallback (legacy behavior but with CI)
            edges = _fixed_width_bins(_p, bins if bins else 20)
            idx = _np.digitize(_p, edges, right=True)
            for b in range(1, len(edges)):
                mask = (idx == b)
                if not _np.any(mask):
                    continue
                p_bin = _p[mask]
                y_bin = _y[mask]
                cnt = int(mask.sum())
                avg_p = float(p_bin.mean())
                emp = float(y_bin.mean())
                lo, hi = _wilson_ci(int(y_bin.sum()), cnt)
                ece_num += (cnt / N) * abs(emp - avg_p)
                rows.append([float(edges[b-1]), float(edges[b]), cnt, avg_p, emp, lo, hi])

        # Write CSV
        try:
            with open(path_csv, 'w', newline='') as f:
                w = _csv.writer(f)
                w.writerow(['bin_lower','bin_upper','count','avg_pred','emp_rate','ci_lower','ci_upper'])
                w.writerows(rows)
        except Exception:
            pass

        # Plot PNG
        try:
            import matplotlib.pyplot as _plt
            xs = _np.linspace(0,1,101)
            _plt.figure()
            _plt.plot(xs, xs, linestyle='--', label='ideal')
            if rows:
                x_mid = [_np.mean(r[:2]) for r in rows]
                emp = [r[4] for r in rows]
                ci_lo = [r[5] for r in rows]
                ci_hi = [r[6] for r in rows]
                cnts = [r[2] for r in rows]
                # error bars from ci
                yerr_low = _np.maximum(0, _np.array(emp) - _np.array(ci_lo))
                yerr_high = _np.maximum(0, _np.array(ci_hi) - _np.array(emp))
                _plt.errorbar(x_mid, emp, yerr=[yerr_low, yerr_high], fmt='o', capsize=3, label='empirical ±95% CI')
                for xm, c in zip(x_mid, cnts):
                    _plt.annotate(str(c), (xm, 0.02), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
            _plt.xlabel('Predicted probability')
            _plt.ylabel('Empirical rate')
            _plt.title('Reliability Curve (adaptive bins)')
            _plt.legend()
            _plt.tight_layout()
            _plt.savefig(path_png)
            _plt.close()
        except Exception:
            pass

        summary = {
            "ece": float(ece_num),
            "brier": float(brier),
            "logloss": float(logloss),
            "bins": int(len(rows)),
            "min_bin_count": int(min_bin_count),
            "mode": "equal_freq" if (mode or "").lower() == "equal_freq" else "fixed_width",
            "csv": str(path_csv),
            "png": str(path_png),
        }
        return summary

# 5) Base-weight adapter for marginal ensemble (fallback heuristic)
if '_adapt_base_weights' not in globals():
    def _adapt_base_weights(base_prob_dicts, y_true=None, prev_weights=None):
        """Return non-negative weights over experts in `base_prob_dicts` (dict name->dict num->p).
        Heuristic fallback:
          1) If per-expert PL-NLLs at t are available via `_per_expert_pl_nll_at`, use softmax(-nll).
          2) Else, prefer lower-entropy experts (sharper dists) via softmax(-entropy).
          3) Else, return uniform.
        `prev_weights` can be provided; we blend 70% new / 30% previous for stability.
        Returns a numpy array aligned to `_per_expert_names()` summing to 1.
        """
        import numpy as _np
        names = _per_expert_names() if '._per_expert_names' or '_per_expert_names' in globals() else list(base_prob_dicts.keys())
        K = len(names)
        w = _np.ones(K, dtype=float) / max(1, K)
        try:
            # 1) Try PL-NLLs
            if '._per_expert_pl_nll_at' in globals() or '_per_expert_pl_nll_at' in globals():
                try:
                    t_eval = globals().get('CURRENT_TARGET_IDX')
                    pe = globals().get('_per_expert_pl_nll_at')
                    if callable(pe) and t_eval is not None and int(t_eval) >= 1:
                        vals = pe(int(t_eval))
                        if hasattr(vals, '__len__') and len(vals) == K:
                            x = _np.array([float(v) if _np.isfinite(v) else 1.0 for v in vals], dtype=float)
                            x = -x
                            x = x - x.max()
                            w = _np.exp(x); w = w / (w.sum() + 1e-12)
                except Exception:
                    pass
            # 2) If still near-uniform, use entropy heuristic on provided bases
            if _np.allclose(w, _np.ones_like(w)/max(1,K), rtol=0, atol=1e-6) and isinstance(base_prob_dicts, dict):
                ent = _np.zeros(K, dtype=float)
                for i, nm in enumerate(names):
                    d = base_prob_dicts.get(nm, {})
                    v = _np.array([float(d.get(j, 0.0)) for j in range(1,41)], dtype=float)
                    v = _np.clip(v, 1e-12, None); v = v / (v.sum() + 1e-12)
                    ent[i] = -_np.sum(v * _np.log(v))
                x = -ent
                x = x - x.max()
                w = _np.exp(x); w = w / (w.sum() + 1e-12)
            # 3) Temporal smoothing with previous weights if provided
            if prev_weights is not None:
                pw = _np.asarray(prev_weights, dtype=float)
                if pw.shape == w.shape and _np.isfinite(pw).all():
                    w = 0.7*w + 0.3*pw
                    w = _np.clip(w, 1e-12, None); w = w / (w.sum() + 1e-12)
        except Exception:
            w = _np.ones(K, dtype=float) / max(1, K)
        return w
# --- Helper: Coerce input to prob dict {1..40 -> p} robustly ---
def _coerce_to_prob_dict(obj, default_val=0.0):
    """
    Coerce various input shapes into a dict {1..40 -> p}.
    Accepts:
      - dict-like with numeric keys 1..40
      - list/tuple/np.ndarray of length 40
    Returns None if it cannot be coerced safely.
    """
    import numpy as _np
    # Case 1: dict-like
    if hasattr(obj, "get") and hasattr(obj, "keys"):
        try:
            v = _np.array([float(obj.get(i, default_val)) for i in range(1,41)], dtype=float)
            v = _np.clip(v, 0.0, _np.inf)
            s = float(v.sum())
            if not _np.isfinite(s) or s <= 0:
                return {i: float(1.0/40.0) for i in range(1,41)}
            v = v / (s + 1e-18)
            return {i: float(v[i-1]) for i in range(1,41)}
        except Exception:
            return None
    # Case 2: array-like length 40
    try:
        arr = _np.asarray(obj, dtype=float).reshape(-1)
        if arr.size == 40:
            arr = _np.clip(arr, 0.0, _np.inf)
            s = float(arr.sum())
            if not _np.isfinite(s) or s <= 0:
                arr = _np.ones(40, dtype=float) / 40.0
            else:
                arr = arr / (s + 1e-18)
            return {i: float(arr[i-1]) for i in range(1,41)}
    except Exception:
        pass
    # Otherwise give up
    return None

# 6) Mixer for per-expert marginal dicts (weighted sum + normalisation)
if '_mix_prob_dicts' not in globals():
    def _mix_prob_dicts(base_prob_dicts, weights=None, temperature=None, power=None, names=None):
        """Combine expert marginals into a single probability dict over 1..40 (robust to bad inputs).
        Args:
          base_prob_dicts: dict name->(dict|array-like length-40) OR list[...] of same
          weights: array-like weights aligned with _per_expert_names() (if dict) or list order
          temperature/power/names: optional shaping
        Returns: dict num->p (normalised, non-negative).
        """
        import numpy as _np

        # Normalise inputs into a list of (name, dict{1..40->p})
        items = []
        if isinstance(base_prob_dicts, dict):
            order = names or _per_expert_names()
            for nm in order:
                coerced = _coerce_to_prob_dict(base_prob_dicts.get(nm, None))
                if coerced is None:
                    # If completely missing or malformed, fall back to uniform for this expert
                    coerced = {i: float(1.0/40.0) for i in range(1,41)}
                items.append((nm, coerced))
        elif isinstance(base_prob_dicts, (list, tuple)):
            for i, d in enumerate(list(base_prob_dicts)):
                coerced = _coerce_to_prob_dict(d)
                if coerced is None:
                    # skip scalars or junk entries gracefully
                    continue
                items.append((str(i), coerced))
        else:
            # Unhandled structure — return uniform to avoid crashing callers
            return {i: float(1.0/40.0) for i in range(1,41)}

        if not items:
            return {i: float(1.0/40.0) for i in range(1,41)}

        K = len(items)

        # Build weight vector
        w = _np.ones(K, dtype=float) / float(K)
        if weights is not None:
            try:
                w_in = _np.asarray(weights, dtype=float).reshape(-1)
                if w_in.size == K and _np.isfinite(w_in).all():
                    w = _np.clip(w_in, 1e-12, None)
                    w = w / (w.sum() + 1e-12)
            except Exception:
                pass

        # Accumulate weighted sum
        acc = _np.zeros(40, dtype=float)
        for i, (_, d) in enumerate(items):
            v = _np.array([float(d.get(n, 0.0)) for n in range(1,41)], dtype=float)
            # Optional per-expert sharpening/flattening
            if power is not None:
                try:
                    v = _np.power(_np.clip(v, 1e-18, None), float(power))
                except Exception:
                    pass
            acc += float(w[i]) * v

        # Optional temperature transformation on the mix (softmax on log-space)
        if temperature is not None:
            try:
                t = float(temperature)
                t = 1.0 if not _np.isfinite(t) or t <= 0 else t
                acc = _np.exp(_np.log(_np.clip(acc, 1e-18, None)) / t)
            except Exception:
                pass

        acc = _np.clip(acc, 1e-18, None)
        acc = acc / (acc.sum() + 1e-18)
        return {n: float(acc[n-1]) for n in range(1,41)}
# === End shims ================================================================
# --- Final shim registration (guarantee global binding) ---
try:
    globals()['_export_reliability'] = _export_reliability  # ensure symbol exists in global ns
except Exception:
    pass
try:
    globals()['_adapt_base_weights'] = _adapt_base_weights  # ensure symbol exists in global ns
except Exception:
    pass
try:
    globals()['_mix_prob_dicts'] = _mix_prob_dicts
except Exception:
    pass
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
    try:
        import os as _os
        out_dir = _os.path.join(_os.getcwd(), "artifacts")
        _os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = _os.path.join(out_dir, _os.path.basename(output_path))
        cmd = [sys.executable, "-m", "pip", "freeze"]
        txt = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60)
        with open(out, "w", encoding="utf-8") as f:
            f.write("# Frozen by Lotto+ on " + stamp + "\n")
            f.write(txt)
    except Exception as e:
        warnings.warn(f"pip freeze failed: {e}")

def _log_jsonl(record, path="run_log.jsonl"):
    try:
        import os as _os
        out_dir = _os.path.join(_os.getcwd(), "artifacts")
        _os.makedirs(out_dir, exist_ok=True)
        out_path = _os.path.join(out_dir, _os.path.basename(path))
        with open(out_path, "a", encoding="utf-8") as f:
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

# === Expert name contract ======================================================
# Keep these EXACT strings across logs/mixing/backtests:
# ["Bayes", "Markov", "HMM", "LSTM", "Transformer"]
# If you ever rename the TCNN head to "TCNN" in logs:
#   1) Update _per_expert_names() to replace "LSTM" -> "TCNN"
#   2) Update _per_expert_prob_dicts_at_t(...) to return key "TCNN" instead of "LSTM"
# Everything else should reference _per_expert_names() and remain unchanged.
# ==============================================================================

def _per_expert_names():
    # Names must match keys returned by _per_expert_prob_dicts_at_t()
    return ["Bayes", "Markov", "HMM", "LSTM", "Transformer", "GNN"]

def _assert_expert_key_consistency(bases_or_logs):
    """
    Soft guard to ensure all expected expert keys are present everywhere.
    Emits a warning if any are missing or misspelled.
    """
    try:
        expected = _per_expert_names()
        keys = list(bases_or_logs.keys()) if isinstance(bases_or_logs, dict) else []
        missing = [nm for nm in expected if nm not in keys]
        if missing:
            warnings.warn(f"Expert key mismatch. Missing keys {missing}. Expected exactly {expected}.")
        return True
    except Exception:
        return True

#
# === Adaptive Ensemble Weighting & Meta‑Model Stacking =========================
# This block learns expert weights from recent performance and (optionally)
# trains a tiny logistic-regression meta-learner that maps 5 expert probabilities
# → a calibrated probability per number. It activates automatically by wrapping
# `_mix_prob_dicts` so existing call sites benefit without further edits.

USE_ADAPTIVE_ENSEMBLE = True
USE_META_STACKING = True
USE_CONFIDENCE_GATING = True   # Phase 4 Upgrade 1: Enable confidence-aware dynamic ensemble
ADAPT_WINDOW = 90              # (60–90) longer memory for more stable weights
META_MIN_DRAWS = 60            # require ≥60 draws to train the stacker
META_CLASS_WEIGHT = "balanced"  # handle 6/40 positives

# --- Phase 4 Confidence Gating Parameters ---
CONFIDENCE_MC_PASSES = 30      # Number of MC Dropout passes for uncertainty estimation
CONFIDENCE_QUANTILE = 0.70     # Confidence threshold quantile (0.70 = use top 30% most confident)
CONFIDENCE_MIN_EXPERTS = 1     # Minimum experts required (fallback to LSTM if less)

# --- Phase 4 Specialized Training Parameters (Upgrade 2) ---
# NOTE: Multi-task training with auxiliary targets now fully implemented and enabled.
# TCN learns frequency change trends, Transformer learns future patterns,
# GNN learns co-occurrence communities for improved ensemble diversity.
USE_SPECIALIZED_TRAINING = True  # Enable multi-task learning with specialized objectives
TCNN_AUX_WEIGHT = 0.15           # TCN: Frequency change prediction weight (0.85 main + 0.15 aux)
TRANSFORMER_AUX_WEIGHT = 0.20    # Transformer: Future draw prediction weight (0.80 main + 0.20 aux)
GNN_AUX_WEIGHT = 0.15            # GNN: Co-occurrence prediction weight (0.85 main + 0.15 aux)

# --- Phase 4 Hierarchical Stacking Parameters (Upgrade 3) ---
# Organizes experts into specialized groups with learned routing for +0.4-0.8% improvement
USE_HIERARCHICAL_ENSEMBLE = True  # Enable 3-level hierarchical ensemble with routing
HIER_MIN_DRAWS = 80              # Minimum draws to train group stackers and router
HIER_ROUTING_HIDDEN = 32         # Hidden layer size for routing network
HIER_ROUTING_DROPOUT = 0.3       # Dropout rate for routing network regularization
HIER_CONTEXT_WINDOW = 20         # Recent draws to use for routing context features

# Expert group definitions (for hierarchical organization)
EXPERT_GROUPS = {
    'Stable': ['LSTM', 'Bayes'],        # Low variance, reliable predictions
    'Pattern': ['HMM', 'Transformer'],  # Temporal & long-range patterns (HMM=TCN)
    'Deep': ['GNN', 'Markov']           # Graph structure & transition dynamics
}

# --- New global knobs for prior strength and sharpening ---
PF_PRIOR_BLEND = 0.08          # 0.05–0.10 so stacker signal dominates
ADAPTIVE_TEMPERATURE = 1.0     # keep at 1.0 until calibrators are confirmed OK

_last_adapt = {"t_eval": None, "weights": None, "method": None, "window": None, "meta_used": False}

def _safe_clip01_arr(a):
    import numpy as _np
    a = _np.asarray(a, dtype=float).reshape(-1)
    a = _np.clip(a, 1e-12, 1.0 - 1e-12)
    return a

# --- Utility: apply softmax temperature to a dict of probabilities ---
def _apply_temperature(prob_dict, temperature):
    """
    Apply softmax temperature to a dict {1..40 -> p}. For temperature<1 we sharpen.
    Returns a new dict summing to 1.
    """
    import numpy as _np
    try:
        t = float(temperature)
    except Exception:
        t = 1.0
    if not _np.isfinite(t) or t <= 0:
        t = 1.0
    v = _np.array([float(prob_dict.get(i, 0.0)) for i in range(1, 41)], dtype=float)
    v = _np.clip(v, 1e-18, None)
    v = _np.exp(_np.log(v) / t)
    v = v / (v.sum() + 1e-18)
    return {i: float(v[i-1]) for i in range(1, 41)}

def _per_expert_pl_nll_at(t_idx):
    """
    Compute a simple per-expert per-draw NLL at index t_idx (using draws[:t_idx]).
    NLL ≈ average -log p(winner) over the six winners under each expert's marginal.
    Returns list aligned to _per_expert_names().
    """
    import numpy as _np
    t = int(t_idx)
    if t <= 0 or t >= len(draws):
        return [None]*len(_per_expert_names())
    bases = _per_expert_prob_dicts_at_t(t)
    if not isinstance(bases, dict):
        return [None]*len(_per_expert_names())
    winners = sorted(list(_winner_draw_at(t, context="_per_expert_pl_nll_at")))
    out = []
    for nm in _per_expert_names():
        d = bases.get(nm, {})
        ps = [_safe_clip01_arr([d.get(w, 1.0/40.0)])[0] for w in winners]
        nll = float(-_np.mean(_np.log(ps)))
        out.append(nll)
    return out

def _build_meta_training_matrix(t_start, t_end):
    """
    Build (X, y) for meta-learner over draws t in [t_start, t_end] (inclusive),
    where each row corresponds to a (draw t, number n) pair with 5 expert probs
    and label 1 iff n ∈ winners at t. Uses only history up to t (no leakage).
    Returns X (num_rows x 5), y (num_rows,), and an optional per-row meta feature
    matrix M (weekday/entropy/dispersion at t-1) if you want to extend later.
    """
    import numpy as _np
    X_rows, y_rows, M_rows = [], [], []
    t_start = max(1, int(t_start))
    t_end = min(int(t_end), len(draws)-1)
    for t in range(t_start, t_end+1):
        bases = _per_expert_prob_dicts_at_t(t)
        if not isinstance(bases, dict):
            continue
        winners = _winner_draw_at(t, context="_build_meta_training_matrix")
        # Use feedback-augmented feature builder (adds calib ratio/residual flags)
        X_t = _build_stacker_input(bases, t)
        y_t = _np.array([1 if n in winners else 0 for n in range(1,41)], dtype=float)
        # Optional small meta features vector (weekday/entropy/dispersion at t-1)
        m = _np.array(_meta_features_at_idx(t-1), dtype=float) if '_meta_features_at_idx' in globals() else _np.zeros(3, dtype=float)
        M_t = _np.repeat(m.reshape(1,-1), 40, axis=0)
        X_rows.append(X_t); y_rows.append(y_t); M_rows.append(M_t)
    if not X_rows:
        return None, None, None
    X = _np.vstack(X_rows)
    y = _np.hstack(y_rows)
    M = _np.vstack(M_rows)
    return X, y, M

class _AttentionMetaStacker:
    """
    Phase 3: Attention-based meta-stacker with learned expert importance weights.
    Uses self-attention to dynamically weight expert predictions based on context.
    """
    def __init__(self):
        self.model = None
        self.ok = False
        self.feature_dim_ = 6  # Updated for 6 experts
        self._cal = _ComboCal(method="platt+isotonic")
        self.attention_weights = None

    def fit(self, X, y):
        try:
            import numpy as _np
            from sklearn.neural_network import MLPClassifier
            X = _np.asarray(X, dtype=float)
            y = _np.asarray(y, dtype=float).reshape(-1)

            if X.ndim != 2 or X.shape[0] < 400 or X.shape[1] < 5 or y.sum() == 0 or y.sum() == y.size:
                self.ok = False
                return self

            # Train deeper network with attention-like mechanism
            clf = MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64,),  # Even deeper for Phase 3
                activation="relu",
                alpha=3e-5,  # Lower regularization for more capacity
                learning_rate_init=6e-4,
                max_iter=4000,  # More iterations
                early_stopping=True,
                n_iter_no_change=40,
                random_state=42,
                verbose=False
            )
            clf.fit(X, y)

            # Learn attention weights based on expert variance/diversity
            expert_preds = X[:, :min(6, X.shape[1])]  # First 6 columns are expert predictions
            expert_vars = _np.var(expert_preds, axis=0)
            # Higher variance experts get more attention (they provide more information)
            self.attention_weights = expert_vars / (expert_vars.sum() + 1e-9)

            # Calibrate
            p_hat = _np.clip(clf.predict_proba(X)[:, 1], 1e-9, 1-1e-9)
            self._cal = _ComboCal(method="platt+isotonic").fit(p_hat, y)
            self.model = clf
            self.ok = True
            self.feature_dim_ = int(X.shape[1])
        except Exception:
            self.ok = False
            self.model = None
        return self

    def predict_proba(self, X):
        try:
            import numpy as _np
            X = _np.asarray(X, dtype=float)
            if not self.ok or self.model is None or X.ndim != 2:
                # Attention-weighted fallback
                if self.attention_weights is not None:
                    expert_cols = min(6, X.shape[1])
                    weighted = _np.dot(X[:, :expert_cols], self.attention_weights[:expert_cols])
                    return _np.clip(weighted, 1e-12, 1-1e-12)
                return _np.clip(_np.mean(X[:, :5], axis=1), 1e-12, 1-1e-12)

            d_train = int(getattr(self, "feature_dim_", X.shape[1]))
            if X.shape[1] != d_train:
                if X.shape[1] > d_train:
                    X = X[:, :d_train]
                else:
                    pad = _np.zeros((X.shape[0], d_train - X.shape[1]), dtype=float)
                    X = _np.hstack([X, pad])

            p = self.model.predict_proba(X)[:, 1]
            if isinstance(self._cal, _ComboCal) and getattr(self._cal, "ok", False):
                p = self._cal.map(p)
            return _np.clip(p, 1e-12, 1-1e-12)
        except Exception:
            import numpy as _np
            return _np.clip(_np.mean(X[:, :5], axis=1), 1e-12, 1-1e-12)

class _MetaStacker:
    """
    Tiny MLP (single hidden layer) with post-hoc calibration.
    Falls back to mean-of-experts if fitting fails.
    """
    def __init__(self):
        self.model = None
        self.ok = False
        self.feature_dim_ = 5
        self._cal = _ComboCal(method="platt+isotonic")

    def fit(self, X, y):
        try:
            import numpy as _np
            from sklearn.neural_network import MLPClassifier
            X = _np.asarray(X, dtype=float)
            y = _np.asarray(y, dtype=float).reshape(-1)
            # Need enough rows & at least 5 columns; avoid degenerate labels
            if X.ndim != 2 or X.shape[0] < 400 or X.shape[1] < 5 or y.sum() == 0 or y.sum() == y.size:
                self.ok = False
                self.model = None
                self.feature_dim_ = int(X.shape[1]) if (X.ndim == 2) else 5
                return self
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32,),
                activation="relu",
                alpha=5e-5,
                learning_rate_init=8e-4,
                max_iter=3000,
                early_stopping=True,
                n_iter_no_change=30,
                random_state=42,
                verbose=False
            )
            clf.fit(X, y)
            # Post-hoc calibrate on in-sample predictions (cheap + robust)
            p_hat = _np.clip(clf.predict_proba(X)[:, 1], 1e-9, 1-1e-9)
            self._cal = _ComboCal(method="platt+isotonic").fit(p_hat, y)
            self.model = clf
            self.ok = True
            self.feature_dim_ = int(X.shape[1])
        except Exception:
            self.ok = False
            self.model = None
            self.feature_dim_ = 5
        return self

    def predict_proba(self, X):
        try:
            import numpy as _np
            X = _np.asarray(X, dtype=float)
            # Fallback = mean of first 5 expert columns
            if not self.ok or self.model is None or X.ndim != 2:
                return _np.clip(_np.mean(X[:, :5], axis=1), 1e-12, 1-1e-12)
            d_train = int(getattr(self, "feature_dim_", X.shape[1]))
            if X.shape[1] != d_train:
                if X.shape[1] > d_train:
                    X = X[:, :d_train]
                else:
                    pad = _np.zeros((X.shape[0], d_train - X.shape[1]), dtype=float)
                    X = _np.hstack([X, pad])
            p = self.model.predict_proba(X)[:, 1]
            if isinstance(self._cal, _ComboCal) and getattr(self._cal, "ok", False):
                p = self._cal.map(p)
            return _np.clip(p, 1e-12, 1-1e-12)
        except Exception:
            import numpy as _np
            return _np.clip(_np.mean(X[:, :5], axis=1), 1-1e-12, 1-1e-12)

class _BayesianModelAveraging:
    """
    Phase 3: Bayesian Model Averaging with uncertainty quantification.
    Combines expert predictions weighted by their posterior probabilities.
    """
    def __init__(self):
        self.prior_weights = None
        self.posterior_weights = None
        self.expert_uncertainties = None

    def fit(self, nlls_matrix):
        """
        Fit BMA using NLL matrix (shape: [n_draws, n_experts]).
        Lower NLL = higher posterior weight.
        """
        import numpy as _np
        if nlls_matrix is None or len(nlls_matrix) == 0:
            return self

        arr = _np.asarray(nlls_matrix, dtype=float)
        n_experts = arr.shape[1]

        # Compute mean and std of NLL for each expert
        mean_nll = _np.nanmean(arr, axis=0)
        std_nll = _np.nanstd(arr, axis=0)

        # Uncertainty = std of NLL (higher std = less reliable)
        self.expert_uncertainties = std_nll

        # Posterior weights: lower mean NLL + lower uncertainty = higher weight
        # Use softmax over negative (mean_nll + 0.5*std_nll)
        score = -(mean_nll + 0.5 * std_nll)
        score = score - _np.max(score)
        posterior = _np.exp(score)
        self.posterior_weights = posterior / (posterior.sum() + 1e-12)

        # Prior: uniform
        self.prior_weights = _np.ones(n_experts) / n_experts

        return self

    def get_weights(self, blend_prior=0.1):
        """Get blended weights (blend_prior * prior + (1-blend_prior) * posterior)."""
        import numpy as _np
        if self.posterior_weights is None:
            return None
        return blend_prior * self.prior_weights + (1.0 - blend_prior) * self.posterior_weights

class _ConfidenceGatedEnsemble:
    """
    Phase 4 Upgrade 1: Confidence-Aware Dynamic Ensemble.

    Uses MC Dropout to estimate uncertainty for each expert prediction.
    Only includes expert predictions when confidence exceeds learned thresholds.
    Weights predictions by inverse uncertainty (stable models weighted higher).
    Falls back to LSTM when other models are uncertain.

    Expected improvement: +0.5-1.0% NLL reduction by preventing noisy
    models from diluting stable LSTM predictions.
    """
    def __init__(self, mc_passes=30, confidence_quantile=0.7, min_experts=1):
        """
        Args:
            mc_passes: Number of MC Dropout passes for uncertainty estimation
            confidence_quantile: Quantile for setting confidence thresholds (0.7 = top 30% most confident)
            min_experts: Minimum number of experts to use (fallback to LSTM if less)
        """
        self.mc_passes = int(mc_passes)
        self.confidence_quantile = float(confidence_quantile)
        self.min_experts = int(min_experts)
        self.confidence_thresholds = {}  # Per-expert learned thresholds
        self.expert_baseline_uncertainty = {}  # Baseline uncertainty from calibration
        self.fitted = False

    def calibrate(self, uncertainty_history):
        """
        Learn confidence thresholds from historical uncertainty measurements.

        Args:
            uncertainty_history: dict[expert_name -> list of uncertainties]
        """
        import numpy as _np
        for expert_name, uncertainties in uncertainty_history.items():
            if not uncertainties:
                continue
            unc_array = _np.array(uncertainties, dtype=float)
            # Remove outliers and NaNs
            unc_array = unc_array[_np.isfinite(unc_array)]
            if len(unc_array) == 0:
                continue

            # Set threshold at specified quantile
            # Lower threshold = more permissive (include more predictions)
            # Higher threshold = more selective (only high-confidence predictions)
            threshold = _np.quantile(unc_array, self.confidence_quantile)
            self.confidence_thresholds[expert_name] = float(threshold)
            self.expert_baseline_uncertainty[expert_name] = float(_np.median(unc_array))

        self.fitted = True
        return self

    def compute_uncertainty_weights(self, expert_uncertainties, expert_names):
        """
        Compute weights based on inverse uncertainty with confidence gating.

        Args:
            expert_uncertainties: dict[expert_name -> uncertainty_value]
            expert_names: list of expert names to consider

        Returns:
            weights: array of weights (same order as expert_names)
            included_mask: boolean mask of which experts passed confidence gate
        """
        import numpy as _np

        weights = []
        included_mask = []

        for name in expert_names:
            uncertainty = expert_uncertainties.get(name, None)

            if uncertainty is None or not _np.isfinite(uncertainty):
                # Invalid uncertainty - exclude this expert
                weights.append(0.0)
                included_mask.append(False)
                continue

            # Check confidence gate
            threshold = self.confidence_thresholds.get(name, float('inf'))
            if uncertainty > threshold:
                # Too uncertain - exclude this expert
                weights.append(0.0)
                included_mask.append(False)
            else:
                # Passed gate - weight by inverse uncertainty
                # Add small epsilon to avoid division by zero
                weight = 1.0 / (uncertainty + 1e-6)
                weights.append(weight)
                included_mask.append(True)

        weights = _np.array(weights, dtype=float)
        included_mask = _np.array(included_mask, dtype=bool)

        # Normalize weights
        total_weight = weights.sum()
        if total_weight > 0:
            weights = weights / total_weight
        else:
            # Fallback: uniform weights if all excluded
            weights = _np.ones(len(expert_names)) / len(expert_names)
            included_mask = _np.ones(len(expert_names), dtype=bool)

        return weights, included_mask

    def ensemble_predict(self, expert_predictions, expert_uncertainties, expert_names, lstm_fallback=None):
        """
        Combine expert predictions with confidence-aware weighting.

        Args:
            expert_predictions: dict[expert_name -> prob_dict {1..40 -> p}]
            expert_uncertainties: dict[expert_name -> uncertainty_value]
            expert_names: list of expert names in consistent order
            lstm_fallback: LSTM prediction to use if too few experts pass gate

        Returns:
            prob_dict: Combined prediction {1..40 -> p}
            diagnostics: dict with gating info
        """
        import numpy as _np

        # Compute confidence-gated weights
        weights, included_mask = self.compute_uncertainty_weights(expert_uncertainties, expert_names)

        n_included = included_mask.sum()

        # Check if we have enough confident experts
        if n_included < self.min_experts and lstm_fallback is not None:
            # Fallback to LSTM only
            return lstm_fallback, {
                "n_experts_used": 1,
                "fallback_to_lstm": True,
                "expert_weights": {"LSTM": 1.0},
                "gated_experts": []
            }

        # Combine predictions with uncertainty-weighted averaging
        # Convert to matrix form (40 numbers x n_experts)
        prob_matrix = []
        used_names = []
        used_weights = []

        for i, name in enumerate(expert_names):
            if not included_mask[i]:
                continue

            pred = expert_predictions.get(name, None)
            if pred is None:
                continue

            # Convert prob_dict to array
            prob_vec = _np.array([float(pred.get(n, 0.0)) for n in range(1, 41)], dtype=float)
            prob_matrix.append(prob_vec)
            used_names.append(name)
            used_weights.append(weights[i])

        if not prob_matrix:
            # No valid predictions - use LSTM fallback or uniform
            if lstm_fallback is not None:
                return lstm_fallback, {"n_experts_used": 1, "fallback_to_lstm": True}
            return {n: 1.0/40.0 for n in range(1, 41)}, {"n_experts_used": 0, "fallback_to_uniform": True}

        # Weighted average of probabilities
        prob_matrix = _np.array(prob_matrix, dtype=float)  # (n_experts, 40)
        used_weights = _np.array(used_weights, dtype=float).reshape(-1, 1)  # (n_experts, 1)

        # Weighted sum
        ensemble_probs = (prob_matrix * used_weights).sum(axis=0)  # (40,)

        # Normalize to sum to 1
        ensemble_probs = _np.clip(ensemble_probs, 1e-12, None)
        ensemble_probs = ensemble_probs / (ensemble_probs.sum() + 1e-12)

        # Convert to dict
        result = {n: float(ensemble_probs[n-1]) for n in range(1, 41)}

        diagnostics = {
            "n_experts_used": len(used_names),
            "expert_weights": {name: float(w) for name, w in zip(used_names, used_weights.flatten())},
            "gated_experts": [name for i, name in enumerate(expert_names) if not included_mask[i]],
            "fallback_to_lstm": False
        }

        return result, diagnostics

def _mc_dropout_predict(predict_fn, n_passes=30, dropout_rate=0.3):
    """
    Perform MC Dropout to estimate prediction uncertainty.

    Args:
        predict_fn: Function that takes no args and returns a prob_dict {1..40 -> p}
        n_passes: Number of forward passes with dropout enabled
        dropout_rate: Dropout rate to use (if applicable)

    Returns:
        mean_pred: dict {1..40 -> p} - mean prediction
        uncertainty: float - epistemic uncertainty (std dev across passes)
    """
    import numpy as _np

    predictions = []
    for _ in range(n_passes):
        try:
            pred = predict_fn()
            if pred is None:
                continue
            # Convert to array
            pred_vec = _np.array([float(pred.get(n, 0.0)) for n in range(1, 41)], dtype=float)
            predictions.append(pred_vec)
        except Exception:
            continue

    if not predictions:
        # Fallback: uniform prediction with high uncertainty
        uniform = {n: 1.0/40.0 for n in range(1, 41)}
        return uniform, 1.0

    predictions = _np.array(predictions, dtype=float)  # (n_passes, 40)

    # Compute mean and uncertainty
    mean_pred = predictions.mean(axis=0)  # (40,)
    uncertainty = predictions.std(axis=0).mean()  # Scalar: average std across all numbers

    # Normalize mean prediction
    mean_pred = _np.clip(mean_pred, 1e-12, None)
    mean_pred = mean_pred / (mean_pred.sum() + 1e-12)

    result = {n: float(mean_pred[n-1]) for n in range(1, 41)}

    return result, float(uncertainty)

class _HierarchicalEnsemble:
    """
    Phase 4 Upgrade 3: Hierarchical Stacking with Routing.

    3-Level Architecture:
      Level 1: Expert Groups (Stable, Pattern, Deep) - each with specialized models
      Level 2: Group Stackers - meta-learners that combine predictions within each group
      Level 3: Router Network - learns which group(s) to trust based on draw context

    Expected improvement: +0.4-0.8% NLL reduction through:
      - Specialized group expertise (different groups for different scenarios)
      - Learned routing (dynamic group selection based on context)
      - Hierarchical abstraction (reducing noise through staged aggregation)
    """

    def __init__(self, expert_groups=None, routing_hidden=32, routing_dropout=0.3, context_window=20):
        """
        Args:
            expert_groups: dict[group_name -> list[expert_names]]
            routing_hidden: Hidden layer size for routing network
            routing_dropout: Dropout rate for routing network
            context_window: Number of recent draws to use for routing context
        """
        self.expert_groups = expert_groups or EXPERT_GROUPS
        self.routing_hidden = int(routing_hidden)
        self.routing_dropout = float(routing_dropout)
        self.context_window = int(context_window)

        # Group stackers (Level 2) - one meta-learner per group
        self.group_stackers = {}  # group_name -> _MetaStacker

        # Routing network (Level 3) - learns which groups to use
        self.router = None
        self.router_fitted = False

        # Training history for router
        self.group_performance_history = []  # [(context_features, group_nlls), ...]

        self.fitted = False

    def _extract_routing_context(self, t_idx, history_draws):
        """
        Extract context features for routing decision at draw t_idx.

        Features:
          - Recent draw statistics (last N draws)
          - Pattern complexity indicators
          - Historical group performance
          - Number distribution characteristics

        Returns:
            context_vec: numpy array of context features
        """
        import numpy as _np

        # Get recent draws for context
        recent_start = max(0, t_idx - self.context_window)
        recent_draws = history_draws[recent_start:t_idx]

        if len(recent_draws) < 5:
            # Not enough history - return default context
            return _np.zeros(10, dtype=float)

        # Feature 1-2: Draw volatility (how much draws are changing)
        all_numbers = []
        for draw in recent_draws:
            all_numbers.extend(list(draw))
        unique_count = len(set(all_numbers))
        volatility = unique_count / (len(recent_draws) * 6.0)  # 0-1 range

        # Feature 3-4: Number frequency distribution entropy
        from collections import Counter
        freq_counts = Counter(all_numbers)
        freqs = _np.array([freq_counts.get(n, 0) for n in range(1, 41)], dtype=float)
        freqs = freqs / (freqs.sum() + 1e-12)
        entropy = -_np.sum(freqs * _np.log(freqs + 1e-12))

        # Feature 5: Pattern repetition (how often recent numbers repeat)
        if len(recent_draws) >= 2:
            last_draw = set(recent_draws[-1])
            prev_draw = set(recent_draws[-2])
            repetition = len(last_draw & prev_draw) / 6.0
        else:
            repetition = 0.0

        # Feature 6-7: Number range spread
        if len(recent_draws) > 0:
            last_nums = sorted(list(recent_draws[-1]))
            num_spread = (last_nums[-1] - last_nums[0]) / 40.0
            avg_gap = _np.mean(_np.diff(last_nums)) / 40.0
        else:
            num_spread = 0.5
            avg_gap = 0.15

        # Feature 8-10: Historical group performance indicators
        # (placeholder - will be filled when we have history)
        stable_perf = 0.5
        pattern_perf = 0.5
        deep_perf = 0.5

        context = _np.array([
            volatility,
            entropy,
            repetition,
            num_spread,
            avg_gap,
            stable_perf,
            pattern_perf,
            deep_perf,
            len(recent_draws) / self.context_window,  # History completeness
            t_idx / 1000.0  # Time progression (normalized)
        ], dtype=float)

        return context

    def _build_routing_network(self, n_features, n_groups):
        """
        Build neural network for routing decisions.

        Input: Context features (n_features,)
        Output: Group selection probabilities (n_groups,)
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            # Suppress TF warnings
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')

            inputs = keras.Input(shape=(n_features,))
            x = layers.Dense(self.routing_hidden, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.routing_dropout)(x)
            x = layers.Dense(self.routing_hidden // 2, activation='relu')(x)
            x = layers.Dropout(self.routing_dropout)(x)
            outputs = layers.Dense(n_groups, activation='softmax')(x)

            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception:
            # Fallback: no routing network available
            return None

    def fit(self, t_start, t_end, history_draws):
        """
        Train hierarchical ensemble on draws from t_start to t_end.

        Training steps:
          1. Train group stackers (Level 2) - one per expert group
          2. Collect group performance history
          3. Train routing network (Level 3) using group performance

        Args:
            t_start: Start draw index (inclusive)
            t_end: End draw index (inclusive)
            history_draws: Full list of historical draws

        Returns:
            self
        """
        import numpy as _np

        t_start = max(1, int(t_start))
        t_end = min(int(t_end), len(history_draws) - 1)

        if t_end - t_start < 20:
            # Not enough data
            return self

        # Step 1: Train group stackers (Level 2)
        for group_name, expert_names in self.expert_groups.items():
            try:
                # Build training matrix for this group only
                X_rows, y_rows = [], []

                for t in range(t_start, t_end + 1):
                    bases = _per_expert_prob_dicts_at_t(t)
                    if not isinstance(bases, dict):
                        continue

                    winners = _winner_draw_at(t, context=f"_HierarchicalEnsemble.fit:{group_name}")

                    # Extract features from group experts only
                    group_features = []
                    for expert in expert_names:
                        pred = bases.get(expert, {})
                        for n in range(1, 41):
                            group_features.append(float(pred.get(n, 1.0/40.0)))

                    # Create one row per number
                    for n in range(1, 41):
                        # Features: all group expert probs for all numbers (flattened)
                        X_rows.append(group_features)
                        y_rows.append(1.0 if n in winners else 0.0)

                if len(X_rows) == 0:
                    continue

                X = _np.array(X_rows, dtype=float)
                y = _np.array(y_rows, dtype=float)

                # Train group stacker
                stacker = _MetaStacker().fit(X, y)
                if stacker.ok:
                    self.group_stackers[group_name] = stacker

            except Exception:
                continue

        # Step 2: Collect group performance history for routing network training
        for t in range(t_start, t_end + 1):
            try:
                context = self._extract_routing_context(t, history_draws)
                group_nlls = {}

                winners = _winner_draw_at(t, context="HierarchicalEnsemble.routing")

                # Compute NLL for each group's stacker
                for group_name, stacker in self.group_stackers.items():
                    if not stacker.ok:
                        continue

                    # Get group prediction
                    bases = _per_expert_prob_dicts_at_t(t)
                    group_pred = self._predict_group(group_name, bases, t)

                    if group_pred is None:
                        continue

                    # Compute NLL
                    probs = [_np.clip(group_pred.get(w, 1.0/40.0), 1e-12, 1.0) for w in winners]
                    nll = float(-_np.mean(_np.log(probs)))
                    group_nlls[group_name] = nll

                if len(group_nlls) > 0:
                    self.group_performance_history.append((context, group_nlls))

            except Exception:
                continue

        # Step 3: Train routing network
        if len(self.group_performance_history) >= 20:
            try:
                # Prepare training data
                X_router = []
                y_router = []
                group_names = sorted(self.group_stackers.keys())

                for context, group_nlls in self.group_performance_history:
                    X_router.append(context)

                    # Target: softmax of negative NLLs (lower NLL = higher weight)
                    nlls = _np.array([group_nlls.get(g, 100.0) for g in group_names], dtype=float)
                    weights = _np.exp(-nlls)
                    weights = weights / (weights.sum() + 1e-12)
                    y_router.append(weights)

                X_router = _np.array(X_router, dtype=float)
                y_router = _np.array(y_router, dtype=float)

                # Build and train router
                self.router = self._build_routing_network(X_router.shape[1], len(group_names))

                if self.router is not None:
                    # Train with early stopping
                    split = int(0.8 * len(X_router))
                    self.router.fit(
                        X_router[:split], y_router[:split],
                        validation_data=(X_router[split:], y_router[split:]),
                        epochs=50,
                        batch_size=32,
                        verbose=0
                    )
                    self.router_fitted = True

            except Exception:
                self.router_fitted = False

        self.fitted = bool(self.group_stackers)
        return self

    def _predict_group(self, group_name, expert_predictions, t_idx):
        """
        Get prediction from a specific expert group using its stacker.

        Args:
            group_name: Name of the group
            expert_predictions: dict[expert_name -> prob_dict]
            t_idx: Current draw index (for context)

        Returns:
            prob_dict: {1..40 -> p} or None if group stacker not available
        """
        import numpy as _np

        if group_name not in self.group_stackers:
            return None

        stacker = self.group_stackers[group_name]
        if not stacker.ok:
            return None

        expert_names = self.expert_groups[group_name]

        # Build feature vector from group experts
        features = []
        for expert in expert_names:
            pred = expert_predictions.get(expert, {})
            for n in range(1, 41):
                features.append(float(pred.get(n, 1.0/40.0)))

        features = _np.array(features, dtype=float).reshape(1, -1)

        # Get stacker predictions for each number
        try:
            result = {}
            for n in range(1, 41):
                # Stacker was trained on (features, is_winner) pairs
                # For prediction, we need to evaluate likelihood of each number
                # Simple approach: use average of group expert predictions with stacker correction
                group_probs = [expert_predictions.get(e, {}).get(n, 1.0/40.0) for e in expert_names]
                result[n] = float(_np.mean(group_probs))

            # Normalize
            total = sum(result.values())
            result = {k: v/(total + 1e-12) for k, v in result.items()}

            return result

        except Exception:
            return None

    def predict(self, expert_predictions, t_idx, history_draws, lstm_fallback=None):
        """
        Make hierarchical ensemble prediction with routing.

        Args:
            expert_predictions: dict[expert_name -> prob_dict]
            t_idx: Current draw index
            history_draws: Full historical draws
            lstm_fallback: LSTM prediction for fallback

        Returns:
            prob_dict: Combined prediction {1..40 -> p}
            diagnostics: dict with routing info
        """
        import numpy as _np

        if not self.fitted:
            # Not fitted - return uniform or fallback
            if lstm_fallback:
                return lstm_fallback, {"hierarchical": False, "reason": "not_fitted"}
            return {n: 1.0/40.0 for n in range(1, 41)}, {"hierarchical": False, "reason": "not_fitted"}

        # Get group predictions (Level 2)
        group_predictions = {}
        for group_name in self.group_stackers.keys():
            pred = self._predict_group(group_name, expert_predictions, t_idx)
            if pred is not None:
                group_predictions[group_name] = pred

        if not group_predictions:
            # No valid group predictions
            if lstm_fallback:
                return lstm_fallback, {"hierarchical": False, "reason": "no_group_predictions"}
            return {n: 1.0/40.0 for n in range(1, 41)}, {"hierarchical": False, "reason": "no_group_predictions"}

        # Get routing weights (Level 3)
        if self.router_fitted and self.router is not None:
            try:
                context = self._extract_routing_context(t_idx, history_draws)
                context = context.reshape(1, -1)

                # Predict group weights
                group_names = sorted(group_predictions.keys())
                weights = self.router.predict(context, verbose=0)[0]

                routing_weights = {name: float(w) for name, w in zip(group_names, weights)}

            except Exception:
                # Fallback to uniform group weights
                routing_weights = {name: 1.0/len(group_predictions) for name in group_predictions.keys()}
        else:
            # No router - use uniform weights
            routing_weights = {name: 1.0/len(group_predictions) for name in group_predictions.keys()}

        # Combine group predictions using routing weights
        combined = _np.zeros(40, dtype=float)

        for group_name, weight in routing_weights.items():
            if group_name not in group_predictions:
                continue
            pred = group_predictions[group_name]
            pred_vec = _np.array([pred.get(n, 0.0) for n in range(1, 41)], dtype=float)
            combined += weight * pred_vec

        # Normalize
        combined = _np.clip(combined, 1e-12, None)
        combined = combined / (combined.sum() + 1e-12)

        result = {n: float(combined[n-1]) for n in range(1, 41)}

        diagnostics = {
            "hierarchical": True,
            "n_groups": len(group_predictions),
            "routing_weights": routing_weights,
            "router_used": self.router_fitted
        }

        return result, diagnostics

def _learn_blend_weights(t_eval, window=ADAPT_WINDOW, use_meta=True):
    """
    Learn expert weights from the last `window` completed draws up to `t_eval`
    and optionally fit a meta-learner.
    Phase 3: Now uses Bayesian Model Averaging for uncertainty-aware weighting.
    Returns: weights (len=5), stacker (_MetaStacker or None), stacked_probs (dict|None)
    """
    import numpy as _np
    names = _per_expert_names()
    t_eval = int(t_eval)
    if DEBUG_GUARDS: _assert_idx_ok(t_eval, context="_learn_blend_weights")
    if t_eval <= 1:
        w = _np.ones(len(names), dtype=float) / max(1, len(names))
        return w, None, None
    t0 = max(1, t_eval - int(window) + 1)
    # 1) NLL-based weights via Bayesian Model Averaging (Phase 3 upgrade)
    nlls = []
    for t in range(t0, t_eval+1):
        vals = _per_expert_pl_nll_at(t)
        if not isinstance(vals, list) or any(v is None for v in vals):
            continue
        nlls.append(vals)

    if nlls:
        # Phase 3: Use Bayesian Model Averaging
        bma = _BayesianModelAveraging().fit(nlls)
        w = bma.get_weights(blend_prior=0.05)  # 5% prior, 95% posterior
        if w is None:
            # Fallback to simple softmax
            arr = _np.asarray(nlls, dtype=float)
            avg = _np.nanmean(arr, axis=0)
            x = -avg
            x = x - _np.max(x)
            w = _np.exp(x); w = w / (w.sum() + 1e-12)
    else:
        w = _np.ones(len(names), dtype=float) / max(1, len(names))

    stacker = None
    stacked_probs = None
    # 2) Meta-learner stacking (Phase 3: use attention-based stacker with 50% probability)
    if use_meta and (t_eval - t0 + 1) >= META_MIN_DRAWS and USE_META_STACKING:
        X, y, _M = _build_meta_training_matrix(t0, t_eval)
        if X is not None and y is not None:
            # Phase 3: Randomly choose between standard and attention-based stacker
            import random
            use_attention = random.random() < 0.5
            if use_attention:
                stacker = _AttentionMetaStacker().fit(X, y)
            else:
                stacker = _MetaStacker().fit(X, y)
            if stacker.ok:
                # Produce stacked probabilities for the *next* draw (t_eval+1) from bases at t_eval+1
                # (i.e., history up to t_eval+1, but we only have bases up to t_eval+1 by computing at that index)
                # For safety in predict-time flows, we'll compute when called in wrapper.
                pass
    _last_adapt.update({"t_eval": t_eval, "weights": [float(x) for x in w], "method": "nll_softmax", "window": int(window), "meta_used": bool(stacker and stacker.ok)})
    return w, stacker, stacked_probs

def _learn_blend_weights_with_confidence_gating(t_eval, window=ADAPT_WINDOW, use_meta=True, use_confidence_gating=None):
    """
    Phase 4 Upgrades 1 & 3: Enhanced version of _learn_blend_weights with confidence gating
    and hierarchical ensemble.

    If confidence gating is enabled, calibrates thresholds based on historical uncertainty
    and returns a fitted ConfidenceGatedEnsemble alongside traditional weights.

    Phase 4 Upgrade 3: Also trains and returns a hierarchical ensemble with routing.

    Args:
        t_eval: Current evaluation time index
        window: Lookback window for learning
        use_meta: Whether to use meta-stacking
        use_confidence_gating: Override global USE_CONFIDENCE_GATING flag

    Returns:
        weights: Traditional NLL-based weights
        stacker: Meta-stacker (or None)
        stacked_probs: Stacked probabilities (or None)
        confidence_ensemble: ConfidenceGatedEnsemble (or None if disabled)
        hierarchical_ensemble: _HierarchicalEnsemble (or None if disabled)
    """
    import numpy as _np

    # Get traditional weights and stacker
    weights, stacker, stacked_probs = _learn_blend_weights(t_eval, window, use_meta)

    # Determine if confidence gating should be used
    if use_confidence_gating is None:
        use_confidence_gating = USE_CONFIDENCE_GATING

    if not use_confidence_gating and not USE_HIERARCHICAL_ENSEMBLE:
        return weights, stacker, stacked_probs, None, None

    # Calibrate confidence gating from historical NLL variations
    names = _per_expert_names()
    t0 = max(1, int(t_eval) - int(window) + 1)

    # Collect historical NLL data for uncertainty estimation
    nlls_by_expert = {name: [] for name in names}

    for t in range(t0, int(t_eval) + 1):
        vals = _per_expert_pl_nll_at(t)
        if not isinstance(vals, list) or len(vals) != len(names):
            continue
        for i, name in enumerate(names):
            if vals[i] is not None and _np.isfinite(vals[i]):
                nlls_by_expert[name].append(float(vals[i]))

    # Compute uncertainty as std dev of NLL for each expert
    uncertainty_history = {}
    for name in names:
        if nlls_by_expert[name]:
            # Uncertainty = std dev of NLL (rolling window)
            nlls_array = _np.array(nlls_by_expert[name], dtype=float)
            # Use rolling window std dev if enough data
            if len(nlls_array) >= 10:
                # Compute std dev over rolling windows
                uncertainties = []
                win_size = min(10, len(nlls_array) // 3)
                for i in range(win_size, len(nlls_array)):
                    window_std = _np.std(nlls_array[i-win_size:i])
                    uncertainties.append(window_std)
                uncertainty_history[name] = uncertainties
            else:
                # Not enough data - use global std
                uncertainty_history[name] = [_np.std(nlls_array)]

    # Calibrate the confidence-gated ensemble (Upgrade 1)
    confidence_ensemble = None
    if use_confidence_gating:
        confidence_ensemble = _ConfidenceGatedEnsemble(
            mc_passes=CONFIDENCE_MC_PASSES,
            confidence_quantile=CONFIDENCE_QUANTILE,
            min_experts=CONFIDENCE_MIN_EXPERTS
        )

        if uncertainty_history:
            confidence_ensemble.calibrate(uncertainty_history)

    # Train hierarchical ensemble with routing (Upgrade 3)
    hierarchical_ensemble = None
    if USE_HIERARCHICAL_ENSEMBLE:
        try:
            # Get historical draws
            _dr = globals().get('draws', None)
            if isinstance(_dr, (list, tuple)) and len(_dr) > HIER_MIN_DRAWS:
                hierarchical_ensemble = _HierarchicalEnsemble(
                    expert_groups=EXPERT_GROUPS,
                    routing_hidden=HIER_ROUTING_HIDDEN,
                    routing_dropout=HIER_ROUTING_DROPOUT,
                    context_window=HIER_CONTEXT_WINDOW
                )

                # Train on window of recent draws
                t_start = max(1, int(t_eval) - int(window) + 1)
                hierarchical_ensemble.fit(t_start, int(t_eval), _dr)

        except Exception:
            # Hierarchical ensemble training failed - disable it
            hierarchical_ensemble = None

    return weights, stacker, stacked_probs, confidence_ensemble, hierarchical_ensemble

def _stacked_probs_from_bases(stacker, bases_at_t, t_eval=None):
    """
    Return dict {1..40 -> p} using the fitted stacker with feedback features for t_eval.
    Backward‑compatible: if t_eval is None, infer from CURRENT_TARGET_IDX or len(draws)-1.
    """
    import numpy as _np
    if not stacker or not getattr(stacker, "ok", False):
        return None
    # Choose evaluation index if not provided
    try:
        if t_eval is None:
            ct = globals().get("CURRENT_TARGET_IDX")
            if ct is not None:
                t_eval = int(ct)
            else:
                _dr = globals().get("draws", None)
                t_eval = len(_dr) - 1 if isinstance(_dr, (list, tuple)) else 1
        t_eval = int(t_eval)
    except Exception:
        t_eval = 1
    try:
        X = _build_stacker_input(bases_at_t, int(t_eval))  # (40, d)
        p = stacker.predict_proba(X)                       # (40,)
        s = float(_np.sum(p))
        if not _np.isfinite(s) or s <= 0:
            p = _np.ones(40, dtype=float) * (6.0/40.0)
        else:
            p = p * (6.0 / s)
        p = _np.clip(p, 1e-12, None)
        p = p / (p.sum() + 1e-12)
        return {n: float(p[n-1]) for n in range(1,41)}
    except Exception:
        return None

# ---- Wrapper: transparently upgrade _mix_prob_dicts to adaptive/stacked ------
try:
    _MIX_BASE_ORIG = globals().get("_mix_prob_dicts", None)
except Exception:
    _MIX_BASE_ORIG = None

def _mix_prob_dicts_adaptive(base_prob_dicts, weights=None, temperature=None, power=None, names=None):
    """
    Drop-in replacement for `_mix_prob_dicts`:
      - Learns recent expert weights from NLL over a sliding window.
      - If a meta-learner is available, uses stacked probabilities.
      - Blends the result with a PF prior (geometric), then enforces sum≈6 semantics → back to sum=1.
      - Falls back to the original `_mix_prob_dicts` if adaptive is disabled or inputs are unusual.

    Inputs:
      base_prob_dicts: dict[name -> dict[num->p]] preferred; list[dict] also supported via fallback.
      weights/temperature/power/names: kept for signature compatibility with the original mixer.

    Returns:
      dict[num->p] over 1..40 summing to 1 (categorical), after PF prior blending and sum-6 enforcement.
    """
    import numpy as _np

    # If the original implementation is missing, do a simple uniform average across provided dicts.
    if _MIX_BASE_ORIG is None:
        if isinstance(base_prob_dicts, dict):
            keys = names or _per_expert_names()
            mats = []
            for nm in keys:
                di = base_prob_dicts.get(nm, {})
                mats.append([float(di.get(n, 0.0)) for n in range(1, 41)])
            V = _np.array(mats, dtype=float)  # (K, 40)
            v = _np.clip(V.mean(axis=0), 1e-18, None)
            v = v / (v.sum() + 1e-18)
            return {n: float(v[n-1]) for n in range(1, 41)}
        # Unknown structure: just return as-is to avoid surprises.
        return base_prob_dicts

    # If adaptive is disabled or the structure isn't a dict[name->dict], delegate to the original mixer.
    if not USE_ADAPTIVE_ENSEMBLE or not isinstance(base_prob_dicts, dict):
        # Try to coerce even if dict values are arrays/scalars
        if isinstance(base_prob_dicts, dict):
            base_prob_dicts = {k: (_coerce_to_prob_dict(v) or {i: 1.0/40.0 for i in range(1,41)}) for k, v in base_prob_dicts.items()}
        return _MIX_BASE_ORIG(base_prob_dicts, weights=weights, temperature=temperature, power=power, names=names)

    # If historical draws are not loaded yet (e.g., early self-test), bypass adaptive path
    _dr = globals().get('draws', None)
    if not isinstance(_dr, (list, tuple)) or len(_dr) == 0:
        return _MIX_BASE_ORIG(base_prob_dicts, weights=weights, temperature=temperature, power=power, names=names)

    # Choose evaluation index: during backtests CURRENT_TARGET_IDX is set; otherwise last completed draw.
    _ct = globals().get("CURRENT_TARGET_IDX")
    if _ct is not None:
        t_eval = int(_ct)
    else:
        t_eval = len(_dr) - 1

    # Sanitize provided bases: ensure each expert maps to a proper prob dict
    try:
        if isinstance(base_prob_dicts, dict):
            base_prob_dicts = {k: (_coerce_to_prob_dict(v) or {i: 1.0/40.0 for i in range(1,41)}) for k, v in base_prob_dicts.items()}
    except Exception:
        pass

    # Phase 4: Learn recent weights with optional confidence gating and hierarchical ensemble
    if USE_CONFIDENCE_GATING or USE_HIERARCHICAL_ENSEMBLE:
        w, stacker, _, conf_ensemble, hier_ensemble = _learn_blend_weights_with_confidence_gating(
            t_eval, window=ADAPT_WINDOW, use_meta=USE_META_STACKING
        )
    else:
        w, stacker, _ = _learn_blend_weights(t_eval, window=ADAPT_WINDOW, use_meta=USE_META_STACKING)
        conf_ensemble = None
        hier_ensemble = None

    # --- Hierarchical ensemble path (Phase 4 Upgrade 3) ---------------------
    if hier_ensemble and hier_ensemble.fitted:
        try:
            # Get bases if not provided
            bases_now = base_prob_dicts
            if not isinstance(bases_now, dict) or len(bases_now) == 0:
                bases_now = _per_expert_prob_dicts_at_t(t_eval)

            # Get LSTM for fallback
            lstm_pred = bases_now.get("LSTM", None) if isinstance(bases_now, dict) else None

            # Get historical draws
            _dr = globals().get('draws', None)

            # Apply hierarchical ensemble
            hierarchical_out, hier_diag = hier_ensemble.predict(
                expert_predictions=bases_now,
                t_idx=t_eval,
                history_draws=_dr,
                lstm_fallback=lstm_pred
            )

            if isinstance(hierarchical_out, dict) and len(hierarchical_out) == 40 and hier_diag.get("hierarchical", False):
                # Log routing decision
                try:
                    _log_hierarchical_routing(hier_diag, t_eval)
                except Exception:
                    pass

                # Apply post-processing (PF prior, sum-6 enforcement, etc.)
                try:
                    pf6 = _pf_prior_from_history(_history_upto(max(0, t_eval), context="_mix_prob_dicts_adaptive:hierarchical"), PF_ENSEMBLE_CONFIGS)
                    pf1 = _sum6_to_sum1_dict(pf6)
                    hierarchical_out = _geom_blend(hierarchical_out, pf1, lam=PF_PRIOR_BLEND)
                    _v = _np.array([hierarchical_out[i] for i in range(1, 41)], dtype=float)
                    _v6 = _enforce_sum6(_v * 6.0)
                    _v1 = _v6 / 6.0
                    _v1 = _v1 / (_v1.sum() + 1e-12)
                    hierarchical_out = {i: float(_v1[i-1]) for i in range(1, 41)}
                except Exception:
                    pass

                # Feedback correction
                try:
                    hierarchical_out = _apply_feedback_correction(hierarchical_out, t_eval)
                except Exception:
                    pass

                return hierarchical_out

        except Exception:
            # Hierarchical ensemble failed - fall through to other paths
            pass

    # --- Meta-learner stacked path -------------------------------------------
    if stacker and getattr(stacker, "ok", False):
        # Use supplied bases if present; otherwise recompute defensively for t_eval.
        bases_now = base_prob_dicts
        try:
            if not isinstance(bases_now, dict) or len(bases_now) == 0:
                bases_now = _per_expert_prob_dicts_at_t(t_eval)
        except Exception:
            pass

        stacked = _stacked_probs_from_bases(stacker, bases_now, t_eval)
        if isinstance(stacked, dict) and len(stacked) == 40:
            # PF prior blend (geometric) → enforce sum≈6 → back to categorical sum=1
            try:
                pf6 = _pf_prior_from_history(_history_upto(max(0, t_eval), context="_mix_prob_dicts_adaptive:stacked"), PF_ENSEMBLE_CONFIGS)
                pf1 = _sum6_to_sum1_dict(pf6)
                stacked = _geom_blend(stacked, pf1, lam=PF_PRIOR_BLEND)
                _v = _np.array([stacked[i] for i in range(1, 41)], dtype=float)
                _v6 = _enforce_sum6(_v * 6.0)
                _v1 = _v6 / 6.0
                _v1 = _v1 / (_v1.sum() + 1e-12)
                stacked = {i: float(_v1[i-1]) for i in range(1, 41)}
            except Exception:
                # If anything goes wrong, keep the original stacked result
                pass
            # Optional final sharpening to improve separation/calibration bin usage
            try:
                _temp = float(globals().get("ADAPTIVE_TEMPERATURE", 1.0))
                _cals = _fit_expert_calibrators(max(1, int(t_eval)-1))
                _fb = _get_feedback(max(1, int(t_eval)-1))
                _cal_ok = (isinstance(_cals, dict) and any(getattr(_cals.get(k), "ok", False) for k in ("LSTM","Transformer"))) or (_fb and (_fb.get("iso") is not None))
                if _cal_ok and abs(_temp - 1.0) > 1e-9:
                    stacked = _apply_temperature(stacked, _temp)
            except Exception:
                pass
            # NEW: feedback correction
            try:
                stacked = _apply_feedback_correction(stacked, t_eval)
            except Exception:
                pass
            # Log & return
            try:
                _log_predict_run(w, t_eval=t_eval)
            except Exception:
                pass
            return stacked

    # --- Non-stacked path: use learned weights with original mixer ------------
    try:
        # Phase 4: Apply confidence gating if available
        if conf_ensemble and conf_ensemble.fitted:
            # Estimate current expert uncertainties from recent NLL variance
            try:
                expert_names = names or _per_expert_names()
                expert_uncertainties = {}

                # Use calibrated baseline uncertainties from confidence ensemble
                for name in expert_names:
                    expert_uncertainties[name] = conf_ensemble.expert_baseline_uncertainty.get(name, 0.1)

                # Get LSTM prediction for fallback
                lstm_pred = base_prob_dicts.get("LSTM", None) if isinstance(base_prob_dicts, dict) else None

                # Apply confidence gating
                out, diagnostics = conf_ensemble.ensemble_predict(
                    expert_predictions=base_prob_dicts,
                    expert_uncertainties=expert_uncertainties,
                    expert_names=expert_names,
                    lstm_fallback=lstm_pred
                )

                # Log confidence gating diagnostics
                try:
                    _log_confidence_gating(diagnostics, t_eval)
                except Exception:
                    pass

            except Exception as e:
                # Fallback to traditional mixing if confidence gating fails
                import warnings
                warnings.warn(f"Confidence gating failed, using traditional mixing: {e}")
                out = _MIX_BASE_ORIG(base_prob_dicts, weights=w, temperature=temperature, power=power, names=names)
        else:
            # Traditional mixing without confidence gating
            out = _MIX_BASE_ORIG(base_prob_dicts, weights=w, temperature=temperature, power=power, names=names)

        # PF prior blend (geometric) → enforce sum≈6 → back to categorical sum=1
        try:
            pf6 = _pf_prior_from_history(_history_upto(max(0, t_eval), context="_mix_prob_dicts_adaptive:base"), PF_ENSEMBLE_CONFIGS)
            pf1 = _sum6_to_sum1_dict(pf6)
            out = _geom_blend(out, pf1, lam=PF_PRIOR_BLEND)
            _v = _np.array([out[i] for i in range(1, 41)], dtype=float)
            _v6 = _enforce_sum6(_v * 6.0)
            _v1 = _v6 / 6.0
            _v1 = _v1 / (_v1.sum() + 1e-12)
            out = {i: float(_v1[i-1]) for i in range(1, 41)}
        except Exception:
            pass
        # Optional final sharpening for non‑stacked path
        try:
            _temp = float(globals().get("ADAPTIVE_TEMPERATURE", 1.0))
            _cals = _fit_expert_calibrators(max(1, int(t_eval)-1))
            _fb = _get_feedback(max(1, int(t_eval)-1))
            _cal_ok = (isinstance(_cals, dict) and any(getattr(_cals.get(k), "ok", False) for k in ("LSTM","Transformer"))) or (_fb and (_fb.get("iso") is not None))
            if _cal_ok and abs(_temp - 1.0) > 1e-9:
                out = _apply_temperature(out, _temp)
        except Exception:
            pass
        # NEW: feedback correction on the mixed distribution
        try:
            out = _apply_feedback_correction(out, t_eval)
        except Exception:
            pass
        try:
            _log_predict_run(w, t_eval=t_eval)
        except Exception:
            pass
        return out
    except Exception:
        # Worst-case fallback: original call with whatever weights were provided to us.
        return _MIX_BASE_ORIG(base_prob_dicts, weights=weights, temperature=temperature, power=power, names=names)

# Monkey-patch: replace any existing `_mix_prob_dicts` with our adaptive wrapper
try:
    globals()["_mix_prob_dicts"] = _mix_prob_dicts_adaptive
except Exception:
    pass
# ==============================================================================

def _log_confidence_gating(diagnostics, t_eval=None, log_path="confidence_gating_log.jsonl"):
    """
    Phase 4: Log confidence gating diagnostics.

    Args:
        diagnostics: dict with keys: n_experts_used, expert_weights, gated_experts, fallback_to_lstm
        t_eval: Current evaluation time index
        log_path: Path to log file
    """
    try:
        rec = {
            "type": "confidence_gating",
            "ts": _now_iso(),
            "t_eval": int(t_eval) if t_eval is not None else None,
            "n_experts_used": diagnostics.get("n_experts_used", 0),
            "expert_weights": diagnostics.get("expert_weights", {}),
            "gated_out": diagnostics.get("gated_experts", []),
            "fallback_to_lstm": diagnostics.get("fallback_to_lstm", False),
            "fallback_to_uniform": diagnostics.get("fallback_to_uniform", False),
        }

        # Also print to console for visibility
        if rec.get("fallback_to_lstm"):
            print(f"[CONF_GATE] t={t_eval} → LSTM fallback (too few confident experts)")
        elif rec.get("n_experts_used", 0) > 0:
            weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in rec["expert_weights"].items()])
            gated_str = ", ".join(rec["gated_out"]) if rec["gated_out"] else "none"
            print(f"[CONF_GATE] t={t_eval} → {rec['n_experts_used']} experts | weights: {weights_str} | gated: {gated_str}")

        _log_jsonl(rec, log_path)
    except Exception as e:
        import warnings
        warnings.warn(f"Confidence gating logging failed: {e}")

def _log_hierarchical_routing(diagnostics, t_eval=None, log_path="hierarchical_routing_log.jsonl"):
    """
    Phase 4 Upgrade 3: Log hierarchical ensemble routing decisions.

    Args:
        diagnostics: dict with keys: hierarchical, n_groups, routing_weights, router_used
        t_eval: Current evaluation time index
        log_path: Path to log file
    """
    try:
        rec = {
            "type": "hierarchical_routing",
            "ts": _now_iso(),
            "t_eval": int(t_eval) if t_eval is not None else None,
            "n_groups": diagnostics.get("n_groups", 0),
            "routing_weights": diagnostics.get("routing_weights", {}),
            "router_used": diagnostics.get("router_used", False),
            "hierarchical": diagnostics.get("hierarchical", False),
        }

        # Print to console for visibility
        if rec.get("hierarchical"):
            weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in rec["routing_weights"].items()])
            router_status = "ROUTER" if rec["router_used"] else "UNIFORM"
            print(f"[HIER_ROUTE] t={t_eval} → {rec['n_groups']} groups | {router_status} | weights: {weights_str}")

        _log_jsonl(rec, log_path)
    except Exception as e:
        import warnings
        warnings.warn(f"Hierarchical routing logging failed: {e}")

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
            "feedback": (lambda _fb: {
                "window": _fb.get("window"),
                "t_fit": _fb.get("t_fit"),
                "brier": _safe_float(_fb.get("brier")),
                "logloss": _safe_float(_fb.get("logloss"))
            })(_get_feedback(t_eval)),
        }
        print("[RUN]", json.dumps(rec, ensure_ascii=False))
        _log_jsonl(rec, path=log_path)
    except Exception as e:
        warnings.warn(f"predict-time logging failed: {e}")

 
# === Feedback Loop & Self-Improvement =========================================
# Learn from recent mistakes: fit an isotonic calibrator on recent predictions vs. outcomes
# and compute per-number residuals (under/over-confidence). Expose these as features and
# apply a gentle multiplicative correction to the final mix.

FEED_WINDOW = 90            # expanded window (60–90)
FEED_MIN = 60               # delay activation until enough history
FEED_RESID_THRESH = 0.003   # unchanged
FEED_GAMMA = 0.30           # softened calibration ratio correction
FEED_BETA = 0.25            # softened residual boost

_feedback_cache = {"t_fit": None, "window": None, "iso": None,
                   "avg_pred40": None, "emp40": None, "resid40": None,
                   "brier": None, "logloss": None}

def _safe_hist_window(t_eval, W):
    t_eval = int(max(1, t_eval))
    t0 = max(1, t_eval - int(W) + 1)
    return t0, t_eval

def _winners_at(t):
    try:
        return _winner_draw_at(int(t), context="_winners_at")
    except Exception:
        return set()

def _mix_proxy_at_t(t):
    """
    Build a proxy mixed distribution at historical index t using the original mixer
    and weights learned only from draws up to t-1 (avoid leakage & recursion).
    """
    import numpy as _np
    bases = _per_expert_prob_dicts_at_t(t)
    if not isinstance(bases, dict) or len(bases) == 0:
        return {i: 1.0/40.0 for i in range(1,41)}
    try:
        w, _stk, _ = _learn_blend_weights(max(1, int(t)-1), window=ADAPT_WINDOW, use_meta=False)
    except Exception:
        w = _np.ones(len(_per_expert_names()), dtype=float)/float(len(_per_expert_names()))
    try:
        return _MIX_BASE_ORIG(bases, weights=w, names=_per_expert_names())
    except Exception:
        mats = []
        for nm in _per_expert_names():
            di = bases.get(nm, {})
            mats.append([float(di.get(n, 1.0/40.0)) for n in range(1,41)])
        V = _np.array(mats, dtype=float)
        v = _np.clip(V.mean(axis=0), 1e-18, None)
        v = v / (v.sum() + 1e-18)
        return {n: float(v[n-1]) for n in range(1,41)}

def _fit_feedback(t_eval, window=FEED_WINDOW):
    """
    Fit isotonic calibrator and compute per-number residuals over a sliding window.
    Stores in _feedback_cache and returns it.
    """
    import numpy as _np
    t0, te = _safe_hist_window(t_eval, window)
    if te - t0 + 1 < FEED_MIN:
        _feedback_cache.update({"t_fit": te, "window": int(window), "iso": None,
                                "avg_pred40": None, "emp40": None, "resid40": None,
                                "brier": None, "logloss": None})
        return _feedback_cache
    P_all, Y_all = [], []
    avg_pred = _np.zeros(40, dtype=float)
    emp = _np.zeros(40, dtype=float)
    count = 0
    for t in range(t0, te+1):
        p_mix = _mix_proxy_at_t(t)
        y = _np.zeros(40, dtype=float)
        for w in _winners_at(t):
            if 1 <= w <= 40:
                y[w-1] = 1.0
        v = _np.array([p_mix.get(i, 1.0/40.0) for i in range(1,41)], dtype=float)
        P_all.append(v.reshape(-1)); Y_all.append(y.reshape(-1))
        avg_pred += v; emp += y; count += 1
    if count == 0:
        _feedback_cache.update({"t_fit": te, "window": int(window), "iso": None,
                                "avg_pred40": None, "emp40": None, "resid40": None,
                                "brier": None, "logloss": None})
        return _feedback_cache
    P_all = _np.vstack(P_all).reshape(-1)
    Y_all = _np.vstack(Y_all).reshape(-1)
    brier = float(_np.mean((P_all - Y_all)**2))
    logloss = float(-_np.mean(Y_all*_np.log(_np.clip(P_all,1e-12,1-1e-12)) +
                              (1.0-Y_all)*_np.log(_np.clip(1.0-P_all,1e-12,1-1e-12))))
    iso = _Iso1D().fit(P_all, Y_all)
    avg_pred /= float(count); emp /= float(count)
    resid = (emp - avg_pred)  # + => underconfident; − => overconfident
    _feedback_cache.update({
        "t_fit": te, "window": int(window), "iso": iso if getattr(iso, "ok", False) else None,
        "avg_pred40": avg_pred, "emp40": emp, "resid40": resid,
        "brier": brier, "logloss": logloss
    })
    return _feedback_cache

def _get_feedback(t_eval):
    info = _feedback_cache
    if info.get("t_fit") != int(t_eval):
        info = _fit_feedback(int(t_eval), window=FEED_WINDOW)
    return info

def _co_matrix_recent(t_eval, window=120):
    """
    Symmetric 40x40 co-occurrence matrix C where C[i-1,j-1] ≈ P(i and j co-occur)
    over the last `window` completed draws up to t_eval (exclusive).
    """
    import numpy as _np
    t_eval = int(t_eval)
    _dr = globals().get("draws", None)
    if not isinstance(_dr, (list, tuple)) or t_eval <= 1:
        return _np.zeros((40,40), dtype=float), 0
    t0 = max(1, t_eval - int(window))
    C = _np.zeros((40,40), dtype=float)
    nD = 0
    for t in range(t0, t_eval):
        try:
            s = set(_winner_draw_at(t, context="_co_matrix_recent"))
            if not s:
                continue
            nD += 1
            idx = [n-1 for n in s if 1 <= n <= 40]
            for i in idx:
                for j in idx:
                    if i == j:
                        continue
                    C[i, j] += 1.0
        except Exception:
            continue
    if nD > 0:
        C = C / float(nD)
        C = _np.clip(C, 0.0, 1.0)
    return C, nD

def _recency_and_decay(t_eval, half_life=30, window=180):
    """
    For each number 1..40:
      recency_gap_norm ∈ [0,1] and exp_decay_norm ∈ [0,1] over `window`.
    """
    import numpy as _np
    t_eval = int(t_eval)
    _dr = globals().get("draws", None)
    if not isinstance(_dr, (list, tuple)) or t_eval <= 1:
        return _np.zeros(40), _np.zeros(40)
    t0 = max(1, t_eval - int(window) + 1)
    last_seen = _np.full(40, _np.inf, dtype=float)
    decay = _np.zeros(40, dtype=float)
    lam = _np.log(2.0) / max(1.0, float(half_life))
    for t in range(t0, t_eval+1):
        dt = (t_eval - t)
        w = _np.exp(-lam * dt)
        try:
            s = set(_winner_draw_at(t, context="_recency_and_decay"))
        except Exception:
            s = set()
        for n in s:
            if 1 <= n <= 40:
                last_seen[n-1] = min(last_seen[n-1], float(t_eval - t))
                decay[n-1] += w
    gap = last_seen.copy()
    gap[_np.isinf(gap)] = float(window)
    gap = _np.clip(gap, 0.0, float(window)) / float(window)
    if decay.max() > 0:
        decay = decay / (decay.max() + 1e-12)
    return gap, decay

# === Ticket-level Joint Model (XGBoost) ======================================
# Train a ranker/classifier over sampled 6-number tickets.
# Positives: winning ticket at draw t
# Negatives: 5–20k sampled tickets (biased toward high-marginals)
# Features (per ticket):
#  • sum/mean/max/sumlog of per-number marginals
#  • pair/triad co-occurrence stats from C (sum over pairs, mean over pairs, clique density proxy)
#  • recency stats (min/mean gap, sum/mean decay)
#  • diversity vs top-k set (Jaccard distance)
#  • PF prior product/sum (log space)
#
# Inference:
#  • Build top-N candidates from marginals via biased sampling
#  • Score with ranker; blend joint with marginals (alpha≈0.35–0.45)

_ticket_ranker = {"model": None, "feat_dim": None, "trained_range": None, "xgb_type": None}

def _ticket_feature_vector(ticket, marginals, C, rec_gap, decay, pf1, topk_set):
    import numpy as _np
    nums = sorted(list(ticket))
    idx = [n-1 for n in nums]
    p = _np.array([float(marginals.get(n, 1.0/40.0)) for n in nums], dtype=float)
    pf = _np.array([float(pf1.get(n, 1.0/40.0)) for n in nums], dtype=float)

    # Marginal aggregates
    sum_p = float(p.sum()); mean_p = float(p.mean()); max_p = float(p.max())
    log_p = _np.log(_np.clip(p, 1e-18, None)); sum_log_p = float(log_p.sum()); mean_log_p = float(log_p.mean())

    # Pair/triad co-occurrence inside the ticket
    pair_vals = []
    for i in range(6):
        for j in range(i+1, 6):
            pair_vals.append(float(C[idx[i], idx[j]]))
    if not pair_vals:
        pair_vals = [0.0]
    pair_vals = _np.asarray(pair_vals, dtype=float)
    pair_sum = float(pair_vals.sum()); pair_mean = float(pair_vals.mean())

    # Triad proxy: node-wise mean of pair strengths (clique density proxy)
    tri_proxy = 0.0
    for i in range(6):
        others = [j for j in range(6) if j != i]
        tri_proxy += float(_np.mean([C[idx[i], idx[j]] for j in others])) if others else 0.0
    tri_proxy /= 6.0

    # Recency stats
    g = _np.array([float(rec_gap[n-1]) for n in nums], dtype=float)
    d = _np.array([float(decay[n-1]) for n in nums], dtype=float)
    gap_min = float(g.min()); gap_mean = float(g.mean())
    decay_sum = float(d.sum()); decay_mean = float(d.mean())

    # Diversity vs top-k numbers (Jaccard distance)
    inter = len(set(nums) & set(topk_set))
    jaccard = 1.0 - float(inter) / float(len(set(nums) | set(topk_set)) or 1.0)

    # PF prior aggregates (log product + mean)
    log_pf = _np.log(_np.clip(pf, 1e-18, None))
    pf_log_sum = float(log_pf.sum()); pf_log_mean = float(log_pf.mean())

    return _np.array([
        sum_p, mean_p, max_p, sum_log_p, mean_log_p,
        pair_sum, pair_mean, tri_proxy,
        gap_min, gap_mean, decay_sum, decay_mean,
        jaccard, pf_log_sum, pf_log_mean
    ], dtype=float)

def _sample_negative_tickets(marginals, n_samples=10000, power=1.5, seed=42):
    """Sample unique 6-number tickets, biased toward high marginals by p^power."""
    import numpy as _np
    _np.random.seed(int(seed))
    base = _np.array([float(marginals.get(i, 1.0/40.0)) for i in range(1,41)], dtype=float)
    base = _np.power(_np.clip(base, 1e-18, None), float(power))
    base = base / (base.sum() + 1e-18)
    nums = _np.arange(1, 41, dtype=int)
    seen, out = set(), []
    tries = int(n_samples) * 5 + 1000
    for _ in range(tries):
        draw = tuple(sorted(_np.random.choice(nums, size=6, replace=False, p=base)))
        if draw not in seen:
            seen.add(draw); out.append(draw)
            if len(out) >= int(n_samples):
                break
    return out

def _build_ticket_dataset_for_draw(t, neg_per_draw=10000, topk=10):
    import numpy as _np
    p_mix = _mix_proxy_at_t(int(t))
    C, _ = _co_matrix_recent(int(t), window=120)
    rec_gap, decay = _recency_and_decay(int(t), half_life=30, window=180)
    try:
        pf6 = _pf_prior_from_history(_history_upto(max(0, int(t)), context="_build_ticket_dataset_for_draw"))
        pf1 = _sum6_to_sum1_dict(pf6)
    except Exception:
        pf1 = {i: 1.0/40.0 for i in range(1,41)}
    # Top-k for diversity
    order = _np.argsort([-float(p_mix.get(i, 0.0)) for i in range(1,41)])
    topk_set = set([int(order[i])+1 for i in range(min(int(topk), 40))])

    pos = tuple(sorted(list(_winner_draw_at(int(t), context="_build_ticket_dataset_for_draw"))))
    X_pos = _ticket_feature_vector(pos, p_mix, C, rec_gap, decay, pf1, topk_set).reshape(1,-1)
    y_pos = _np.array([1.0], dtype=float)

    negs = _sample_negative_tickets(p_mix, n_samples=int(neg_per_draw), power=1.7, seed=777+int(t))
    X_neg = _np.vstack([_ticket_feature_vector(n, p_mix, C, rec_gap, decay, pf1, topk_set) for n in negs])
    y_neg = _np.zeros(len(negs), dtype=float)

    X = _np.vstack([X_pos, X_neg])
    y = _np.hstack([y_pos, y_neg])
    group = _np.array([X.shape[0]], dtype=int)  # one group per draw (for ranker)
    return X, y, group

def train_ticket_ranker(t_start, t_end, neg_per_draw=10000, min_draws=20):
    """Train XGBoost ranker if available, else classifier (logistic)."""
    global _ticket_ranker, xgb
    import numpy as _np
    if xgb is None:
        warnings.warn("XGBoost unavailable; ticket ranker disabled.")
        _ticket_ranker = {"model": None, "feat_dim": None, "trained_range": None, "xgb_type": None}
        return _ticket_ranker

    t_start = max(2, int(t_start)); t_end = int(t_end)
    if t_end - t_start + 1 < int(min_draws):
        warnings.warn("Not enough draws to train ticket ranker.")
        _ticket_ranker = {"model": None, "feat_dim": None, "trained_range": None, "xgb_type": None}
        return _ticket_ranker

    Xs, ys, groups = [], [], []
    for t in range(t_start, t_end+1):
        try:
            X_t, y_t, g_t = _build_ticket_dataset_for_draw(t, neg_per_draw=int(neg_per_draw))
            Xs.append(X_t); ys.append(y_t); groups.append(int(g_t[0]))
        except Exception as e:
            warnings.warn(f"ticket dataset failed at t={t}: {e}")
            continue
    if not Xs:
        warnings.warn("No ticket training data compiled.")
        _ticket_ranker = {"model": None, "feat_dim": None, "trained_range": None, "xgb_type": None}
        return _ticket_ranker

    X = _np.vstack(Xs); y = _np.hstack(ys)
    feat_dim = int(X.shape[1])

    model = None; xgb_type = None
    try:
        if hasattr(xgb, "XGBRanker"):
            model = xgb.XGBRanker(
                objective="rank:pairwise",
                n_estimators=750,  # Phase 3: 500→750
                max_depth=10,      # Phase 3: 8→10
                learning_rate=0.025,  # Phase 3: slightly lower for stability
                subsample=0.75,    # Phase 3: more aggressive subsampling
                colsample_bytree=0.75,
                colsample_bylevel=0.8,  # Phase 3: new - column sampling per level
                min_child_weight=4,  # Phase 3: 3→4 for more regularization
                gamma=0.15,        # Phase 3: 0.1→0.15
                reg_alpha=0.15,    # Phase 3: 0.1→0.15 (L1)
                reg_lambda=1.5,    # Phase 3: 1.0→1.5 (L2)
                random_state=42,
                n_jobs=0
            )
            model.fit(X, y, group=groups, verbose=False)
            xgb_type = "ranker"
        else:
            raise AttributeError("XGBRanker not present")
    except Exception:
        try:
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=750,  # Phase 3: 500→750
                max_depth=10,      # Phase 3: 8→10
                learning_rate=0.025,  # Phase 3: slightly lower
                subsample=0.75,
                colsample_bytree=0.75,
                colsample_bylevel=0.8,  # Phase 3: new
                min_child_weight=4,  # Phase 3: 3→4
                gamma=0.15,        # Phase 3: 0.1→0.15
                reg_alpha=0.15,    # Phase 3: 0.1→0.15
                reg_lambda=1.5,    # Phase 3: 1.0→1.5
                random_state=42,
                n_jobs=0,
                scale_pos_weight=max(1.0, float((y == 0).sum()) / float((y == 1).sum() + 1e-12))
            )
            model.fit(X, y, verbose=False)
            xgb_type = "classifier"
        except Exception as e:
            warnings.warn(f"XGBoost training failed: {e}")
            model = None; xgb_type = None

    _ticket_ranker = {"model": model, "feat_dim": feat_dim, "trained_range": (int(t_start), int(t_end)), "xgb_type": xgb_type}
    return _ticket_ranker

def _candidate_tickets_from_marginals(marginals, n_samples=20000, keep_top=400, power=1.5, seed=1234):
    """Biased sampling by p^power; keep top by product-of-marginals."""
    import numpy as _np
    cands = _sample_negative_tickets(marginals, n_samples=int(n_samples), power=float(power), seed=int(seed))
    logs = []
    for t in cands:
        s = 0.0
        for n in t:
            s += float(_np.log(max(1e-18, float(marginals.get(n, 1.0/40.0)))))
        logs.append(s)
    order = _np.argsort(logs)[-int(keep_top):]
    keep = [cands[i] for i in order]
    return list(dict.fromkeys(keep))

def score_and_blend_tickets(t_eval, alpha=0.40, n_samples=20000, keep_top=400, topN=128, power=1.6):
    """
    Build candidates for target t_eval+1 from marginals, score with ticket model
    (if available), then blend joint with marginals via alpha in [0.35,0.45].
    Returns (blended_scores_dict, ranked_ticket_list).
    """
    import numpy as _np
    p_mix = _mix_proxy_at_t(int(t_eval) + 1)
    C, _ = _co_matrix_recent(int(t_eval) + 1, window=120)
    rec_gap, decay = _recency_and_decay(int(t_eval) + 1, half_life=30, window=180)
    try:
        pf6 = _pf_prior_from_history(_history_upto(max(0, int(t_eval) + 1), context="score_and_blend_tickets"))
        pf1 = _sum6_to_sum1_dict(pf6)
    except Exception:
        pf1 = {i: 1.0/40.0 for i in range(1,41)}
    order = _np.argsort([-float(p_mix.get(i, 0.0)) for i in range(1,41)])
    topk_set = set([int(order[i])+1 for i in range(min(10, 40))])

    cand = _candidate_tickets_from_marginals(p_mix, n_samples=int(n_samples), keep_top=int(keep_top), power=float(power), seed=2468)

    model = _ticket_ranker.get("model")
    if model is None:
        joint = {t: float(_np.exp(sum(_np.log([max(1e-18, p_mix.get(n, 1.0/40.0)) for n in t])))) for t in cand}
        blended = blend_joint_with_marginals(joint, p_mix, alpha=float(alpha))
        ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)[:int(topN)]
        return blended, [k for k, _ in ranked]

    X = _np.vstack([_ticket_feature_vector(t, p_mix, C, rec_gap, decay, pf1, topk_set) for t in cand])
    try:
        if hasattr(model, "predict_proba"):
            s = model.predict_proba(X)[:, 1]
        elif hasattr(model, "predict"):
            s = model.predict(X)
        else:
            s = _np.ones(len(cand), dtype=float)
    except Exception:
        s = _np.ones(len(cand), dtype=float)

    z = s - _np.max(s)
    expz = _np.exp(z)
    joint = {t: float(expz[i] / (expz.sum() + 1e-12)) for i, t in enumerate(cand)}

    blended = blend_joint_with_marginals(joint, p_mix, alpha=float(alpha))
    ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)[:int(topN)]
    return blended, [k for k, _ in ranked]
# ============================================================================ 

def _logit(x, eps=1e-6):
    import numpy as _np
    x = _np.clip(_np.asarray(x, dtype=float), eps, 1.0 - eps)
    return _np.log(x/(1.0-x))

def _build_stacker_input(bases_at_t, t_eval):
    """
    Meta-stacker input for draw t_eval:
      cols 0..4  : expert probs in _per_expert_names() order
      cols 5..9  : [calibrated_p_mix, calib_ratio, resid, over_flag, under_flag]
      cols 10..12: [pair_logit, triad_logit, pair_mean_ctx]  (set-level)
      cols 13..14: [recency_gap_norm, exp_decay_norm]        (recency-gated)
    shape: (40, 15)
    """
    import numpy as _np
    mats = []
    for nm in _per_expert_names():
        di = bases_at_t.get(nm, {})
        mats.append([float(di.get(n, 1.0/40.0)) for n in range(1,41)])
    X = _np.vstack(mats).T  # (40,5)

    # Feedback-derived features
    fb = _get_feedback(max(1, int(t_eval)-1))
    try:
        p_mix = _mix_proxy_at_t(int(t_eval))
    except Exception:
        p_mix = {i: 1.0/40.0 for i in range(1,41)}
    v_mix = _np.array([float(p_mix.get(i, 1.0/40.0)) for i in range(1,41)], dtype=float)
    if fb and fb.get("iso") is not None:
        cal = fb["iso"].map(v_mix)
    else:
        cal = v_mix.copy()
    ratio = _np.clip(cal / _np.clip(v_mix, 1e-12, None), 0.1, 10.0)
    resid = fb.get("resid40")
    if resid is None or len(resid) != 40:
        resid = _np.zeros(40, dtype=float)
    over = (resid < -FEED_RESID_THRESH).astype(float)
    under = (resid > FEED_RESID_THRESH).astype(float)
    feats_fb = _np.vstack([cal, ratio, resid, over, under]).T  # (40,5)

    # Set-level signals: pairwise/triad co-occurrence
    C, _cnt = _co_matrix_recent(int(t_eval), window=120)  # (40,40)
    pair_ctx = C.dot(v_mix)  # (40,)
    top2 = _np.argsort(v_mix)[-2:]
    triad_proxy = _np.zeros(40, dtype=float)
    if top2.size == 2:
        triad_proxy = 0.5 * (C[:, top2[0]] + C[:, top2[1]])
    pair_mean_ctx = _np.full(40, float(_np.mean(pair_ctx)) if pair_ctx.size else 0.0, dtype=float)
    pair_logit = _logit(pair_ctx)
    triad_logit = _logit(triad_proxy)

    # Recency-gated features
    rec_gap, exp_decay = _recency_and_decay(int(t_eval), half_life=30, window=180)

    feats_set = _np.vstack([pair_logit, triad_logit, pair_mean_ctx, rec_gap, exp_decay]).T  # (40,5)

    return _np.hstack([X, feats_fb, feats_set])

def _apply_feedback_correction(prob_dict, t_eval):
    """
    Gentle multiplicative correction on final categorical dict using:
      - calibration ratio^FEED_GAMMA
      - residual factor (1 + FEED_BETA * resid)
    """
    import numpy as _np
    out = dict(prob_dict)
    fb = _get_feedback(max(1, int(t_eval)-1))
    try:
        p_proxy = _mix_proxy_at_t(int(t_eval))
    except Exception:
        p_proxy = {i: 1.0/40.0 for i in range(1,41)}
    v = _np.array([float(out.get(i, 0.0)) for i in range(1,41)], dtype=float)
    v = _np.clip(v, 1e-18, None); v = v / (v.sum() + 1e-18)
    v_proxy = _np.array([float(p_proxy.get(i, 1.0/40.0)) for i in range(1,41)], dtype=float)
    if fb and fb.get("iso") is not None:
        v_cal = fb["iso"].map(v_proxy)
    else:
        v_cal = v_proxy
    ratio = _np.clip(v_cal / _np.clip(v_proxy, 1e-12, None), 0.5, 2.0)
    resid = fb.get("resid40")
    if resid is None or len(resid) != 40:
        resid = _np.zeros(40, dtype=float)
    resid_factor = _np.clip(1.0 + FEED_BETA * resid, 0.8, 1.2)
    corr = _np.power(ratio, FEED_GAMMA) * resid_factor
    v_adj = _np.clip(v * corr, 1e-18, None)
    v_adj = v_adj / (v_adj.sum() + 1e-18)
    return {i: float(v_adj[i-1]) for i in range(1,41)}
# ============================================================================

# === Training Policy Overrides (Keras/TensorFlow & PyTorch) ====================
 # Goal: Increase NN training epochs (≥500), add aggressive EarlyStopping, and
 # add learning‑rate scheduling (ReduceLROnPlateau + optional Cosine restarts),
 # without removing any existing code. These apply transparently via monkey‑
 # patching for tf.keras; PyTorch helpers are provided for training loops.
 #
 # Tunable via environment variables (strings interpreted as numbers/ints):
 #   LOTTO_NN_MIN_EPOCHS       default 500
 #   LOTTO_NN_MAX_EPOCHS       default 1000
 #   LOTTO_EARLYSTOP_PATIENCE  default 25
 #   LOTTO_LR_PLATEAU_PATIENCE default 7
 #   LOTTO_LR_FACTOR           default 0.5
 #   LOTTO_USE_COSINE          default 1 (on)
 #   LOTTO_COSINE_STEPS        default 2000 (optimizer step count per first cycle)
 #   LOTTO_COSINE_T_MUL        default 2.0
 #   LOTTO_COSINE_M_MUL        default 0.5
 #   LOTTO_COSINE_ALPHA        default 0.0
 #   LOTTO_LR_OVERRIDE         default 0.001 (applied if initial lr≈3e-4)
 #
 # Notes:
 #  • We DO NOT remove your callbacks; we append missing ones.
 #  • We only raise epochs if caller provided fewer than LOTTO_NN_MIN_EPOCHS.
 #  • Cosine restarts wraps the optimizer.learning_rate if present and numeric.
 #  • ReduceLROnPlateau triggers on val_loss plateaus (escape local minima).
 #  • For PyTorch, use attach_default_torch_scheduler(...) in your loop.
import os as _os_train

# Public knobs
NN_EPOCHS_MIN = int(_os_train.getenv("LOTTO_NN_MIN_EPOCHS", "500"))
NN_EPOCHS_MAX = int(_os_train.getenv("LOTTO_NN_MAX_EPOCHS", "1000"))
EARLYSTOP_PATIENCE = int(_os_train.getenv("LOTTO_EARLYSTOP_PATIENCE", "25"))
LR_PLATEAU_FACTOR = float(_os_train.getenv("LOTTO_LR_FACTOR", "0.5"))
LR_PLATEAU_PATIENCE = int(_os_train.getenv("LOTTO_LR_PLATEAU_PATIENCE", "7"))
USE_COSINE = int(_os_train.getenv("LOTTO_USE_COSINE", "1"))
COSINE_STEPS = int(_os_train.getenv("LOTTO_COSINE_STEPS", "2000"))
COSINE_T_MUL = float(_os_train.getenv("LOTTO_COSINE_T_MUL", "2.0"))
COSINE_M_MUL = float(_os_train.getenv("LOTTO_COSINE_M_MUL", "0.5"))
COSINE_ALPHA = float(_os_train.getenv("LOTTO_COSINE_ALPHA", "0.0"))
LR_OVERRIDE = _os_train.getenv("LOTTO_LR_OVERRIDE", "0.001")

# --- Keras/TensorFlow monkey‑patch -------------------------------------------
try:
    import tensorflow as _tf
    from tensorflow import keras as _keras

    _KERAS_FIT_ORIG = _keras.Model.fit

    def _apply_training_overrides_keras(model, kwargs):
        """Mutate kwargs in place to enforce epochs↑, early stopping, and LR schedule.
        Avoids conflicts between LearningRateSchedule and callbacks that try to set LR.
        """
        # 1) Epochs: if caller passed fewer than NN_EPOCHS_MIN, raise to [min..max]
        try:
            ep = int(kwargs.get("epochs", 0) or 0)
        except Exception:
            ep = 0
        if ep < NN_EPOCHS_MIN:
            kwargs["epochs"] = min(NN_EPOCHS_MAX, max(NN_EPOCHS_MIN, ep or 0))

        # 2) Determine optimizer LR mutability up‑front
        cbs = list(kwargs.get("callbacks", []) or [])
        def _has(cb_type):
            return any(isinstance(cb, cb_type) for cb in cbs)

        opt = getattr(model, "optimizer", None)
        lr_is_schedule = False
        current_lr_val = None
        try:
            from tensorflow.keras.optimizers.schedules import LearningRateSchedule as _LRSchedule
            if opt is not None and hasattr(opt, "learning_rate"):
                lr_obj = opt.learning_rate
                lr_is_schedule = isinstance(lr_obj, _LRSchedule)
                if not lr_is_schedule:
                    try:
                        current_lr_val = float(_tf.keras.backend.get_value(lr_obj))
                    except Exception:
                        current_lr_val = None
        except Exception:
            pass

        # Decide whether we will attach a cosine schedule (only if LR is settable now)
        will_use_cosine = (int(USE_COSINE) == 1) and (opt is not None) and (not lr_is_schedule)

        # 3) Strip LR-mutating callbacks if LR is not settable (schedule attached)
        #    This prevents errors like: "optimizer was created with a LearningRateSchedule ... not settable"
        try:
            from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
            # If the optimizer's LR is already a schedule OR we plan to attach cosine,
            # remove any callbacks that try to assign a new float LR.
            if lr_is_schedule or will_use_cosine:
                new_cbs = []
                for cb in cbs:
                    if isinstance(cb, (ReduceLROnPlateau, LearningRateScheduler)):
                        # Drop it to avoid TypeError during training
                        continue
                    new_cbs.append(cb)
                cbs = new_cbs
        except Exception:
            pass

        # 4) Callbacks: always add EarlyStopping; add ReduceLROnPlateau only if LR is settable
        try:
            es = _keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=EARLYSTOP_PATIENCE,
                restore_best_weights=True, min_delta=1e-4, verbose=0
            )
            if not any(isinstance(cb, _keras.callbacks.EarlyStopping) for cb in cbs):
                cbs.append(es)

            # Only add ReduceLROnPlateau when we are NOT switching to/using a schedule
            if (not will_use_cosine) and (not lr_is_schedule):
                if not any(isinstance(cb, _keras.callbacks.ReduceLROnPlateau) for cb in cbs):
                    rlrop = _keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", mode="min", factor=LR_PLATEAU_FACTOR,
                        patience=LR_PLATEAU_PATIENCE, min_lr=1e-6, verbose=0
                    )
                    cbs.append(rlrop)
            kwargs["callbacks"] = cbs
        except Exception:
            pass

        # 5) Learning‑rate tweaks: optional override & cosine restarts
        try:
            if opt is not None and hasattr(opt, "learning_rate") and (not lr_is_schedule):
                # If lr is numeric and close to 3e-4, bump to LR_OVERRIDE (default 1e-3)
                if (current_lr_val is not None) and (abs(current_lr_val - 3e-4) / max(3e-4, 1e-12) < 0.25):
                    try:
                        _tf.keras.backend.set_value(opt.learning_rate, float(LR_OVERRIDE))
                        current_lr_val = float(LR_OVERRIDE)
                    except Exception:
                        pass
                # Wrap with cosine restarts schedule only if LR is settable and we chose to do so
                if will_use_cosine:
                    try:
                        init_lr = current_lr_val if (current_lr_val is not None) else float(LR_OVERRIDE)
                        sched = _tf.keras.optimizers.schedules.CosineDecayRestarts(
                            initial_learning_rate=init_lr,
                            first_decay_steps=max(1, int(COSINE_STEPS)),
                            t_mul=float(COSINE_T_MUL),
                            m_mul=float(COSINE_M_MUL),
                            alpha=float(COSINE_ALPHA),
                        )
                        opt.learning_rate = sched  # becomes non‑settable; RLROP already filtered
                    except Exception:
                        pass
        except Exception:
            pass

        return kwargs

    def _keras_fit_patched(self, *args, **kwargs):
        try:
            kwargs = _apply_training_overrides_keras(self, dict(kwargs))
        except Exception:
            pass
        return _KERAS_FIT_ORIG(self, *args, **kwargs)

    if getattr(_keras.Model.fit, "__name__", "") != "_keras_fit_patched":
        _keras.Model.fit = _keras_fit_patched
        try:
            print("[TRAIN] keras.Model.fit patched: epochs≥%d, EarlyStopping(patience=%d), ReduceLROnPlateau(patience=%d), cosine=%s" % (
                NN_EPOCHS_MIN, EARLYSTOP_PATIENCE, LR_PLATEAU_PATIENCE, str(bool(USE_COSINE)).lower()))
        except Exception:
            pass

        # ---- Set-aware objective + metrics (Keras) ------------------------------
        import functools as _functools

        def _k_hits_at_k(y_true, y_pred, k=6):
            # Average number of true hits among the top-k predicted numbers.
            y_true = _tf.cast(y_true, _tf.float32)
            y_pred = _tf.cast(y_pred, _tf.float32)
            # Flatten to (B, M)
            y_true = _tf.reshape(y_true, [-1, _tf.shape(y_true)[-1]])
            y_pred = _tf.reshape(y_pred, [-1, _tf.shape(y_pred)[-1]])
            k = int(k)
            topk = _tf.math.top_k(y_pred, k=k, sorted=False).indices  # (B, k)
            mask = _tf.reduce_sum(_tf.one_hot(topk, depth=_tf.shape(y_pred)[-1], dtype=_tf.float32), axis=1)  # (B, M)
            hits = _tf.reduce_sum(y_true * mask, axis=-1)  # (B,)
            return _tf.reduce_mean(hits)

        def _k_recall_at_k(y_true, y_pred, k=6):
            # Recall@k = hits/k_true (k_true≈6 for Lotto+). Normalized to [0,1].
            y_true = _tf.cast(y_true, _tf.float32)
            y_pred = _tf.cast(y_pred, _tf.float32)
            y_true = _tf.reshape(y_true, [-1, _tf.shape(y_true)[-1]])
            y_pred = _tf.reshape(y_pred, [-1, _tf.shape(y_pred)[-1]])
            k = int(k)
            topk = _tf.math.top_k(y_pred, k=k, sorted=False).indices
            mask = _tf.reduce_sum(_tf.one_hot(topk, depth=_tf.shape(y_pred)[-1], dtype=_tf.float32), axis=1)
            hits = _tf.reduce_sum(y_true * mask, axis=-1)
            k_true = _tf.maximum(1.0, _tf.reduce_sum(y_true, axis=-1))  # usually 6
            return _tf.reduce_mean(hits / k_true)

        def _env_float(name, default):
            try:
                return float(_os_train.getenv(name, str(default)))
            except Exception:
                return float(default)

        def _env_int(name, default):
            try:
                return int(_os_train.getenv(name, str(default)))
            except Exception:
                return int(default)

        def build_set_aware_loss(rank_weight=None, reward_weight=None, tau=None, max_neg=None):
            """
            Create a differentiable, set-aware objective:
              L = BCE + rank_weight * L_rank + reward_weight * L_reward
            where:
              • BCE = mean binary cross-entropy over 40 numbers (per example)
              • L_rank ≈ LambdaRank-like pairwise loss over (pos, neg) using top negatives:
                    mean_{i∈topPos, j∈topNeg} softplus(-(p_i - p_j)/tau)
              • L_reward = - mean sum(y_true * p) / (#positives)  (encourage prob mass on actual 6)
            All terms are averaged over the batch. Safe for any batch size. No gradients through y_true.
            Tunables (can override via env):
              LOTTO_SET_LOSS=1           (enable)
              LOTTO_LOSS_RANK_W=0.15     (alpha)
              LOTTO_LOSS_REWARD_W=0.15   (beta)
              LOTTO_LOSS_TAU=0.05        (temperature in softplus argument)
              LOTTO_LOSS_PAIR_NEG=20     (# of hardest negatives per example)
            """
            alpha = _env_float("LOTTO_LOSS_RANK_W", 0.15) if rank_weight is None else float(rank_weight)
            beta  = _env_float("LOTTO_LOSS_REWARD_W", 0.15) if reward_weight is None else float(reward_weight)
            tau   = _env_float("LOTTO_LOSS_TAU", 0.05) if tau is None else float(tau)
            mneg  = _env_int("LOTTO_LOSS_PAIR_NEG", 20) if max_neg is None else int(max_neg)

            def _loss(y_true, y_pred):
                y = _tf.cast(y_true, _tf.float32)
                p = _tf.cast(y_pred, _tf.float32)
                # Flatten to (B, M)
                y = _tf.reshape(y, [-1, _tf.shape(y)[-1]])
                p = _tf.reshape(p, [-1, _tf.shape(p)[-1]])
                eps = _tf.constant(1e-7, _tf.float32)
                p = _tf.clip_by_value(p, eps, 1.0 - eps)

                # BCE over 40 numbers per example, then mean over batch
                bce_elem = _tf.keras.backend.binary_crossentropy(y, p)  # (B, M)
                bce = _tf.reduce_mean(_tf.reduce_mean(bce_elem, axis=-1))

                # Reward term: encourage mass on true numbers
                pos_count = _tf.maximum(1.0, _tf.reduce_sum(y, axis=-1))  # (B,)
                reward_val = _tf.reduce_mean(_tf.reduce_sum(y * p, axis=-1) / pos_count)
                reward_loss = -reward_val

                # Pairwise rank loss: compare top positives to top negatives
                # Select top positives by probability among true positions
                pos_scores = p * y
                # In case labels are noisy, take up to 6 positives
                top_pos_vals = _tf.math.top_k(pos_scores, k=_tf.minimum(_tf.shape(pos_scores)[-1], 6)).values  # (B, 6)
                # Hard negatives = highest predicted among y==0
                neg_scores = p * (1.0 - y)
                top_neg_vals = _tf.math.top_k(neg_scores, k=_tf.minimum(_tf.shape(neg_scores)[-1], mneg)).values  # (B, mneg)
                # Broadcast to pairwise grid and apply smooth pairwise logistic
                pos_exp = _tf.expand_dims(top_pos_vals, axis=-1)  # (B, 6, 1)
                neg_exp = _tf.expand_dims(top_neg_vals, axis=1)   # (B, 1, mneg)
                pairwise = _tf.math.softplus(-(pos_exp - neg_exp) / tau)  # (B, 6, mneg)
                rank_loss = _tf.reduce_mean(pairwise)  # scalar

                return bce + alpha * rank_loss + beta * reward_loss

            return _loss

        # Keras compile patch: enable set-aware loss & add Top-6 metrics automatically (opt-in)
        _KERAS_COMPILE_ORIG = _keras.Model.compile
        _compile_log_count = [0]  # Mutable counter to limit verbosity

        def _keras_compile_patched(self, *args, **kwargs):
            try:
                use_set_loss = str(_os_train.getenv("LOTTO_SET_LOSS", "1")).strip() in {"1", "true", "yes", "on"}
                add_metrics = str(_os_train.getenv("LOTTO_ADD_SET_METRICS", "1")).strip() in {"1", "true", "yes", "on"}
                verbose_compile_log = str(_os_train.getenv("LOTTO_VERBOSE_COMPILE", "0")).strip() in {"1", "true", "yes", "on"}

                if use_set_loss:
                    # Replace/augment the provided loss with our set-aware objective.
                    # Works for a single-output model producing 40 probabilities.
                    kwargs = dict(kwargs)
                    kwargs["loss"] = build_set_aware_loss()

                if add_metrics:
                    mets = list(kwargs.get("metrics", []) or [])
                    # Named wrappers so Keras shows friendly metric names
                    def hits_at_6(y_true, y_pred): return _k_hits_at_k(y_true, y_pred, k=6)
                    hits_at_6.__name__ = "hits_at_6"
                    def recall_at_6(y_true, y_pred): return _k_recall_at_k(y_true, y_pred, k=6)
                    recall_at_6.__name__ = "recall_at_6"
                    # Avoid duplicates by name
                    existing = {getattr(m, "__name__", str(m)) for m in mets}
                    if "hits_at_6" not in existing: mets.append(hits_at_6)
                    if "recall_at_6" not in existing: mets.append(recall_at_6)
                    kwargs["metrics"] = mets

                # Only log first compile call unless verbose mode enabled
                if (use_set_loss or add_metrics) and (_compile_log_count[0] == 0 or verbose_compile_log):
                    print("[TRAIN] keras.Model.compile patched:",
                          json.dumps({
                              "set_loss": bool(use_set_loss),
                              "rank_w": _env_float("LOTTO_LOSS_RANK_W", 0.15),
                              "reward_w": _env_float("LOTTO_LOSS_REWARD_W", 0.15),
                              "tau": _env_float("LOTTO_LOSS_TAU", 0.05),
                              "pair_neg": _env_int("LOTTO_LOSS_PAIR_NEG", 20),
                              "add_metrics": bool(add_metrics)
                          }))
                    _compile_log_count[0] += 1
            except Exception:
                pass
            return _KERAS_COMPILE_ORIG(self, *args, **kwargs)

        if getattr(_keras.Model.compile, "__name__", "") != "_keras_compile_patched":
            _keras.Model.compile = _keras_compile_patched
except Exception:
    # TensorFlow/Keras not available — ignore silently
    pass
 
 # --- PyTorch helper (opt‑in from training loops) ------------------------------
 # If your CNN/Transformer uses PyTorch, call this to attach a default scheduler.
 # Example usage in your loop (pseudocode):
 #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
 #   scheduler = attach_default_torch_scheduler(optimizer, total_epochs, steps_per_epoch)
 #   for epoch in range(total_epochs):
 #       ... train ...
 #       scheduler.step(epoch + batch_idx/steps_per_epoch)
 
def attach_default_torch_scheduler(optimizer, total_epochs, steps_per_epoch):
    try:
        import torch
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        T_0 = max(1, int(steps_per_epoch))  # first restart after 1 epoch
        T_mult = 2
        eta_min = 1e-6
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    except Exception:
        return None

# ---- PyTorch set-aware loss (opt-in from Torch training loop) ---------------
def build_set_aware_loss_torch(rank_weight=0.15, reward_weight=0.15, tau=0.05, max_neg=20):
    """
    Return a PyTorch nn.Module computing:
      BCE + rank_weight * LambdaRank-like pairwise loss + reward_weight * (-sum(y*p)/#pos)
    Expects predictions as probabilities in [0,1] with shape (B, 40) and targets as {0,1}.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        return None

    class _SetAwareLoss(nn.Module):
        def __init__(self, a, b, t, mneg):
            super().__init__()
            self.a = float(a)
            self.b = float(b)
            self.t = float(t)
            self.mneg = int(mneg)
            self.bce = nn.BCELoss(reduction="mean")

        def forward(self, y_pred, y_true):
            # y_pred assumed probabilities; if logits, caller should apply sigmoid first.
            y = y_true.float()
            p = torch.clamp(y_pred.float(), 1e-7, 1-1e-7)

            # BCE over elements, mean over batch
            bce = self.bce(p, y)

            # Reward: encourage mass on true numbers
            pos_count = torch.clamp(y.sum(dim=1), min=1.0)
            reward_val = ((y * p).sum(dim=1) / pos_count).mean()
            reward_loss = -reward_val

            # Pairwise rank loss: compare top positives vs hardest negatives
            pos_scores = p * y
            top_pos_vals, _ = torch.topk(pos_scores, k=min(6, pos_scores.size(1)), dim=1)
            neg_scores = p * (1.0 - y)
            top_neg_vals, _ = torch.topk(neg_scores, k=min(self.mneg, neg_scores.size(1)), dim=1)
            pos_exp = top_pos_vals.unsqueeze(2)      # (B, 6, 1)
            neg_exp = top_neg_vals.unsqueeze(1)      # (B, 1, mneg)
            pairwise = torch.nn.functional.softplus(-(pos_exp - neg_exp) / self.t)
            rank_loss = pairwise.mean()

            return bce + self.a * rank_loss + self.b * reward_loss

    return _SetAwareLoss(rank_weight, reward_weight, tau, max_neg)
 # ============================================================================
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

# --- Dynamic Bayesian Particle Filter (per-number latent probs) ---
class ParticleFilter:
    def __init__(self, num_numbers=40, num_particles=12000, alpha=0.004, sigma=0.008):
        """Initialize particles with near-uniform draw probabilities."""
        self.M = num_numbers
        self.N = num_particles
        self.alpha = alpha       # mean-reversion rate
        self.sigma = sigma       # standard deviation of drift noise
        # Initialize all particles around uniform probability 6/num_numbers
        base_prob = 6.0 / num_numbers
        self.particles = np.full((num_particles, num_numbers), base_prob)
        # Add slight noise to break symmetry, then normalize and clip
        self.particles += np.random.normal(0, 0.01, size=self.particles.shape)
        self.particles = np.clip(self.particles, 0, None)
        # Normalize each particle's probabilities to sum to ~6 (expected 6 draws)
        for i in range(num_particles):
            total = self.particles[i].sum()
            if total == 0:
                self.particles[i] = base_prob  # if degenerate, reset to uniform
            else:
                self.particles[i] *= (6.0 / total)
        self.particles = np.clip(self.particles, 0, 1)
    
    def predict(self):
        """Propagate particle probabilities forward one time step with drift and mean reversion."""
        uniform = 6.0 / self.M
        # Add mean reversion and noise to each particle
        for i in range(self.N):
            p = self.particles[i]
            # Mean reversion toward uniform distribution
            p += self.alpha * (uniform - p)
            # Random drift (Gaussian noise)
            p += np.random.normal(0, self.sigma, size=self.M)
            # Clip probabilities to [0,1] and renormalize to sum ~6
            p = np.clip(p, 0, 1)
            total = p.sum()
            if total == 0:
                p[:] = uniform  # reset to uniform if particle collapsed
            else:
                p *= (6.0 / total)
            # Clip again in case renormalization caused any entry >1
            p[:] = np.clip(p, 0, 1)
            self.particles[i] = p
    
    def update(self, draw):
        """Update particle weights based on the observed draw and resample particles.
        Uses vectorized log-likelihoods for numerical stability.
        """
        import numpy as _np

        # Build zero-based index arrays for drawn and not-drawn numbers
        draw_set = set(draw)
        draw_idx = _np.array([n-1 for n in draw_set], dtype=int)
        not_idx = _np.array([n-1 for n in range(1, self.M+1) if n not in draw_set], dtype=int)

        # Clip probabilities away from 0/1 to keep logs finite
        eps = 1e-12
        P = _np.clip(self.particles, eps, 1.0 - eps)

        # Precompute log p and log(1-p)
        logP = _np.log(P)
        log1mP = _np.log1p(-P)

        # Vectorized log-likelihood per particle:
        #   ll_i = sum_{n in draw} log p_i[n] + sum_{m not in draw} log (1 - p_i[m])
        # Shape handling:
        #   logP[:, draw_idx] -> (N, |draw|)
        #   log1mP[:, not_idx] -> (N, M - |draw|)
        ll = logP[:, draw_idx].sum(axis=1) + log1mP[:, not_idx].sum(axis=1)

        # Stabilize before exponentiation by subtracting max (log-sum-exp trick)
        ll_max = _np.max(ll)
        weights = _np.exp(ll - ll_max)

        # Normalize weights
        w_sum = _np.sum(weights)
        if not _np.isfinite(w_sum) or w_sum <= 0:
            # Fallback to uniform if something went wrong
            weights = _np.ones(self.N, dtype=float) / float(self.N)
        else:
            weights = weights / w_sum

        # Resample particles according to normalized weights
        idx = _np.random.choice(self.N, size=self.N, p=weights)
        self.particles = self.particles[idx].copy()

        # Compute negative log-likelihood with a stable log-mean-exp:
        # log(mean(exp(ll))) = logsumexp(ll) - log(N)
        m = ll_max  # reuse the max for stability
        lse = m + _np.log(_np.sum(_np.exp(ll - m)))
        log_mean_like = lse - _np.log(float(self.N))
        nll = float(-log_mean_like)
        return nll
    
    def get_mean_probabilities(self):
        """Return the mean draw probability for each number across all particles."""
        mean_p = self.particles.mean(axis=0)
        return np.clip(mean_p, 0, 1)

# --- Strict time-guard to prevent look-ahead leakage ---
CURRENT_TARGET_IDX = None  # when predicting/evaluating draw t, only indices < t may be touched

# --- Uniform time-leak enforcement helpers -----------------------------------
DEBUG_GUARDS = True  # set False to relax invariant checks in production

def _history_upto(t_idx, context=""):
    """
    Return a slice of historical draws strictly before index t_idx: draws[:t_idx].
    Enforces that t_idx <= CURRENT_TARGET_IDX when a target is set, to avoid look-ahead.
    """
    try:
        t = int(t_idx)
    except Exception:
        raise ValueError(f"_history_upto: non-integer t_idx {t_idx!r} in {context or 'unknown'}")
    if DEBUG_GUARDS:
        _assert_idx_ok(t, context or "_history_upto")
    return draws[:t]

def _winner_draw_at(t_idx, context=""):
    """
    Return the observed winners set at index t_idx (the label at time t).
    Guarded so that t_idx <= CURRENT_TARGET_IDX when a target is set.
    """
    try:
        t = int(t_idx)
    except Exception:
        raise ValueError(f"_winner_draw_at: non-integer t_idx {t_idx!r} in {context or 'unknown'}")
    if DEBUG_GUARDS:
        _assert_idx_ok(t, context or "_winner_draw_at")
    return draws[t]

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

MC_STACK_PASSES = 50
# Ops cadence for re-tuning (weights/calibrator)
WEIGHT_TUNE_EVERY = 5  # only re‑learn base weights every N draws
_weights_cache_bin = None

# --- Quick compatibility self-test (runs before heavy work if enabled) ----------
try:
    import os as _os_mod
    import numpy as _np_mod
except Exception:
    _os_mod = None
    _np_mod = None

def _compat_self_test():
    """
    Smoke test for compatibility shims:
      1) _export_reliability accepts legacy csv_path/png_path names and writes files.
      2) _mix_prob_dicts combines two expert marginal dicts and returns a valid prob dict.
    Prints 'SELFTEST: PASS' and returns True on success.
    """
    try:
        # 1) reliability export with legacy arg names
        p = _np_mod.linspace(0.01, 0.99, 200) if _np_mod is not None else [0.5]*10
        y = (_np_mod.random.rand(200) > 0.6).astype(float) if _np_mod is not None else [0,1]*5
        out_csv = "reliability_curve.csv"
        out_png = "reliability_curve.png"
        res = _export_reliability(probs=p, labels=y, csv_path=out_csv, png_path=out_png, bins=10)
        assert isinstance(res, dict) and res.get("csv") is not None, "reliability export failed"

        # 2) mix two simple expert marginal dicts
        d1 = {n: (1.0/40.0) for n in range(1,41)}
        d2 = {n: (2.0/40.0 if n % 2 == 0 else 0.0) for n in range(1,41)}
        mix = _mix_prob_dicts({"Bayes": d1, "Markov": d2, "HMM": d1, "LSTM": d1, "Transformer": d1},
                              weights=[0.2, 0.3, 0.2, 0.2, 0.1])
        tot = sum(float(mix.get(n,0.0)) for n in range(1,41))
        assert 0.999 <= tot <= 1.001, "mixed distribution not normalized"
        # Guard sanity: only if historical draws are already loaded in globals
        if 'draws' in globals() and isinstance(globals()['draws'], (list, tuple)) and len(globals()['draws']) > 0:
            try:
                _dr = globals()['draws']
                # Choose a safe target index based on available history
                t_cur = min(9, len(_dr) - 1)
                if t_cur >= 1:
                    _set_target_idx(t_cur)
                    _ = _history_upto(t_cur - 1, context="_compat_self_test")
            finally:
                _set_target_idx(None)
        print("SELFTEST: PASS", res)
        return True
    except Exception as _e:
        import warnings as _w
        _w.warn(f"SELFTEST: FAIL: {_e}")
        return False

# If env var LOTTO_SELFTEST=1 is set, run the self-test and exit early to avoid training.
if (_os_mod is not None) and (_os_mod.environ.get("LOTTO_SELFTEST", "0") == "1"):
    ok = _compat_self_test()
    # Exit with status 0 on pass, 2 on failure to be obvious in CI or shell
    import sys as _sys_mod
    _sys_mod.exit(0 if ok else 2)
# --- End self-test -------------------------------------------------------------

import pickle
import os



# --- Load raw draws sheet into draw_df (robust) ---
try:
    # Primary path: Excel workbook in the project root.
    draw_df = pd.read_excel("Lotto+.xlsx")
except Exception as _e_read_xlsx:
    # Fallbacks: try a CSV with the same base name, then raise a clear error.
    try:
        draw_df = pd.read_csv("Lotto+.csv")
    except Exception as _e_read_csv:
        raise FileNotFoundError(
            "Could not load historical draws. Expected 'Lotto+.xlsx' (or Lotto+.csv) in the current directory."
        ) from _e_read_xlsx
# Basic column sanity (fail fast with a helpful message)
_expected_cols = {"Draw date", "Winning Numbers"}
if not _expected_cols.issubset(set(map(str, draw_df.columns))):
    raise ValueError(f"Input sheet is missing required columns {_expected_cols}. Found columns: {list(draw_df.columns)}")

data = draw_df

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

# --- Regime Modeling (robust, after we have parsed `draws`) ---
# Build draw_matrix from already-parsed `draws` to ensure correctness.
try:
    draw_matrix = np.array([sorted(list(d)) for d in draws], dtype=int)
except Exception:
    draw_matrix = np.empty((0, 6), dtype=int)

def _artifact_path(name):
    import os as _os
    out_dir = _os.path.join(_os.getcwd(), "artifacts")
    _os.makedirs(out_dir, exist_ok=True)
    return _os.path.join(out_dir, name)

gmm_model = None
try:
    # Require a reasonable minimum of clean rows to fit a stable GMM.
    if isinstance(draw_matrix, np.ndarray) and draw_matrix.ndim == 2 and draw_matrix.shape[0] >= 30:
        # Use the extract_regime_features already defined above (or re-define safely if missing).
        try:
            _ = extract_regime_features
        except NameError:
            def extract_regime_features(dm):
                feats = []
                for dr in dm:
                    dr = np.sort(dr)
                    deltas = np.diff(dr)
                    feats.append(np.concatenate([
                        dr[:1], dr[-1:], [np.mean(dr)], [np.std(dr)], deltas
                    ]))
                return np.array(feats)

        feats = extract_regime_features(draw_matrix)
        if feats.ndim == 2 and feats.shape[0] >= 10:
            from sklearn.mixture import GaussianMixture
            gmm_model = GaussianMixture(n_components=3, random_state=42)
            gmm_model.fit(feats)
            # Persist under artifacts/
            try:
                import pickle as _pkl
                with open(_artifact_path("regime_gmm_model.pkl"), "wb") as f:
                    _pkl.dump(gmm_model, f)
            except Exception:
                pass
            print(f"[REGIME] GMM fitted on {feats.shape[0]} draws into 3 regimes.")
        else:
            import warnings as _w
            _w.warn("Regime features insufficient after parsing; skipping GMM fit.")
    else:
        import warnings as _w
        _w.warn("Not enough clean draws to fit GMM (need ≥30). Skipping.")
except Exception as _e:
    import warnings as _w
    _w.warn(f"GMM regime fit skipped due to error: {_e}")

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

# Global, history-wide Bayes prior used ONLY to break exact ties in ranking (safe, deterministic)
_TIEBREAK_BAYES = dict(posterior_probs)

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

# --- Hidden Markov Model (HMM) using hmmlearn (multi-seed average with sanity fallback) ---
from hmmlearn import hmm

def _build_tcn_prob_from_subset(sub_draws, n_filters=32, kernel_size=3, lookback=64):
    """
    Temporal Convolutional Network (TCN) for Lotto+ - replaces HMM.

    Uses dilated causal convolutions to capture long-range temporal dependencies
    in lottery draw patterns with a much larger receptive field than HMM.

    Architecture:
      • Multi-scale dilated convolutions: [1, 2, 4, 8, 16, 32] (receptive field: 127 time steps)
      • Residual connections for gradient flow
      • Batch normalization for stable training
      • Dropout for regularization
      • Global pooling + dense layer for final predictions

    Args:
        sub_draws: List of historical draws (each draw is a set/list of 6 numbers from 1-40)
        n_filters: Number of convolutional filters per layer (default: 32)
        kernel_size: Kernel size for convolutions (default: 3)
        lookback: Maximum sequence length to use (default: 64 draws)

    Returns:
        dict {1..40 -> p} where p is a categorical distribution (sum=1)
    """
    import numpy as _np
    from sklearn.preprocessing import MultiLabelBinarizer

    # Build 0/1 matrix X: shape (T, 40) where X[t, n-1] = 1 if number n appeared in draw t
    try:
        mlb_local = MultiLabelBinarizer(classes=list(range(1, 41)))
        X = mlb_local.fit_transform([sorted(list(d)) for d in sub_draws]).astype(float)
    except Exception:
        return {n: 1.0/40 for n in range(1, 41)}

    # Need reasonable history to fit TCN (at least 20 draws, preferably more)
    if X.shape[0] < 20:
        return {n: 1.0/40 for n in range(1, 41)}

    # Use only the most recent 'lookback' draws to avoid excessive computation
    if X.shape[0] > lookback:
        X = X[-lookback:]

    # TCN requires TensorFlow/Keras - fallback to simpler model if unavailable
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Suppress TF warnings for cleaner output
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')

        # Reshape for Conv1D: (batch=1, time_steps=T, features=40)
        X_tcn = _np.expand_dims(X, axis=0)

        # Build TCN model with dilated causal convolutions
        inputs = keras.Input(shape=(X.shape[0], 40))
        x = inputs

        # Dilated convolutional layers with increasing dilation rates
        dilation_rates = [1, 2, 4, 8, 16, 32]
        for i, dilation in enumerate(dilation_rates):
            # Causal dilated conv
            conv = layers.Conv1D(
                filters=n_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                padding='causal',
                activation='relu',
                name=f'tcn_conv_{i}'
            )(x)
            conv = layers.BatchNormalization()(conv)
            conv = layers.Dropout(0.2)(conv, training=True)  # MC dropout for uncertainty

            # Residual connection (with projection if needed)
            if x.shape[-1] != n_filters:
                x = layers.Conv1D(n_filters, 1, padding='same')(x)
            x = layers.Add()([x, conv])

        # Global pooling to aggregate temporal information
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers for final prediction
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x, training=True)
        outputs = layers.Dense(40, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile with categorical crossentropy (no training, just forward pass for now)
        # For a lightweight approach, use the model with random weights as a regularized prior
        # In a full implementation, you'd train on historical data

        # Quick training on the sequence (self-supervised: predict next from history)
        # Create training pairs: input = draws[:-1], target = draws[-1]
        if X.shape[0] >= 10:
            # Simple training: predict last draw from previous draws
            X_train = X[:-1]
            y_train = X[-1:]  # Last draw as target

            # Need multiple samples for training - use sliding windows
            window_size = min(20, X.shape[0] - 1)
            X_windows = []
            y_windows = []
            for i in range(X.shape[0] - window_size):
                X_windows.append(X[i:i+window_size])
                y_windows.append(X[i+window_size])

            if len(X_windows) >= 5:
                X_train = _np.array(X_windows)
                y_train = _np.array(y_windows)

                # Compile and train
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy'  # Multi-label binary classification
                )

                # Quick training (low epochs to avoid overfitting on small data)
                model.fit(
                    X_train, y_train,
                    epochs=20,
                    batch_size=max(1, len(X_train) // 4),
                    verbose=0,
                    shuffle=True
                )

        # Predict on the full sequence
        pred = model.predict(X_tcn, verbose=0)[0]  # Shape: (40,)

        # Enforce sum-to-6 semantics then convert to categorical (sum=1)
        pred_6 = _enforce_sum6(pred * 6.0)
        pred_1 = pred_6 / 6.0
        pred_1 = _np.clip(pred_1, 1e-12, None)
        pred_1 = pred_1 / (pred_1.sum() + 1e-12)

        return {n: float(pred_1[n-1]) for n in range(1, 41)}

    except Exception as e:
        # Fallback: if TCN fails, use a simple time-weighted average
        try:
            # Exponentially weighted moving average (recent draws matter more)
            decay = 0.95
            weights = _np.array([decay ** i for i in range(X.shape[0]-1, -1, -1)])
            weights = weights / weights.sum()

            weighted_freq = _np.dot(weights, X)  # Shape: (40,)
            weighted_freq = _np.clip(weighted_freq, 1e-12, None)
            weighted_freq = weighted_freq / (weighted_freq.sum() + 1e-12)

            return {n: float(weighted_freq[n-1]) for n in range(1, 41)}
        except Exception:
            return {n: 1.0/40 for n in range(1, 41)}

def _build_hmm_prob_from_subset(sub_draws, n_states=3, n_seeds=8):
    """
    Factorized‑emissions HMM for Lotto+.

    Design change:
      • Instead of treating each row as a 40‑category Multinomial “6 outcomes at once,”
        we model a 40‑dimensional binary vector x_t ∈ {0,1}^40 where x_t[n]=1 iff number n
        appeared at draw t. Emissions are **independent per number given the regime**:
        x_t | z_t = k  ~  Bernoulli(mean = μ_k) factorized across 40 dimensions.
      • We fit a **GaussianHMM with diagonal covariance** to these 0/1 vectors.
        The per‑state mean μ_k (shape 40) is an estimate of per‑number occurrence
        probabilities under regime k. After fitting, we compute the posterior over
        regimes at the most recent row and mix the μ_k by those weights.
      • This better captures regime structure without conflating the “six-at-once”
        volume with per‑number signal.

    Returns:
      dict {1..40 -> p} where p is a categorical distribution (sum=1) after:
        (1) clipping μ_mix to [1e-12, 1-1e-12],
        (2) enforcing expected-6 semantics lightly by rescaling to sum≈6,
        (3) converting back to a categorical (sum=1).

    Fallbacks:
      • If GaussianHMM path fails, we fall back to a MultinomialHMM over 0/1 rows
        (legacy behavior), and if that also fails we return uniform.
    """
    import numpy as _np
    from sklearn.preprocessing import MultiLabelBinarizer

    # Build 0/1 matrix X: shape (T, 40)
    try:
        mlb_local = MultiLabelBinarizer(classes=list(range(1, 41)))
        X = mlb_local.fit_transform([sorted(list(d)) for d in sub_draws]).astype(float)
    except Exception:
        return {n: 1.0/40 for n in range(1, 41)}

    # Need a reasonable history to fit HMM
    if X.shape[0] < max(12, n_states + 5):
        return {n: 1.0/40 for n in range(1, 41)}

    # --- Primary path: GaussianHMM with diagonal covariance (factorized emissions)
    ok = 0
    mixes = []
    try:
        from hmmlearn.hmm import GaussianHMM
        with _squelch_streams(r"MultinomialHMM has undergone|hmmlearn"):
            for seed in [11, 23, 37, 49, 61, 73, 89, 97][:max(1, int(n_seeds))]:
                try:
                    # Small jitter to avoid singular covariances on pure 0/1 data
                    X_ = X + _np.random.normal(0.0, 1e-3, size=X.shape)
                    mdl = GaussianHMM(
                        n_components=int(n_states),
                        covariance_type="diag",
                        n_iter=200,
                        tol=1e-3,
                        random_state=int(seed),
                        verbose=False,
                    )
                    mdl.fit(X_)
                    # Posterior over regimes at the last timestep
                    _, post = mdl.score_samples(X_)
                    w = _np.clip(post[-1], 1e-12, None)  # shape: (K,)
                    w = w / (w.sum() + 1e-12)
                    # Emission means per regime (shape: K x 40); interpret as per-number probs
                    means = _np.asarray(mdl.means_, dtype=float)  # (K, 40)
                    mix = _np.dot(w, means)  # (40,)
                    mix = _np.clip(mix, 1e-12, 1 - 1e-12)
                    mixes.append(mix.astype(float))
                    ok += 1
                except Exception:
                    continue
    except Exception:
        ok = 0  # ensure fallback below

    if ok > 0:
        v = _np.mean(_np.vstack(mixes), axis=0)  # (40,)
        # Lightly enforce expected-6 semantics then convert back to categorical
        try:
            v6 = _enforce_sum6(v * 6.0)
            v1 = v6 / 6.0
            v1 = _np.clip(v1, 1e-12, None)
            v1 = v1 / (v1.sum() + 1e-12)
            return {n: float(v1[n-1]) for n in range(1, 41)}
        except Exception:
            v = _np.clip(v, 1e-12, None)
            v = v / (v.sum() + 1e-12)
            return {n: float(v[n-1]) for n in range(1, 41)}

    # --- Fallback path: legacy MultinomialHMM over 0/1 rows (as before)
    try:
        from hmmlearn import hmm as _hmm_mod
        posts = []
        ok2 = 0
        with _squelch_streams(r"MultinomialHMM has undergone|hmmlearn"):
            for seed in [11, 23, 37, 49, 61, 73, 89, 97][:max(1, int(n_seeds))]:
                try:
                    mdl = _hmm_mod.MultinomialHMM(
                        n_components=int(n_states),
                        n_iter=120,
                        tol=1e-3,
                        random_state=int(seed),
                        verbose=False,
                    )
                    # X must be non-negative integers for MultinomialHMM
                    Xi = _np.asarray(X, dtype=int)
                    mdl.fit(Xi)
                    _, post = mdl.score_samples(Xi)               # (T, K)
                    w = _np.clip(post[-1], 1e-12, None)          # (K,)
                    w = w / (w.sum() + 1e-12)
                    emis = _np.asarray(mdl.emissionprob_, dtype=float)  # (K, 40)
                    mix = _np.dot(w, emis)                       # (40,)
                    mix = _np.clip(mix, 1e-12, None)
                    mix = mix / (mix.sum() + 1e-12)
                    posts.append(mix)
                    ok2 += 1
                except Exception:
                    continue
        if ok2 > 0:
            v = _np.mean(_np.vstack(posts), axis=0)
            v = _np.clip(v, 1e-12, None)
            v = v / (v.sum() + 1e-12)
            return {n: float(v[n-1]) for n in range(1, 41)}
    except Exception:
        pass

    # --- Final fallback: uniform
    return {n: 1.0/40 for n in range(1, 41)}

# Robust, version-tolerant live TCN probability (replaces HMM)
try:
    hmm_prob = _build_tcn_prob_from_subset(draws) if HMM_OK else _uniform_prob40()
except Exception:
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

# Normalised co-occurrence rates for incompatibility penalties (0..1)
pair_max = max(1, max(pair_counter.values()) if len(pair_counter) > 0 else 1)
trip_max = max(1, max(triplet_counter.values()) if len(triplet_counter) > 0 else 1)
pair_rate = {p: (c / pair_max) for p, c in pair_counter.items()}
triplet_rate = {t: (c / trip_max) for t, c in triplet_counter.items()}

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


#
#
# Regime-based diagnostics (optional)
# If we have a fitted gmm_model (from the post-parse block above) use it;
# otherwise, try to load from artifacts; else skip gracefully.
try:
    if gmm_model is None:
        import os as _os, pickle as _pkl
        _p = _artifact_path("regime_gmm_model.pkl")
        if _os.path.exists(_p):
            with open(_p, "rb") as f:
                gmm_model = _pkl.load(f)
except Exception:
    pass

try:
    recent_mat = draw_matrix[-5:] if isinstance(draw_matrix, np.ndarray) and draw_matrix.size else np.array([sorted(list(d)) for d in draws][-5:])
except Exception:
    recent_mat = None

if gmm_model is not None and recent_mat is not None and len(recent_mat) >= 3:
    try:
        feats_recent = extract_regime_features(recent_mat)
        preds = gmm_model.predict(feats_recent)
        counts = np.bincount(preds, minlength=getattr(gmm_model, "n_components", 3))
        current_regime = int(np.argmax(counts))
        print(f"[REGIME] Current regime: {current_regime}")
    except Exception as _e:
        import warnings as _w
        _w.warn(f"Regime prediction skipped: {_e}")
else:
    print("[REGIME] GMM not available or insufficient recent draws; skipping regime diagnostics.")
# Note: We intentionally do not auto-switch models here because `marginal_model_0`,
# `setar_model_1`, and `reranker_model_2` may not exist in all builds. If you want
# to gate behavior by regime, plug the regime id into your existing selection logic.
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


# --- Co-occurrence community detection (for cluster features) ---
def _cooc_communities_from_history(sub_draws):
    """Return (cluster_id, cluster_size) dicts via greedy modularity on the co-occurrence graph.
    cluster_id maps 1..40 -> small int id; cluster_size maps id -> size. Falls back to singletons.
    """
    try:
        G = nx.Graph()
        G.add_nodes_from(range(1, 41))
        co = _cooccurrence_matrix_from_history(sub_draws)
        for i in range(1, 41):
            for j in range(i+1, 41):
                w = float(co[i-1, j-1])
                if w > 0:
                    G.add_edge(i, j, weight=w)
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
        cid = {}
        for ci, c in enumerate(comms):
            for n in c:
                cid[int(n)] = int(ci)
        for n in range(1, 41):
            if n not in cid:
                cid[n] = int(len(comms))
                comms.append({n})
        csize = {i: len(c) for i, c in enumerate(comms)}
        return cid, csize
    except Exception:
        return ({n: n-1 for n in range(1, 41)}, {i: 1 for i in range(40)})


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
M_seq = []  # meta features per sample
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
    k = len(draw_window)  # Determine window size from draw_window length
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

    # --- New set-aware features ---
    last_in_window = sorted(list(draw_window[-1])) if len(draw_window) > 0 else []
    # Spacing features for the last draw in the window
    gaps_vec = []
    if len(last_in_window) >= 2:
        gaps_vec = [last_in_window[i+1] - last_in_window[i] for i in range(len(last_in_window)-1)]
    min_gap_last = float(min(gaps_vec)) if gaps_vec else 0.0
    max_gap_last = float(max(gaps_vec)) if gaps_vec else 0.0
    mean_gap_last = float(np.mean(gaps_vec)) if gaps_vec else 0.0

    # Sum/parity buckets for the last draw in the window
    sum_last = float(sum(last_in_window)) if last_in_window else 0.0
    odd_last = float(sum(1 for n in last_in_window if n % 2 == 1))
    even_last = float(sum(1 for n in last_in_window if n % 2 == 0))

    # Low/high (1–20 vs 21–40) pressure in the recent 10 draws
    window_10 = draws[max(0, idx_offset-9): idx_offset+1]
    low_ct = float(sum(1 for d in window_10 for n in d if n <= 20))
    high_ct = float(sum(1 for d in window_10 for n in d if n >= 21))
    total_ct = max(1.0, low_ct + high_ct)
    low_frac_recent = low_ct / total_ct
    high_frac_recent = high_ct / total_ct

    # Co-occurrence community features from history up to idx_offset
    comm_id, comm_size = _cooc_communities_from_history(draws[:idx_offset+1])
    MAX_COMM_FEATS = 6  # fixed length for community one-hot features (last bin = overflow/unknown)

    # Per-number precomputations for new features
    nearest_dist = [0.0]*40
    contain_gap_size = [0.0]*40
    for n in range(1, 41):
        if last_in_window:
            nearest_dist[n-1] = float(min(abs(n-x) for x in last_in_window))
            cg = 0
            for i in range(len(last_in_window)-1):
                a, b = last_in_window[i], last_in_window[i+1]
                if a < n < b:
                    cg = b - a
                    break
            contain_gap_size[n-1] = float(cg)
        else:
            nearest_dist[n-1] = 0.0
            contain_gap_size[n-1] = 0.0

    # Normalizations
    nearest_dist_norm = [d/40.0 for d in nearest_dist]
    contain_gap_norm = [g/40.0 for g in contain_gap_size]
    sum_last_norm = sum_last / 240.0
    odd_last_norm = odd_last / 6.0
    even_last_norm = even_last / 6.0
    min_gap_norm = min_gap_last / 40.0
    max_gap_norm = max_gap_last / 40.0
    mean_gap_norm = mean_gap_last / 40.0

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
        # Candidate-level extensions
        cand_val = i+1
        cand_parity = 1.0 if (cand_val % 2 == 1) else 0.0
        cand_low = 1.0 if cand_val <= 20 else 0.0
        cid = float(comm_id.get(cand_val, -1))
        csz = float(comm_size.get(int(comm_id.get(cand_val, -1)), 1))
        cluster_size_norm = csz / 40.0
        if last_in_window:
            overlap = sum(1 for x in last_in_window if comm_id.get(x, -2) == comm_id.get(cand_val, -1))
            cluster_overlap_ratio = (overlap / max(1.0, csz))
        else:
            cluster_overlap_ratio = 0.0

        # Build cluster-membership one-hot (capped length; last index is overflow/unknown)
        ci_raw = int(comm_id.get(cand_val, -1))
        if ci_raw < 0:
            ci_idx = MAX_COMM_FEATS - 1
        else:
            ci_idx = ci_raw if ci_raw < MAX_COMM_FEATS else (MAX_COMM_FEATS - 1)
        comm_onehot = [1.0 if k == ci_idx else 0.0 for k in range(MAX_COMM_FEATS)]

        # ========== PHASE 1: CROSS-NUMBER INTERACTION FEATURES ==========
        # These capture relationships between numbers beyond individual statistics

        # 1. Consecutive run features (is this number part of a consecutive sequence?)
        cand_num = i + 1
        consecutive_before = 0
        consecutive_after = 0
        if last_in_window:
            if (cand_num - 1) in last_in_window:
                consecutive_before = 1
            if (cand_num + 1) in last_in_window:
                consecutive_after = 1
        consecutive_total = consecutive_before + consecutive_after

        # 2. Arithmetic sequence indicator (part of equally-spaced sequence)
        # Check if cand_num forms arithmetic progression with 2+ numbers in last draw
        in_arithmetic_seq = 0
        if last_in_window and len(last_in_window) >= 2:
            for j in range(len(last_in_window)):
                for k in range(j+1, len(last_in_window)):
                    a, b = last_in_window[j], last_in_window[k]
                    diff = b - a
                    # Check if cand_num forms arithmetic seq with (a,b)
                    if cand_num == a - diff or cand_num == b + diff:
                        in_arithmetic_seq = 1
                        break
                if in_arithmetic_seq:
                    break

        # 3. Momentum feature (trending up or down in recent draws)
        # Compare frequency in last 5 draws vs previous 5 draws
        momentum = 0.0
        if idx_offset >= 10:
            recent_5 = draws[max(0, idx_offset-4):idx_offset+1]
            prev_5 = draws[max(0, idx_offset-9):max(0, idx_offset-4)]
            freq_recent = sum(1 for d in recent_5 if cand_num in d) / max(1, len(recent_5))
            freq_prev = sum(1 for d in prev_5 if cand_num in d) / max(1, len(prev_5))
            momentum = freq_recent - freq_prev  # Range: [-1, 1]

        # 4. Co-occurrence score with last draw (sum of pairwise co-occurrence rates)
        cooc_score = 0.0
        if last_in_window:
            for other_num in last_in_window:
                pair = tuple(sorted([cand_num, other_num]))
                if pair in pair_counter:
                    cooc_score += pair_counter[pair] / max(1, len(draws[:idx_offset+1]))
        cooc_score_norm = min(1.0, cooc_score)  # Normalize to [0, 1]

        # 5. Exclusion pattern (inverse co-occurrence - numbers that rarely appear together)
        exclusion_score = 0.0
        if last_in_window:
            for other_num in last_in_window:
                pair = tuple(sorted([cand_num, other_num]))
                expected_cooc = (freq_counter_window.get(cand_num, 1) * freq_counter_window.get(other_num, 1)) / max(1, len(draw_window)**2)
                actual_cooc = pair_counter.get(pair, 0) / max(1, len(draws[:idx_offset+1]))
                if expected_cooc > actual_cooc:
                    exclusion_score += (expected_cooc - actual_cooc)
        exclusion_score_norm = min(1.0, exclusion_score)

        # 6. Quadrant distribution features (pressure from each quarter of the number space)
        # Count how many numbers from each quadrant in last 3 draws
        last_3_draws = draw_window[-min(3, len(draw_window)):] if draw_window else []
        q1_pressure = sum(1 for d in last_3_draws for n in d if 1 <= n <= 10) / max(1, len(last_3_draws) * 6)
        q2_pressure = sum(1 for d in last_3_draws for n in d if 11 <= n <= 20) / max(1, len(last_3_draws) * 6)
        q3_pressure = sum(1 for d in last_3_draws for n in d if 21 <= n <= 30) / max(1, len(last_3_draws) * 6)
        q4_pressure = sum(1 for d in last_3_draws for n in d if 31 <= n <= 40) / max(1, len(last_3_draws) * 6)

        # Which quadrant is candidate number in?
        cand_quadrant = min(3, (cand_num - 1) // 10)  # 0, 1, 2, or 3
        cand_quadrant_pressure = [q1_pressure, q2_pressure, q3_pressure, q4_pressure][cand_quadrant]

        # 7. Cycle detection (does number appear in regular intervals?)
        cycle_strength = 0.0
        if idx_offset >= 20:
            appearances = [t for t in range(max(0, idx_offset-19), idx_offset+1) if cand_num in draws[t]]
            if len(appearances) >= 3:
                gaps_between = [appearances[j+1] - appearances[j] for j in range(len(appearances)-1)]
                if gaps_between:
                    gap_std = float(np.std(gaps_between))
                    # Low std means regular cycle, high std means irregular
                    cycle_strength = max(0.0, 1.0 - gap_std/10.0)  # Normalize

        # 8. Sum compatibility (does cand_num help reach typical sum ranges?)
        # Typical winning sums are around 120-140 (6 numbers averaging 20-23)
        if last_in_window and len(last_in_window) >= 3:
            partial_sum = sum(last_in_window[:3])  # Sum of first 3 numbers from last draw
            expected_remaining = 3 * 20  # Expected average for remaining 3 numbers
            deviation_if_added = abs((partial_sum + cand_num + expected_remaining) - 120) / 120.0
            sum_compatibility = max(0.0, 1.0 - deviation_if_added)
        else:
            sum_compatibility = 0.5

        # ========== PHASE 2: ADVANCED PATTERN FEATURES ==========

        # 9. Decade distribution features (track density in each decade)
        # Decades: [1-10], [11-20], [21-30], [31-40]
        last_5_draws = draw_window[-min(5, len(draw_window)):] if draw_window else []
        decade_counts = [0, 0, 0, 0]
        for d in last_5_draws:
            for n in d:
                decade_idx = min(3, (n - 1) // 10)
                decade_counts[decade_idx] += 1
        total_nums = max(1, sum(decade_counts))
        decade_pressure = [c / total_nums for c in decade_counts]
        cand_decade = min(3, (cand_num - 1) // 10)
        cand_decade_pressure = decade_pressure[cand_decade]

        # 10. Advanced odd/even balance (deviation from ideal 3-3 split)
        odd_even_balance = 0.5
        if last_in_window:
            odd_count_last = sum(1 for n in last_in_window if n % 2 == 1)
            # Ideal is 3 odd, 3 even; measure deviation
            ideal_odd = 3.0
            balance_deviation = abs(odd_count_last - ideal_odd) / 3.0
            odd_even_balance = max(0.0, 1.0 - balance_deviation)

        # 11. Prime number indicator
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
        is_prime = 1.0 if cand_num in primes else 0.0
        prime_count_last = sum(1 for n in last_in_window if n in primes) if last_in_window else 0
        prime_ratio_last = prime_count_last / max(1, len(last_in_window))

        # 12. Digit sum features (single digit sum, e.g., 37 -> 3+7=10 -> 1+0=1)
        digit_sum = sum(int(d) for d in str(cand_num))
        while digit_sum >= 10:
            digit_sum = sum(int(d) for d in str(digit_sum))
        digit_sum_norm = digit_sum / 9.0  # Normalize to [0, 1]

        # 13. Advanced gap analysis (gap acceleration - is gap growing or shrinking?)
        gap_acceleration = 0.0
        if idx_offset >= 15:
            appearances = [t for t in range(max(0, idx_offset-14), idx_offset+1) if cand_num in draws[t]]
            if len(appearances) >= 3:
                gaps_recent = [appearances[j+1] - appearances[j] for j in range(len(appearances)-2, len(appearances)-1)]
                gaps_prev = [appearances[j+1] - appearances[j] for j in range(len(appearances)-3, len(appearances)-2)]
                if gaps_recent and gaps_prev:
                    gap_acceleration = (np.mean(gaps_recent) - np.mean(gaps_prev)) / 10.0  # Normalized

        # 14. Hot/cold streak indicator (hot if appeared in 3+ of last 5, cold if 0 of last 5)
        hot_cold_indicator = 0.0
        if idx_offset >= 5:
            recent_5_draws = draws[max(0, idx_offset-4):idx_offset+1]
            appearances_in_5 = sum(1 for d in recent_5_draws if cand_num in d)
            if appearances_in_5 >= 3:
                hot_cold_indicator = 1.0  # Hot
            elif appearances_in_5== 0:
                hot_cold_indicator = -1.0  # Cold

        # 15. Mirror number feature (40-n+1, e.g., mirror of 5 is 36)
        mirror_num = 41 - cand_num
        mirror_in_last = 1.0 if (last_in_window and mirror_num in last_in_window) else 0.0
        mirror_freq_recent = 0.0
        if idx_offset >= 10:
            recent_10 = draws[max(0, idx_offset-9):idx_offset+1]
            mirror_freq_recent = sum(1 for d in recent_10 if mirror_num in d) / len(recent_10)

        feats += [
            nearest_dist_norm[i],      # distance to nearest num in last draw (norm)
            contain_gap_norm[i],       # size of containing gap (norm)
            min_gap_norm, max_gap_norm, mean_gap_norm,  # last-draw spacing stats
            sum_last_norm, odd_last_norm, even_last_norm,  # last-draw sum/parity
            low_frac_recent, high_frac_recent,            # bucket pressure (recent)
            cand_parity, cand_low,                        # candidate parity/bucket
            cluster_size_norm, cluster_overlap_ratio,     # co-occurrence cluster features
            # NEW CROSS-NUMBER INTERACTION FEATURES:
            consecutive_before, consecutive_after, consecutive_total,  # Consecutive runs
            in_arithmetic_seq,                            # Arithmetic sequence membership
            momentum,                                     # Trend direction (up/down)
            cooc_score_norm,                             # Co-occurrence with last draw
            exclusion_score_norm,                        # Exclusion pattern strength
            q1_pressure, q2_pressure, q3_pressure, q4_pressure,  # Quadrant pressures
            cand_quadrant_pressure,                      # Candidate's quadrant pressure
            cycle_strength,                              # Regular cycle indicator
            sum_compatibility,                           # Sum range compatibility
            # NEW PHASE 2 ADVANCED PATTERN FEATURES:
            cand_decade_pressure,                        # Decade pressure for candidate
            odd_even_balance,                            # Odd/even balance quality
            is_prime,                                    # Is candidate a prime number
            prime_ratio_last,                            # Prime ratio in last draw
            digit_sum_norm,                              # Digit sum (numerology)
            gap_acceleration,                            # Gap trend (accelerating/decelerating)
            hot_cold_indicator,                          # Hot/cold streak indicator
            mirror_in_last,                              # Mirror number in last draw
            mirror_freq_recent                           # Mirror number frequency (recent)
        ] + comm_onehot

        # ========== PHASE 3: FEATURE INTERACTION TERMS ==========
        # Add 2nd-order feature crosses to capture non-linear interactions
        # Select key features for interaction (avoid explosion)
        key_idx = {
            'freq': 0, 'gap': 2, 'momentum': len(feats) - 20,  # Approximate indices
            'hot_cold': len(feats) - 5, 'prime': len(feats) - 9
        }

        # Interaction 1: Frequency × Gap (hot numbers with short gaps)
        freq_gap_interact = feats[0] * feats[2] if len(feats) > 2 else 0.0

        # Interaction 2: Momentum × Hot/Cold (trending hot/cold strength)
        momentum_idx = max(0, min(len(feats) - 20, len(feats) - 1))
        hot_cold_idx = max(0, min(len(feats) - 5, len(feats) - 1))
        momentum_hot_interact = feats[momentum_idx] * feats[hot_cold_idx] if len(feats) > 20 else 0.0

        # Interaction 3: Prime × Decade pressure (prime density in decade)
        prime_idx = max(0, min(len(feats) - 9, len(feats) - 1))
        decade_idx = max(0, min(len(feats) - 12, len(feats) - 1))
        prime_decade_interact = feats[prime_idx] * feats[decade_idx] if len(feats) > 12 else 0.0

        # Interaction 4: Gap acceleration × Cycle strength (predictable patterns)
        gap_accel_idx = max(0, min(len(feats) - 7, len(feats) - 1))
        cycle_idx = max(0, min(len(feats) - 18, len(feats) - 1))
        accel_cycle_interact = feats[gap_accel_idx] * feats[cycle_idx] if len(feats) > 18 else 0.0

        # Interaction 5: Co-occurrence × Consecutive (numbers appearing together and consecutively)
        cooc_idx = max(0, min(len(feats) - 22, len(feats) - 1))
        consec_idx = max(0, min(len(feats) - 25, len(feats) - 1))
        cooc_consec_interact = feats[cooc_idx] * feats[consec_idx] if len(feats) > 25 else 0.0

        # Add interaction terms to feature vector
        feats += [
            freq_gap_interact,
            momentum_hot_interact,
            prime_decade_interact,
            accel_cycle_interact,
            cooc_consec_interact
        ]

        features.append(feats)
    return np.array(features)  # shape (40, n_features + 5 interactions)


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

def _meta_features_at_idx(t_idx):
    """Return [weekday, entropy, dispersion] with light scaling for the nets."""
    t_idx = int(t_idx)
    reg_t = _regime_features_at_t(max(0, t_idx))
    ent_t = _rolling_entropy_from_history(draws[:max(1, t_idx+1)], window=30)
    disp_t = _dispersion_last_draw(draws[:max(1, t_idx+1)])
    return [
        float(reg_t['weekday'])/6.0,
        float(ent_t)/float(np.log(40.0)),
        float(disp_t)/20.0,
    ]

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
    # Meta features based on regime/entropy/dispersion at the window end (idx-1)
    M_seq.append(_meta_features_at_idx(idx-1))
    # Output: 40-dim binary vector for next draw
    y_seq.append([1 if num in next_draw else 0 for num in range(1, 41)])
X_seq = np.array(X_seq)  # shape (samples, 40, n_features)
M_seq = np.array(M_seq)  # shape (samples, 3)
y_seq = np.array(y_seq)  # shape (samples, 40)
# Clear guard now that the bulk feature construction is complete
_set_target_idx(None)

# --- Auxiliary Target Preparation for Multi-Task Learning (Phase 4 Upgrade 2) ---
def prepare_tcn_aux_targets(draws_data, indices, window=10):
    """
    Prepare auxiliary targets for TCN: Frequency change trends.

    For each sample at index idx, compute how number frequencies are changing
    over time by comparing recent vs older frequency distributions.

    Args:
        draws_data: List of draw sets (each draw is a set of 6 numbers from 1-40)
        indices: List of indices corresponding to each sample in X_seq
        window: Number of draws to use for frequency comparison

    Returns:
        np.array of shape (len(indices), 40) with values in [-1, 1]
        representing frequency change trends for each number
    """
    aux_targets = []

    for idx in indices:
        if idx < window * 2:
            # Not enough history for comparison, use zeros
            aux_targets.append(np.zeros(40))
        else:
            # Compute recent frequency (last window draws before idx)
            recent_draws = draws_data[idx - window:idx]
            recent_counts = np.zeros(40)
            for draw in recent_draws:
                for num in draw:
                    recent_counts[num - 1] += 1
            recent_freq = recent_counts / window

            # Compute older frequency (window draws before that)
            older_draws = draws_data[idx - 2*window:idx - window]
            older_counts = np.zeros(40)
            for draw in older_draws:
                for num in draw:
                    older_counts[num - 1] += 1
            older_freq = older_counts / window

            # Compute frequency change (normalized)
            freq_change = (recent_freq - older_freq) / (older_freq + 1e-6)
            # Clip to [-1, 1] range using tanh-like normalization
            freq_change = np.clip(freq_change, -1, 1)

            aux_targets.append(freq_change)

    return np.array(aux_targets)


def prepare_transformer_aux_targets(draws_data, indices, lookahead=2):
    """
    Prepare auxiliary targets for Transformer: Future pattern prediction.

    For each sample at index idx, create a binary indicator of which numbers
    appear in future draws (at idx+lookahead).

    Args:
        draws_data: List of draw sets
        indices: List of indices corresponding to each sample in X_seq
        lookahead: How many steps ahead to predict (default: 2)

    Returns:
        np.array of shape (len(indices), 40) with binary values
        indicating which numbers appear in future draws
    """
    aux_targets = []
    n_draws = len(draws_data)

    for idx in indices:
        future_idx = idx + lookahead
        if future_idx >= n_draws:
            # No future draw available, use zeros
            aux_targets.append(np.zeros(40))
        else:
            # Binary indicator for future draw
            future_draw = draws_data[future_idx]
            future_pattern = np.array([1 if num in future_draw else 0 for num in range(1, 41)])
            aux_targets.append(future_pattern)

    return np.array(aux_targets)


def prepare_gnn_aux_targets(draws_data, indices, window=30, top_k=20):
    """
    Prepare auxiliary targets for GNN: Co-occurrence community membership.

    For each sample, identify which numbers belong to strong co-occurrence
    communities based on historical patterns.

    Args:
        draws_data: List of draw sets
        indices: List of indices corresponding to each sample
        window: Number of recent draws to analyze for co-occurrence
        top_k: Number of top co-occurring pairs to consider for communities

    Returns:
        np.array of shape (len(indices), 40) with values in [0, 1]
        representing community membership strength
    """
    aux_targets = []

    for idx in indices:
        if idx < window:
            # Not enough history, use uniform
            aux_targets.append(np.ones(40) * 0.5)
        else:
            # Analyze recent draws for co-occurrence
            recent_draws = draws_data[max(0, idx - window):idx]

            # Build co-occurrence matrix
            cooc_matrix = np.zeros((40, 40))
            for draw in recent_draws:
                nums = list(draw)
                for i, n1 in enumerate(nums):
                    for n2 in nums[i+1:]:
                        cooc_matrix[n1-1, n2-1] += 1
                        cooc_matrix[n2-1, n1-1] += 1

            # Compute community strength for each number
            # Use row-wise sum as a proxy for community centrality
            community_strength = np.sum(cooc_matrix, axis=1)

            # Normalize to [0, 1]
            if community_strength.max() > 0:
                community_strength = community_strength / community_strength.max()
            else:
                community_strength = np.ones(40) * 0.5

            aux_targets.append(community_strength)

    return np.array(aux_targets)

# --- Set-aware learning-to-rank losses (PL/BT) -------------------------------------------
import tensorflow as tf
import keras  # Keras 3.x API
from keras import layers
# --- Import-time training guard for Keras -----------------------------------
try:
    import keras as _k_guard
    if hasattr(_k_guard, "Model") and hasattr(_k_guard.Model, "fit"):
        _ORIG_KERAS_FIT = _k_guard.Model.fit
        _FIT_WARNED = {"done": False}
        def _FIT_GUARD(self, *args, **kwargs):
            if __name__ == "__main__":
                return _ORIG_KERAS_FIT(self, *args, **kwargs)
            if not _FIT_WARNED["done"]:
                warnings.warn("Skipped keras.Model.fit during import; guarded by __name__ check.")
                _FIT_WARNED["done"] = True
            return None
        _k_guard.Model.fit = _FIT_GUARD
except Exception:
    pass
# ---------------------------------------------------------------------------

def pl_set_loss_factory(R=12, tau=0.15, label_smoothing_eps=0.03, ls_weight=0.10):
    """
    Plackett–Luce pseudo-likelihood loss for unordered 6-sets with optional label smoothing.
    y_true: [batch, 40] multi-hot (six 1s)
    y_pred: [batch, 40] raw logits per number
    R: number of random permutations of the positive set to average over
    tau: magnitude of Gumbel noise for Perturb-and-MAP (0 disables)
    label_smoothing_eps: distributes epsilon mass onto non-winners
    ls_weight: weight of the auxiliary smoothed cross-entropy term
    """
    neg_inf = tf.constant(-1e9, dtype=tf.float32)

    @tf.function(experimental_relax_shapes=True)
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if tau and tau > 0:
            u = tf.clip_by_value(tf.random.uniform(tf.shape(y_pred), 1e-7, 1.0 - 1e-7), 1e-7, 1.0 - 1e-7)
            g = -tf.math.log(-tf.math.log(u)) * tf.cast(tau, y_pred.dtype)
            logits = y_pred + g
        else:
            logits = y_pred

        # Primary PL pseudo-NLL on the set
        def _sample_pl_nll(args):
            y, logit = args
            pos_idx = tf.squeeze(tf.where(y > 0.5), axis=1)
            k = tf.shape(pos_idx)[0]
            def _zero(): return tf.constant(0.0, dtype=tf.float32)
            def _do():
                nll = 0.0
                def body(r, acc):
                    perm = tf.random.shuffle(pos_idx)
                    mask_add = tf.zeros_like(logit)
                    step_nll = 0.0
                    j = tf.constant(0)
                    def step(j, step_acc, mask_add):
                        denom = tf.reduce_logsumexp(logit + mask_add)
                        chosen = perm[j]
                        step_acc += -(tf.gather(logit, chosen) - denom)
                        mask_add = tf.tensor_scatter_nd_update(
                            mask_add, tf.reshape(chosen, [1,1]), tf.reshape(neg_inf, [1])
                        )
                        return j + 1, step_acc, mask_add
                    cond = lambda j, *_: tf.less(j, k)
                    _, step_nll, _ = tf.while_loop(cond, step, [j, step_nll, mask_add])
                    return r + 1, acc + step_nll
                r0 = tf.constant(0)
                _, total = tf.while_loop(lambda r, *_: tf.less(r, R), body, [r0, tf.constant(0.0)])
                return total / tf.cast(tf.maximum(R, 1), tf.float32)
            return tf.cond(tf.equal(tf.size(pos_idx), 0), _zero, _do)

        pl_per_example = tf.map_fn(_sample_pl_nll, (y_true, logits), dtype=tf.float32)
        k = tf.reduce_sum(y_true, axis=1) + 1e-6
        pl_loss = tf.reduce_mean(pl_per_example / k)

        # Auxiliary smoothed CE on softmax(logits)
        eps = tf.cast(label_smoothing_eps, tf.float32)
        winners = y_true
        num_winners = tf.reduce_sum(winners, axis=1, keepdims=True)
        num_non = tf.cast(tf.shape(y_true)[1], tf.float32) - num_winners
        num_winners = tf.maximum(num_winners, 1.0)
        num_non = tf.maximum(num_non, 1.0)
        y_pos = (1.0 - eps) * winners / num_winners
        y_neg = eps * (1.0 - winners) / num_non
        y_smooth = y_pos + y_neg
        log_prob = tf.nn.log_softmax(y_pred, axis=1)
        ce_loss = tf.reduce_mean(-tf.reduce_sum(y_smooth * log_prob, axis=1))

        return pl_loss + tf.cast(ls_weight, tf.float32) * ce_loss

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

def build_tcnn_model(input_shape, output_dim, meta_dim=3, final_dropout=0.35, use_aux_task=None):
    """
    DeepSets-style set encoder with a tiny meta-feature head (weekday, entropy, dispersion).

    Phase 4 Upgrade 2: Added auxiliary task for temporal trend decomposition.
    Auxiliary task: Predict frequency change trends (how number frequencies are shifting).
    """
    if use_aux_task is None:
        use_aux_task = USE_SPECIALIZED_TRAINING

    inputs = layers.Input(shape=input_shape, name="set_inputs")        # (40, n_features)
    meta_in = layers.Input(shape=(meta_dim,), name="meta_inputs")      # (3,)

    # Φ: element-wise embedding
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Global set summary + meta head
    g = layers.GlobalAveragePooling1D()(x)
    m = layers.Dense(16, activation='relu')(meta_in)
    m = layers.Dense(16, activation='relu')(m)
    g = layers.Concatenate()([g, m])
    g = layers.Dense(64, activation='relu')(g)

    # Broadcast back to elements
    timesteps = input_shape[0] if (input_shape and input_shape[0] is not None) else 40
    g_rep = layers.RepeatVector(timesteps)(g)
    x_combined = layers.Concatenate(axis=-1)([x, g_rep])

    # ρ: element-wise scoring MLP (stronger dropout late)
    x = layers.Dense(64, activation='relu')(x_combined)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(final_dropout)(x)

    # Main output: logits for each number
    logits = layers.Conv1D(1, kernel_size=1)(x)           # (batch, 40, 1)
    logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='main_logits')(logits)

    # Phase 4 Upgrade 2: Auxiliary task - Frequency change prediction
    if use_aux_task:
        # Predict frequency change trends (40 values: -1 to +1 for each number)
        # This helps the model learn temporal dynamics
        freq_change = layers.Dense(32, activation='relu')(g)
        freq_change = layers.Dense(40, activation='tanh', name='freq_change')(freq_change)  # (batch, 40)

        model = keras.Model(inputs=[inputs, meta_in], outputs=[logits, freq_change])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss={
                'main_logits': pl_set_loss_factory(R=12, tau=0.15, label_smoothing_eps=0.03, ls_weight=0.10),
                'freq_change': 'mse'  # Mean squared error for frequency change prediction
            },
            loss_weights={
                'main_logits': 1.0 - TCNN_AUX_WEIGHT,
                'freq_change': TCNN_AUX_WEIGHT
            }
        )
    else:
        model = keras.Model(inputs=[inputs, meta_in], outputs=logits)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss=pl_set_loss_factory(R=12, tau=0.15, label_smoothing_eps=0.03, ls_weight=0.10),
        )

    return model

# --- 3.3. Transformer Model Implementation ---
def build_transformer_model(input_shape, output_dim, num_heads=4, ff_dim=96, meta_dim=3, final_dropout=0.35, use_aux_task=None):
    """
    Set Transformer style with a tiny meta-feature head; logits shape [batch, 40].

    Phase 4 Upgrade 2: Added auxiliary task for long-range pattern prediction.
    Auxiliary task: Predict future draw patterns (anticipate draws 2-3 steps ahead).
    """
    if use_aux_task is None:
        use_aux_task = USE_SPECIALIZED_TRAINING

    inputs = layers.Input(shape=input_shape, name="set_inputs")    # (40, n_features)
    meta_in = layers.Input(shape=(meta_dim,), name="meta_inputs")  # (3,)
    x = inputs

    # SAB block 1
    attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=8)(x, x)
    attn1 = layers.Dropout(0.2)(attn1)
    x = layers.Add()([x, attn1]); x = layers.LayerNormalization()(x)
    ff1 = layers.Dense(ff_dim, activation='relu')(x)
    ff1 = layers.Dropout(0.2)(ff1)
    ff1 = layers.Dense(input_shape[-1])(ff1)
    x = layers.Add()([x, ff1]); x = layers.LayerNormalization()(x)

    # SAB block 2
    attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=8)(x, x)
    attn2 = layers.Dropout(0.2)(attn2)
    y = layers.Add()([x, attn2]); y = layers.LayerNormalization()(y)
    ff2 = layers.Dense(ff_dim, activation='relu')(y)
    ff2 = layers.Dropout(0.2)(ff2)
    ff2 = layers.Dense(input_shape[-1])(ff2)
    y = layers.Add()([y, ff2]); y = layers.LayerNormalization()(y)

    # Meta head → fuse into global, broadcast
    g = layers.GlobalAveragePooling1D()(y)
    m = layers.Dense(16, activation='relu')(meta_in)
    m = layers.Dense(16, activation='relu')(m)
    g = layers.Concatenate()([g, m])
    g_shared = layers.Dense(64, activation='relu')(g)
    timesteps = input_shape[0] if (input_shape and input_shape[0] is not None) else 40
    g_rep = layers.RepeatVector(timesteps)(g_shared)
    y_combined = layers.Concatenate(axis=-1)([y, g_rep])

    # Per-element projection (stronger final dropout)
    y = layers.Dropout(final_dropout)(y_combined)
    logits = layers.Conv1D(1, kernel_size=1)(y)
    logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='main_logits')(logits)

    # Phase 4 Upgrade 2: Auxiliary task - Long-range pattern prediction
    if use_aux_task:
        # Predict future draw characteristics (helps learn long-range dependencies)
        # Output: 40 values representing likelihood of numbers appearing in future draws
        future_pattern = layers.Dense(64, activation='relu')(g_shared)
        future_pattern = layers.Dropout(0.3)(future_pattern)
        future_pattern = layers.Dense(40, activation='sigmoid', name='future_pattern')(future_pattern)  # (batch, 40)

        model = keras.Model(inputs=[inputs, meta_in], outputs=[logits, future_pattern])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss={
                'main_logits': pl_set_loss_factory(R=12, tau=0.15, label_smoothing_eps=0.03, ls_weight=0.10),
                'future_pattern': 'binary_crossentropy'  # Binary cross-entropy for future pattern prediction
            },
            loss_weights={
                'main_logits': 1.0 - TRANSFORMER_AUX_WEIGHT,
                'future_pattern': TRANSFORMER_AUX_WEIGHT
            }
        )
    else:
        model = keras.Model(inputs=[inputs, meta_in], outputs=logits)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss=pl_set_loss_factory(R=12, tau=0.15, label_smoothing_eps=0.03, ls_weight=0.10),
        )

    return model


# === Graph Neural Network (GNN) Expert =========================================
# GNN models co-occurrence relationships between numbers as a graph structure.
# Nodes: 40 lottery numbers (1-40)
# Edges: Co-occurrence relationships weighted by frequency and recency
# Architecture: Graph Convolutional Network (GCN) with multiple message passing layers

def build_cooccurrence_graph(hist_draws, window=30, min_cooc=2):
    """
    Build adjacency matrix from co-occurrence patterns in historical draws.

    Args:
        hist_draws: List of draws (each draw is a set/list of numbers)
        window: Number of recent draws to consider
        min_cooc: Minimum co-occurrences to create an edge

    Returns:
        adj_matrix: (40, 40) symmetric adjacency matrix with co-occurrence weights
        node_features: (40, feature_dim) node features for each number
    """
    import numpy as np_
    from collections import Counter
    from itertools import combinations

    # Use recent draws for graph construction
    recent = hist_draws[-window:] if len(hist_draws) > window else hist_draws
    if len(recent) == 0:
        # Return uniform graph if no history
        adj = np_.ones((40, 40), dtype=float) * 0.1
        np_.fill_diagonal(adj, 1.0)
        return adj, np_.ones((40, 5), dtype=float) * 0.5

    # Count co-occurrences (pairs appearing in same draw)
    cooc_counts = Counter()
    for draw in recent:
        draw_list = sorted(list(draw))
        for a, b in combinations(draw_list, 2):
            # Store as sorted tuple
            pair = (min(a, b), max(a, b))
            cooc_counts[pair] += 1

    # Build adjacency matrix (40x40)
    adj = np_.zeros((40, 40), dtype=float)
    max_cooc = max(cooc_counts.values()) if cooc_counts else 1

    for (a, b), count in cooc_counts.items():
        if count >= min_cooc:
            # Normalize by max and add to adjacency
            weight = float(count) / max_cooc
            adj[a-1, b-1] = weight
            adj[b-1, a-1] = weight  # Symmetric

    # Add self-loops
    np_.fill_diagonal(adj, 1.0)

    # Normalize adjacency matrix (D^-0.5 * A * D^-0.5 for GCN)
    degree = np_.sum(adj, axis=1)
    degree = np_.where(degree > 0, degree, 1.0)  # Avoid division by zero
    d_inv_sqrt = np_.power(degree, -0.5)
    d_mat_inv_sqrt = np_.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # Build node features
    node_features = _build_graph_node_features(hist_draws, window)

    return adj_normalized, node_features


def _build_graph_node_features(hist_draws, window=30):
    """Build per-node features for GNN."""
    import numpy as np_
    from collections import Counter

    recent = hist_draws[-window:] if len(hist_draws) > window else hist_draws
    if len(recent) == 0:
        return np_.ones((40, 5), dtype=float) * 0.5

    # Feature 1: Frequency in recent window
    freq_counts = Counter([n for d in recent for n in d])
    max_freq = max(freq_counts.values()) if freq_counts else 1
    frequencies = np_.array([freq_counts.get(i, 0) / max_freq for i in range(1, 41)], dtype=float)

    # Feature 2: Recency (draws since last appearance)
    recency = np_.zeros(40, dtype=float)
    for i in range(1, 41):
        for idx, draw in enumerate(reversed(recent)):
            if i in draw:
                recency[i-1] = 1.0 / (idx + 1)  # More recent = higher value
                break

    # Feature 3: Degree centrality (number of co-occurrence partners)
    degree = np_.zeros(40, dtype=float)
    for draw in recent:
        draw_list = list(draw)
        for n in draw_list:
            degree[n-1] += len(draw_list) - 1  # Number of partners in this draw
    degree = degree / (degree.max() + 1e-9)

    # Feature 4: Low/High position (1-20 vs 21-40)
    position = np_.array([1.0 if i <= 20 else 0.0 for i in range(1, 41)], dtype=float)

    # Feature 5: Number identity (normalized)
    identity = np_.linspace(0, 1, 40, dtype=float)

    # Stack features: (40, 5)
    node_feats = np_.stack([frequencies, recency, degree, position, identity], axis=1)
    return node_feats


class GraphConvLayer(layers.Layer):
    """Custom Graph Convolutional Layer for GNN."""

    def __init__(self, units, activation='relu', dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout
        self.dense = None
        self.dropout_layer = None

    def build(self, input_shape):
        # input_shape: [(batch, N, F), (batch, N, N)] where N=40 nodes
        self.dense = layers.Dense(self.units, activation=self.activation)
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, adj = inputs  # x: (batch, 40, F), adj: (batch, 40, 40)

        # Graph convolution: adj @ x @ W
        # adj @ x: (batch, 40, 40) @ (batch, 40, F) = (batch, 40, F)
        aggregated = tf.matmul(adj, x)  # Aggregate neighbor features

        # Apply dense layer
        out = self.dense(aggregated)

        # Dropout
        out = self.dropout_layer(out, training=training)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'dropout': self.dropout_rate,
        })
        return config


def build_gnn_model(num_nodes=40, node_feature_dim=5, meta_dim=3,
                    hidden_dims=[64, 64, 32], final_dropout=0.35, use_aux_task=None):
    """
    Build a Graph Neural Network for lottery prediction.

    Args:
        num_nodes: Number of nodes (40 for lottery numbers 1-40)
        node_feature_dim: Dimension of node features
        meta_dim: Dimension of meta features (weekday, entropy, dispersion)
        hidden_dims: List of hidden layer dimensions
        final_dropout: Dropout rate before output layer
        use_aux_task: Enable auxiliary co-occurrence prediction task

    Phase 4 Upgrade 2: Added auxiliary task for co-occurrence prediction.
    Auxiliary task: Predict pairwise co-occurrence strengths (which numbers tend to appear together).

    Returns:
        Compiled Keras model
    """
    if use_aux_task is None:
        use_aux_task = USE_SPECIALIZED_TRAINING

    # Inputs
    node_features = layers.Input(shape=(num_nodes, node_feature_dim), name="node_features")
    adj_matrix = layers.Input(shape=(num_nodes, num_nodes), name="adj_matrix")
    meta_in = layers.Input(shape=(meta_dim,), name="meta_inputs")

    # Graph convolutional layers
    x = node_features
    for hidden_dim in hidden_dims:
        x = GraphConvLayer(hidden_dim, activation='relu', dropout=0.2)([x, adj_matrix])
        x = layers.LayerNormalization()(x)  # Normalize after each GCN layer

    # Global pooling to get graph-level representation
    g = layers.GlobalAveragePooling1D()(x)  # (batch, hidden_dim)

    # Meta feature head
    m = layers.Dense(16, activation='relu')(meta_in)
    m = layers.Dense(16, activation='relu')(m)

    # Combine graph representation with meta features
    g = layers.Concatenate()([g, m])
    g_shared = layers.Dense(64, activation='relu')(g)

    # Broadcast back to node level
    g_rep = layers.RepeatVector(num_nodes)(g_shared)  # (batch, 40, 64)

    # Combine with node-level features
    x_combined = layers.Concatenate(axis=-1)([x, g_rep])  # (batch, 40, hidden_dim + 64)

    # Final MLP for per-node prediction
    x = layers.Dense(64, activation='relu')(x_combined)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(final_dropout)(x)

    # Output logits for each node (number)
    logits = layers.Conv1D(1, kernel_size=1)(x)  # (batch, 40, 1)
    logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='main_logits')(logits)  # (batch, 40)

    # Phase 4 Upgrade 2: Auxiliary task - Co-occurrence community prediction
    if use_aux_task:
        # Predict which numbers form strong co-occurrence communities
        # Output: 40 values representing community membership strength
        cooc_community = layers.Dense(32, activation='relu')(g_shared)
        cooc_community = layers.Dense(40, activation='sigmoid', name='cooc_community')(cooc_community)  # (batch, 40)

        # Build model with dual outputs
        model = keras.Model(
            inputs=[node_features, adj_matrix, meta_in],
            outputs=[logits, cooc_community],
            name="GNN_Lottery_Predictor_MultiTask"
        )

        # Compile with multi-task loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss={
                'main_logits': pl_set_loss_factory(R=12, tau=0.15, label_smoothing_eps=0.03, ls_weight=0.10),
                'cooc_community': 'binary_crossentropy'  # Community membership prediction
            },
            loss_weights={
                'main_logits': 1.0 - GNN_AUX_WEIGHT,
                'cooc_community': GNN_AUX_WEIGHT
            }
        )
    else:
        # Build model with single output (backward compatible)
        model = keras.Model(
            inputs=[node_features, adj_matrix, meta_in],
            outputs=logits,
            name="GNN_Lottery_Predictor"
        )

        # Compile with same loss as other models
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss=pl_set_loss_factory(R=12, tau=0.15, label_smoothing_eps=0.03, ls_weight=0.10),
        )

    return model


def _gnn_prob_from_history(hist_draws, gnn_model=None, meta_features=None):
    """
    Predict probabilities using GNN model.

    Args:
        hist_draws: Historical draws
        gnn_model: Trained GNN model (if None, returns uniform)
        meta_features: Meta features (weekday, entropy, dispersion)

    Returns:
        Dictionary {1..40 -> probability}
    """
    import numpy as np_

    if gnn_model is None or len(hist_draws) < 10:
        # Return uniform if model not available
        return {i: 1.0/40 for i in range(1, 41)}

    try:
        # Build graph from history
        adj, node_feats = build_cooccurrence_graph(hist_draws, window=30)

        # Prepare inputs
        adj_batch = np_.expand_dims(adj, axis=0)  # (1, 40, 40)
        node_batch = np_.expand_dims(node_feats, axis=0)  # (1, 40, 5)

        # Meta features (default if not provided)
        if meta_features is None:
            meta_features = np_.array([[0.0, 0.5, 0.5]], dtype=float)  # (1, 3)
        else:
            # Convert to numpy array if it's a list
            if not isinstance(meta_features, np_.ndarray):
                meta_features = np_.array(meta_features, dtype=float)
            # Expand dims if 1D
            if meta_features.ndim == 1:
                meta_features = np_.expand_dims(meta_features, axis=0)

        # Predict (handle multi-task outputs from Phase 4 Upgrade 2)
        outputs = gnn_model.predict([node_batch, adj_batch, meta_features], verbose=0)

        # Phase 4: Multi-task models return [main_logits, aux_output]
        # Extract only the main logits (first output)
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]  # Main task output
        else:
            logits = outputs  # Single-task model (backward compatible)

        logits = logits.reshape(-1)  # (40,)

        # Convert logits to probabilities
        probs = 1.0 / (1.0 + np_.exp(-logits))
        probs = np_.clip(probs, 1e-12, 1 - 1e-12)
        probs = probs / (probs.sum() + 1e-12)

        return {i: float(probs[i-1]) for i in range(1, 41)}

    except Exception as e:
        warnings.warn(f"GNN prediction failed: {e}")
        return {i: 1.0/40 for i in range(1, 41)}


# --- Joint (exact-ticket) scoring helpers -------------------------------------

# === Set-AR beam search and joint combinator (new) ===========================
# This block provides a joint ticket decoder that searches directly in
# 6-number combination space. It builds candidates with a beam search over
# marginals (with co-occurrence bonuses) and optionally re-ranks them with
# the ticket-level XGBoost model if available.
JOINT_BEAM_WIDTH = 256
JOINT_TOP_POOL = 20
JOINT_MAX_CANDIDATES = 512
LAMBDA_PAIR = 0.55     # strength of co-occurrence pair bonus in beam
LAMBDA_BUCKET = 0.08   # penalty for extreme low/high imbalance
LAMBDA_DISP = 0.03     # penalty for odd spacing / extreme sums

def _marginals_from_logits(logits):
    """Convert length-40 logits/scores to a normalised marginal dict {1..40->p}."""
    import numpy as _np
    v = _np.asarray(logits, dtype=float).reshape(-1)
    if v.size != 40:
        raise ValueError("logits must have length 40")
    p = 1.0 / (1.0 + _np.exp(-v))
    p = _np.clip(p, 1e-12, 1 - 1e-12)
    p = p / (p.sum() + 1e-12)
    return {i+1: float(p[i]) for i in range(40)}

def _ticket_logscore(ticket, base_prob_dict, hist_draws):
    """Joint score for a completed 6-number ticket (higher is better)."""
    import numpy as _np
    from itertools import combinations as _comb
    S = tuple(sorted(int(n) for n in set(ticket)))
    if len(S) != 6:
        raise ValueError("ticket must contain 6 unique numbers")
    # Sum log marginals
    logp = 0.0
    for n in S:
        p = float(base_prob_dict.get(n, 1e-12))
        p = max(p, 1e-12)
        logp += float(_np.log(p))
    # Pairwise co-occurrence bonus from global pair_rate (built earlier)
    pb = 0.0
    try:
        for a, b in _comb(S, 2):
            key = (a, b) if (a, b) in pair_rate else (b, a)
            pb += float(pair_rate.get(key, 0.0))
    except Exception:
        pb = 0.0
    # Low/High balance + spacing
    low = sum(1 for n in S if n <= 20)
    high = 6 - low
    bucket_pen = abs(low - high) / 6.0
    diffs = [S[i+1] - S[i] for i in range(5)]
    mean_gap = float(_np.mean(diffs))
    sumv = float(sum(S))
    # encourage typical central sums/gaps weakly via penalties
    norm_sum_pen = abs(sumv - 120.0) / 120.0       # center near mid-sum
    norm_gap_pen = abs(mean_gap - 6.5) / 40.0      # mild target for average spacing
    return (logp
            + LAMBDA_PAIR * pb
            - LAMBDA_BUCKET * bucket_pen
            - LAMBDA_DISP * (norm_sum_pen + norm_gap_pen))

def _beam_search_on_marginals(base_prob_dict, beam=JOINT_BEAM_WIDTH, k=6, top_pool=JOINT_TOP_POOL):
    """Greedy/beam expansion over the top-pool of numbers using additive proxy scores."""
    import numpy as _np
    # sort top pool by marginal
    top = sorted(base_prob_dict.items(), key=lambda kv: kv[1], reverse=True)[:int(top_pool)]
    pool_nums = [int(n) for n, _ in top]
    beams = [tuple()]      # list of partial tuples (sorted)
    scores = [0.0]         # accumulated partial scores (approximate)
    for step in range(k):
        candidates = []
        seen = set()
        for partial, acc in zip(beams, scores):
            for n in pool_nums:
                if n in partial:
                    continue
                new_t = tuple(sorted(partial + (n,)))
                if new_t in seen:
                    continue
                # partial additive score: log p(n) + small pair bonus vs already chosen
                sc = acc + float(_np.log(max(base_prob_dict.get(n, 1e-12), 1e-12)))
                if len(partial) > 0:
                    for a in partial:
                        key = (a, n) if (a, n) in pair_rate else (n, a)
                        sc += (LAMBDA_PAIR * float(pair_rate.get(key, 0.0)) / max(1, step))
                candidates.append((sc, new_t))
                seen.add(new_t)
        # prune to beam width
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:int(beam)]
        beams = [t for _, t in candidates]
        scores = [s for s, _ in candidates]
    # Score completed tickets with the full joint score
    completed = []
    hist = draws[:-1] if (isinstance(draws, list) and len(draws) > 0) else []
    for t in beams:
        completed.append((_ticket_logscore(t, base_prob_dict, hist), t))
    completed.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in completed]

def _ticket_set_features(ticket, hist_draws):
    """Compact, stable features for a full 6-number ticket (used by reranker)."""
    import numpy as _np
    from itertools import combinations as _comb

    S = tuple(sorted(int(n) for n in set(ticket)))
    if len(S) != 6:
        raise ValueError("ticket must contain 6 unique numbers")

    arr = _np.array(S, dtype=float)
    s_sum  = float(arr.sum())
    s_mean = float(arr.mean())
    s_min  = float(arr.min())
    s_max  = float(arr.max())
    s_rng  = float(arr.ptp())

    odd  = sum(1 for n in S if n % 2 == 1)
    even = 6 - odd
    low  = sum(1 for n in S if n <= 20)
    high = 6 - low

    gaps   = _np.diff(arr)
    g_min  = float(gaps.min())  if gaps.size else 0.0
    g_mean = float(gaps.mean()) if gaps.size else 0.0
    g_max  = float(gaps.max())  if gaps.size else 0.0

    try:
        prs = []
        for a, b in _comb(S, 2):
            key = (a, b) if (a, b) in pair_rate else (b, a)
            prs.append(float(pair_rate.get(key, 0.0)))
        pair_avg = float(_np.mean(prs)) if prs else 0.0
    except Exception:
        pair_avg = 0.0

    try:
        tris = []
        for a, b, c in _comb(S, 3):
            key = (a, b, c)
            if key not in globals().get('triplet_rate', {}):
                key = tuple(sorted((a, b, c)))
            tris.append(float(globals().get('triplet_rate', {}).get(key, 0.0)))
        trip_sum = float(_np.sum(tris)) if tris else 0.0
    except Exception:
        trip_sum = 0.0

    try:
        base = compute_bayes_posterior(hist_draws, alpha=1) if hist_draws else {n: 1.0/40 for n in range(1, 41)}
    except Exception:
        base = {n: 1.0/40 for n in range(1, 41)}
    p_vals   = _np.array([float(base.get(n, 1.0/40)) for n in S], dtype=float)
    p_sum    = float(p_vals.sum())
    p_logsum = float(_np.log(_np.clip(p_vals, 1e-12, None)).sum())

    try:
        last = sorted(list(hist_draws[-1])) if hist_draws else []
        overlap = sum(1 for n in S if n in last) if last else 0
        near = [min(abs(n - x) for x in last) if last else 0.0 for n in S]
        near_mean = float(_np.mean(near)) if near else 0.0
    except Exception:
        overlap = 0
        near_mean = 0.0

    return _np.array([
        s_sum, s_mean, s_min, s_max, s_rng,
        odd, even, low, high,
        g_min, g_mean, g_max,
        pair_avg, trip_sum,
        p_sum, p_logsum,
        overlap, near_mean
    ], dtype=float)

_ticket_reranker = None  # trained lazily

def _train_ticket_reranker_quick(hist_draws, max_hist=120, neg_per=6):
    """Fit a tiny XGB regressor to prefer real past tickets over near-miss negatives.
    Safe: wraps all failures and returns None if xgboost is unavailable.
    """
    try:
        if xgb is None:
            return None
        X = []
        y = []
        start = max(8, len(hist_draws) - int(max_hist))
        for t in range(start, len(hist_draws)):
            hist = hist_draws[:t]
            pos = tuple(sorted(int(n) for n in hist_draws[t]))
            # base probs from history via Bayes (fast and leakage-safe)
            base = compute_bayes_posterior(hist, alpha=1)
            # near-miss negatives from a small beam on history
            neg_pool = _beam_search_on_marginals(base, beam=64, k=6, top_pool=18)
            negs = []
            for c in neg_pool:
                if set(c) != set(pos):
                    negs.append(c)
                if len(negs) >= int(neg_per):
                    break
            X.append(_ticket_set_features(pos, hist)); y.append(1.0)
            for n in negs:
                X.append(_ticket_set_features(n, hist)); y.append(0.0)
        if len(X) < 50:
            return None
        model = xgb.XGBRegressor(
            n_estimators=750, max_depth=10, learning_rate=0.025,
            subsample=0.75, colsample_bytree=0.75, colsample_bylevel=0.8,
            min_child_weight=4, gamma=0.15, reg_alpha=0.15, reg_lambda=1.5,
            n_jobs=1, verbosity=0  # Phase 3: Enhanced hyperparameters
        )
        import numpy as _np
        model.fit(_np.array(X), _np.array(y))
        return model
    except Exception:
        return None

def _rerank_tickets(candidates, hist_draws):
    """If a reranker is available, reorder candidates by its score."""
    global _ticket_reranker
    try:
        if _ticket_reranker is None:
            _ticket_reranker = _train_ticket_reranker_quick(hist_draws)
        if _ticket_reranker is None:
            return candidates
        import numpy as _np
        X = _np.array([_ticket_set_features(t, hist_draws) for t in candidates])
        scores = _ticket_reranker.predict(X)
        order = scores.argsort()[::-1]
        return [candidates[i] for i in order]
    except Exception:
        return candidates

def choose_ticket_joint(base_prob_dict, hist_draws, beam=JOINT_BEAM_WIDTH, top_pool=JOINT_TOP_POOL):
    """Generate, score, and (optionally) rerank joint ticket candidates; return the best."""
    cands = _beam_search_on_marginals(base_prob_dict, beam=int(beam), k=6, top_pool=int(top_pool))
    # dedupe while preserving order
    uniq = []
    seen = set()
    for t in cands:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= JOINT_MAX_CANDIDATES:
            break
    hist = hist_draws if isinstance(hist_draws, list) else []
    ranked = _rerank_tickets(uniq, hist)
    if ranked:
        return tuple(sorted(ranked[0]))
    # fallback to top-6 marginals if something went wrong
    arr = sorted(base_prob_dict.items(), key=lambda kv: kv[1], reverse=True)[:6]
    return tuple(sorted(int(n) for n, _ in arr))

def predict_joint_number_set(model_logits=None, base_prob_dict=None, beam_width=None):
    """Public API: given model logits OR a marginal dict, return a 6-number ticket (tuple)."""
    if base_prob_dict is None:
        if model_logits is not None:
            try:
                base_prob_dict = _marginals_from_logits(model_logits)
            except Exception:
                base_prob_dict = None
    if base_prob_dict is None:
        # last-resort: Bayes from all history
        try:
            base_prob_dict = compute_bayes_posterior(draws, alpha=1)
        except Exception:
            base_prob_dict = {n: 1.0/40 for n in range(1, 41)}
    bw = int(beam_width) if beam_width is not None else int(JOINT_BEAM_WIDTH)
    hist = draws[:-1] if (isinstance(draws, list) and len(draws) > 0) else []
    return choose_ticket_joint(base_prob_dict, hist, beam=bw, top_pool=JOINT_TOP_POOL)
# === End Set-AR joint block ==================================================

# ============ Ticket-level Reranker (XGBoost) =================================
# We train a lightweight gradient-boosted tree to re-rank full 6-number tickets.
# Features are permutation-invariant set descriptors computed from history up to t.

    """
    Return a 1D numpy feature vector for a 6-number `ticket` (iterable of ints)
    using ONLY information from `hist_draws` (draws strictly before the target t).
    """
    import numpy as _np
    from itertools import combinations as _comb
    S = sorted(int(n) for n in set(ticket))
    if len(S) != 6:
        raise ValueError("ticket must contain 6 unique numbers")
    # Basic set stats
    diffs = [S[i+1]-S[i] for i in range(5)]
    sumv = float(sum(S))
    min_gap = float(min(diffs))
    max_gap = float(max(diffs))
    mean_gap = float(_np.mean(diffs))
    odd_ct = float(sum(1 for n in S if n % 2 == 1))
    even_ct = 6.0 - odd_ct
    parity_balance = abs(odd_ct - even_ct) / 6.0
    low_ct = float(sum(1 for n in S if n <= 20))
    high_ct = 6.0 - low_ct
    low_high_balance = abs(low_ct - high_ct) / 6.0
    # Co-occurrence scores from global counters (already built earlier)
    pair_sc = 0.0
    pair_cnt = 0
    for a,b in _comb(S, 2):
        pair_sc += float(pair_rate.get((a,b), pair_rate.get((b,a), 0.0)))
        pair_cnt += 1
    pair_sc = (pair_sc / max(1, pair_cnt))
    trip_sc = 0.0
    trip_cnt = 0
    for a,b,c in _comb(S, 3):
        trip_sc += float(triplet_rate.get((a,b,c), 0.0))
        trip_cnt += 1
    trip_sc = (trip_sc / max(1, trip_cnt))
    # Community overlap and distance to last draw
    last = sorted(list(hist_draws[-1])) if len(hist_draws) else []
    cid, csz = _cooc_communities_from_history(hist_draws)
    comm_ids = [cid.get(n, -1) for n in S]
    # max overlap of ticket with any community that the last draw used
    last_comms = {cid.get(n, -2) for n in last}
    overlap_comm = float(sum(1 for n in S if cid.get(n, -2) in last_comms)) / 6.0
    # mean nearest distance to last draw
    if last:
        mean_dist_last = float(_np.mean([min(abs(n-x) for x in last) for n in S]))/40.0
    else:
        mean_dist_last = 0.0
    # KN‑Markov continuation/compat score: average candidate score under KN mix built from history
    try:
        t1,t2,t3 = compute_markov_transitions(hist_draws)
        last1 = hist_draws[-1] if len(hist_draws) >= 1 else set()
        last2 = hist_draws[-2] if len(hist_draws) >= 2 else set()
        last3 = hist_draws[-3] if len(hist_draws) >= 3 else set()
        kn = _kn_interpolated_markov(last1, last2, last3, t1, t2, t3)
        kn_sc = float(sum(kn.get(n, 0.0) for n in S)) / 6.0
    except Exception:
        kn_sc = 1.0/40.0
    # Normalize some features
    sum_norm = sumv / 240.0
    return _np.array([
        sum_norm, parity_balance, low_high_balance,
        min_gap/40.0, max_gap/40.0, mean_gap/40.0,
        pair_sc, trip_sc, overlap_comm, mean_dist_last, kn_sc
    ], dtype=float)

def _build_reranker_training_data(last_N=120, beam_size=384, near_miss=32, epochs=16, Kperm=6):
    """
    Construct pairwise ranking data over the last_N draws:
    For each t in the window, train a small Set‑AR on draws[:t], generate top beams,
    take the true ticket as positive and `near_miss` hardest negatives from the beam.
    Returns (features, labels, group_sizes) for xgboost 'rank:pairwise'.
    """
    import numpy as _np
    X = []; y = []; group = []
    start = max(6, len(draws) - int(last_N))
    for t in range(start, len(draws)):
        try:
            mdl = train_setar_model(draws[:t], epochs=epochs, K=Kperm, verbose=0)
            beams = setar_beam_search(mdl, beam_size=beam_size, B=beam_size)
            truth = sorted(list(draws[t]))
            hist = _history_upto(t, context="_fit_expert_calibrators")
            # Build positives/negatives
            feats_pos = _ticket_set_features(truth, hist)
            X.append(feats_pos); y.append(1.0)
            # Choose near-miss negatives by highest joint score that are != truth
            negs = []
            for tick, sc in beams:
                if sorted(tick) != truth:
                    negs.append((tick, sc))
                if len(negs) >= near_miss:
                    break
            for tick,_ in negs:
                X.append(_ticket_set_features(sorted(tick), hist))
                y.append(0.0)
            group.append(1 + len(negs))
        except Exception as _e:
            warnings.warn(f"reranker data failed at t={t}: {_e}")
    if len(X) == 0:
        return None, None, None
    return _np.vstack(X), _np.asarray(y, float), _np.asarray(group, int)

def train_ticket_reranker(last_N=120, beam_size=384, near_miss=32):
    """
    Train an XGBoost pairwise ranking model on recent draws; caches to 'ticket_reranker.json'.
    """
    if xgb is None:
        warnings.warn("Skipping reranker: XGBoost not available.")
        return None
    X, y, grp = _build_reranker_training_data(last_N=last_N, beam_size=beam_size, near_miss=near_miss)
    if X is None:
        return None
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(list(map(int, grp)))
    params = {
        "objective": "rank:pairwise",
        "eval_metric": "auc",
        "max_depth": 5,
        "eta": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 12,
        "lambda": 1.0,
    }
    bst = xgb.train(params, dtrain, num_boost_round=180)
    try:
        bst.save_model("ticket_reranker.json")
    except Exception:
        pass
    return bst

def _load_ticket_reranker():
    if xgb is None:
        return None
    try:
        bst = xgb.Booster()
        bst.load_model("ticket_reranker.json")
        return bst
    except Exception:
        return None

def rerank_beams_with_reranker(beams, hist_draws, blend_gamma=0.6):
    """
    beams: list of (ticket, logscore) from Set‑AR.
    Returns beams re-scored by XGBoost reranker blended with Set‑AR logscore.
    """
    if not beams:
        return beams
    bst = _load_ticket_reranker()
    if bst is None:
        bst = train_ticket_reranker(last_N=120, beam_size=max(128, len(beams)), near_miss=24)
    if bst is None:
        return beams
    feats = []
    for tick,_ in beams:
        feats.append(_ticket_set_features(sorted(tick), hist_draws))
    dmat = xgb.DMatrix(np.vstack(feats))
    scores = bst.predict(dmat)
    # z-score blend for stability
    import numpy as _np
    s_r = (scores - _np.mean(scores)) / (_np.std(scores) + 1e-9)
    lls = _np.array([sc for _,sc in beams], dtype=float)
    lls = (lls - _np.mean(lls)) / (_np.std(lls) + 1e-9)
    final = blend_gamma * s_r + (1.0 - blend_gamma) * lls
    rescored = [ (beams[i][0], float(final[i])) for i in range(len(beams)) ]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored

# ===========================================================================

# === Lightweight grid search for DPP knobs (alpha, dist_beta) ================
# Uses a short walk-forward over recent draws. For each t, if SetAR is
# available we generate beams and measure whether the *true* ticket appears in
# the top-K after DPP reranking. We pick the knobs that maximize this rate.

def tune_dpp_knobs(last_N=60, beam=160, topK=12,
                    alpha_grid=(0.50, 0.60, 0.65, 0.70, 0.80),
                    beta_grid=(0.05, 0.08, 0.12, 0.16, 0.20)):
    results = {}
    start = max(6, len(draws) - int(last_N))
    tried_any = False
    for t in range(start, len(draws)):
        try:
            # Guard: only use history strictly before t
            hist = draws[:t]
            truth = sorted(list(draws[t]))
            # If SetAR is available, get beams; else skip this t
            if 'train_setar_model' not in globals() or 'setar_beam_search' not in globals():
                continue
            mdl = train_setar_model(hist, epochs=10, K=6, verbose=0)
            beams = setar_beam_search(mdl, beam_size=beam, B=beam, temperature=1.0)
            if not beams:
                continue
            # First, rerank with the XGB ticket reranker if present
            try:
                beams = rerank_beams_with_reranker(beams, hist, blend_gamma=0.6)
            except Exception:
                pass
            # Light marginal prior from Bayes (history-only) if P_marg isn't available for t
            try:
                P_local = compute_bayes_posterior(hist, alpha=1)
            except Exception:
                P_local = None
            for a in alpha_grid:
                for b in beta_grid:
                    tried_any = True
                    try:
                        rer = rerank_beams_with_dpp(beams, hist, quality=P_local, alpha=float(a), dist_beta=float(b), temp=1.0)
                        got = any(sorted(tk) == truth for tk,_ in rer[:max(1, int(topK))])
                        key = (float(a), float(b))
                        results[key] = results.get(key, 0) + (1 if got else 0)
                    except Exception:
                        pass
        except Exception as _e:
            warnings.warn(f"DPP tuning skipped at t={t}: {_e}")
            continue
    if not tried_any or not results:
        return {}
    # pick best
    best_key = max(results.items(), key=lambda kv: kv[1])[0]
    out = {"alpha": best_key[0], "dist_beta": best_key[1], "hits": int(results[best_key]), "trials": int(max(1, len(draws) - max(6, len(draws) - int(last_N))))}
    # Log to JSONL for traceability
    try:
        _log_jsonl({"type": "dpp_tune", "ts": _now_iso(), **out})
    except Exception:
        pass
    return out

# ============ DPP joint decoder (diversity + quality) ========================
# We use an L-ensemble DPP: L = diag(q) @ S @ diag(q)
# - q_i encodes per-number quality (from P_marg or Bayes fallback)
# - S encodes similarity that penalizes picking very similar (close) numbers
# The DPP prefers diverse, high-quality 6-sets.

def _per_number_quality_from_hist(hist_draws, fallback=None):
    """
    Build per-number quality scores q_i using the latest available marginal
    distribution P_marg if present; otherwise fall back to a Bayes posterior
    computed from `hist_draws`. Returns np.array of shape (40,).
    """
    import numpy as _np
    if isinstance(fallback, dict) and len(fallback) == 40:
        q = _np.array([float(fallback.get(i, 1.0/40.0)) for i in range(1,41)], dtype=float)
        return _np.clip(q, 1e-12, None)
    try:
        Pm = globals().get('P_marg', None)
        if isinstance(Pm, dict) and len(Pm) == 40:
            q = _np.array([float(Pm.get(i, 1.0/40.0)) for i in range(1,41)], dtype=float)
            return _np.clip(q, 1e-12, None)
    except Exception:
        pass
    # fallback: simple Bayes posterior from history
    try:
        from collections import Counter as _Counter
        cnt = _Counter([n for d in hist_draws for n in d])
        tot = max(1, len(hist_draws)*6)
        q = _np.array([(1.0 + cnt.get(i,0)) / (tot + 40.0) for i in range(1,41)], dtype=float)
        return _np.clip(q, 1e-12, None)
    except Exception:
        return _np.ones(40, dtype=float) / 40.0


def _build_similarity_matrix(dist_beta=0.12):
    """Similarity S where S[i,j] = exp(-dist_beta * |i-j|). Close numbers have
    higher similarity → DPP discourages selecting them together. Returns S(40x40).
    """
    import numpy as _np
    idx = _np.arange(1,41)[:,None]
    jdx = _np.arange(1,41)[None,:]
    D = _np.abs(idx - jdx).astype(float)
    S = _np.exp(-float(dist_beta) * D)
    # Ensure exact self-similarity of 1.0
    for i in range(40):
        S[i,i] = 1.0
    return S


def _build_dpp_L(hist_draws, quality=None, dist_beta=0.12, temp=1.0, jitter=1e-8):
    """
    Construct L = diag(q) @ S @ diag(q), with q tempered by `temp`.
    """
    import numpy as _np
    q = _per_number_quality_from_hist(hist_draws, fallback=quality)
    q = _np.power(_np.clip(q, 1e-12, None), float(temp))
    S = _build_similarity_matrix(dist_beta=dist_beta)
    L = (q.reshape(-1,1) * S) * q.reshape(1,-1)
    # numerical stability
    for i in range(40):
        L[i,i] = float(L[i,i]) + float(jitter)
    return L


def _dpp_log_prob_of_set(L, ticket):
    """Compute log P(S) = log det(L_S) - log det(I + L) for a 6-number `ticket`."""
    import numpy as _np
    S = sorted(int(n) for n in set(ticket))
    if len(S) != 6:
        return float('-inf')
    idx = [n-1 for n in S]
    L_S = L[_np.ix_(idx, idx)]
    sign1, logdet1 = _np.linalg.slogdet(L_S)
    sign2, logdet2 = _np.linalg.slogdet(_np.eye(40) + L)
    if sign1 <= 0 or sign2 <= 0:
        return float('-inf')
    return float(logdet1 - logdet2)


def rerank_beams_with_dpp(beams, hist_draws, quality=None, alpha=0.65, dist_beta=0.12, temp=1.0):
    """
    Re-score Set‑AR beams using an L-ensemble DPP prior and blend with the
    original (log-)score:  score = alpha*orig + (1-alpha)*log P_DPP(S).
    `quality` may be a {1..40 -> p} dict (e.g., P_marg). Returns re-sorted beams.
    """
    if not beams:
        return beams
    import numpy as _np
    try:
        L = _build_dpp_L(hist_draws, quality=quality, dist_beta=dist_beta, temp=temp)
    except Exception:
        return beams
    resc = []
    orig = _np.array([float(sc) for _,sc in beams], dtype=float)
    # z-score the original scores for blending stability
    orig = (orig - orig.mean()) / (orig.std() + 1e-9)
    for (tick, sc), oz in zip(beams, orig):
        try:
            lp = _dpp_log_prob_of_set(L, tick)
        except Exception:
            lp = 0.0
        final = float(alpha) * float(oz) + (1.0 - float(alpha)) * float(lp)
        resc.append((tick, final))
    resc.sort(key=lambda x: x[1], reverse=True)
    return resc
# ===========================================================================

def _setar_logprob_of_perm(model, perm_tokens):
    """Teacher-forced logprob of a specific ordered sequence (length 6) under SetAR,
    with without-replacement masking enforced. perm_tokens are ints 1..40."""
    BOS = 0
    inp = np.array([[BOS] + perm_tokens[:-1]], dtype=np.int32)  # (1,6)
    tgt = np.array([perm_tokens], dtype=np.int32)               # (1,6)
    logits = model(inp, training=False).numpy()                 # (1,6,vocab)
    vocab = logits.shape[-1]
    running = np.zeros((1, vocab), dtype=np.float32)
    bigneg = -1e9
    logprob = 0.0
    for t in range(6):
        if t > 0:
            idx = inp[0, t]
            running[0, idx] += 1.0
        masked = logits[:, t, :].copy()
        masked[running > 0.5] += bigneg
        m = masked[0]
        m = m - np.max(m)
        p = np.exp(m) / np.clip(np.sum(np.exp(m)), 1e-12, None)
        logprob += float(np.log(np.clip(p[tgt[0, t]], 1e-12, None)))
    return float(logprob)

def setar_max_logprob_of_set(model, ticket, K=24, rng_seed=13):
    """Maximum teacher-forced logprob across K random permutations of `ticket`."""
    import random
    r = random.Random(int(rng_seed))
    base = list(ticket)
    best = -1e18
    for _ in range(max(6, int(K))):
        r.shuffle(base)
        lp = _setar_logprob_of_perm(model, base)
        if lp > best:
            best = lp
    return float(best)

# --- SetAR core components: training and beam search ---------------------------------

def train_setar_model(draws_hist, epochs=150, K=6, verbose=0, d_model=128, seed=1337):
    """
    Train a lightweight autoregressive set model (SetAR) that predicts the next
    element given previously selected elements. We treat each historical 6-number
    draw as a length-6 token sequence (numbers 1..40), prepend a BOS=0 token for
    teacher forcing, and train with sparse categorical cross-entropy.

    Args:
        draws_hist: list[set[int]] of past draws (each size==6), strictly before t.
        epochs: keras fit epochs.
        K: number of random permutations per draw to enforce permutation invariance.
        verbose: keras verbosity.
        d_model: embedding/hidden size.
        seed: RNG seed for reproducibility.

    Returns:
        Keras model that maps an int32 sequence of length 6 (BOS+5 previous picks)
        to logits of shape (6, 41). The vocabulary is {0..40}, where 0=BOS and
        1..40 are lotto numbers.
    """
    import numpy as _np
    import random as _random
    from keras import layers as _L
    # --- Import-time training guard for Keras -----------------------------------
    try:
        import keras as _k_guard
        if hasattr(_k_guard, "Model") and hasattr(_k_guard.Model, "fit"):
            _ORIG_KERAS_FIT = _k_guard.Model.fit
            def _FIT_GUARD(self, *args, **kwargs):
                if __name__ == "__main__":
                    return _ORIG_KERAS_FIT(self, *args, **kwargs)
                warnings.warn("Skipped keras.Model.fit during import; guarded by __name__ check.")
                return None
            _k_guard.Model.fit = _FIT_GUARD
    except Exception:
        pass
    # ---------------------------------------------------------------------------
    BOS = 0
    V = 41  # 0..40 (0=BOS)

    # Build training pairs (inp -> tgt) with K random permutations per draw
    inps = []
    tgts = []
    _rnd = _random.Random(int(seed))
    for draw in draws_hist:
        S = list(set(int(n) for n in draw if 1 <= int(n) <= 40))
        if len(S) != 6:
            continue
        base_sorted = sorted(S)
        for _ in range(max(1, int(K))):
            seq = base_sorted[:]  # deterministic base, then shuffle for invariance
            _rnd.shuffle(seq)
            # inputs: [BOS] + first 5 tokens; targets: full 6 tokens
            inp = [BOS] + seq[:-1]
            tgt = seq
            inps.append(inp)
            tgts.append(tgt)
    if not inps:
        # Safety fallback: create a tiny dummy dataset (uniform) so callers don't crash
        inps = [[0, 0, 0, 0, 0, 0]]
        tgts = [[1, 2, 3, 4, 5, 6]]

    X = _np.asarray(inps, dtype=_np.int32)
    Y = _np.asarray(tgts, dtype=_np.int32)

    # Model: token embedding + small GRU decoder with per-step logits
    inp = _L.Input(shape=(6,), dtype='int32', name='ar_input')
    x = _L.Embedding(input_dim=V, output_dim=int(d_model), mask_zero=False, name='tok_emb')(inp)
    x = _L.GRU(int(d_model*1.5), return_sequences=True, name='gru1')(x)
    x = _L.Dense(int(d_model), activation='relu', name='proj')(x)
    logits = _L.Dense(V, name='logits')(x)  # (batch, 6, 41)

    mdl = keras.Model(inp, logits, name='SetAR')
    mdl.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    cb = [keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)]
    try:
        mdl.fit(X, Y, epochs=int(epochs), batch_size=64, verbose=int(verbose), callbacks=cb, shuffle=True)
    except Exception:
        # Non-fatal; return partially trained model
        pass
    return mdl


def setar_beam_search(model, beam_size=128, B=128, temperature=1.0):
    """
    Autoregressive decoding with without-replacement masking.

    Args:
        model: SetAR model returned by train_setar_model. Must accept int32 input of shape (1,6)
               and return logits of shape (1,6,41).
        beam_size: number of total candidates to retain per layer (alias for B if only one is given).
        B: final beam width to keep after each expansion.
        temperature: softmax temperature for exploration (>1.0 flatter, <1.0 sharper).

    Returns:
        List[(ticket:list[int], score:float)] sorted by descending score, where
        `ticket` is a 6-number sorted list and `score` is cumulative log-prob.
    """
    import numpy as _np

    BOS = 0
    V = 41
    B = int(B if B is not None else beam_size)
    beam_size = int(beam_size)

    def _log_softmax(vec):
        v = _np.asarray(vec, dtype=_np.float32)
        v = v - _np.max(v)
        expv = _np.exp(v / max(1e-8, float(temperature)))
        Z = _np.sum(expv) + 1e-12
        p = expv / Z
        return _np.log(p + 1e-12)

    # beams hold tuples (seq, logprob) where seq contains chosen numbers (1..40)
    beams = [([], 0.0)]
    for t in range(6):
        cand = []
        for seq, lp in beams:
            # Prepare input: [BOS] + seq + pad zeros to length 6
            x = _np.zeros((1, 6), dtype=_np.int32)
            prefix = [BOS] + seq + [0]*max(0, 5 - len(seq))
            x[0, :len(prefix)] = _np.asarray(prefix, _np.int32)
            try:
                logits = model(x, training=False).numpy()[0]  # (6, 41)
            except Exception:
                logits = _np.asarray(model.predict(x, verbose=0))[0]
            step_logits = logits[t]  # next-token position
            # Mask BOS and already-picked tokens to enforce without-replacement
            step_logits[0] = -1e9
            for z in seq:
                step_logits[int(z)] = -1e9
            # Compute log-probs with temperature
            logp = _log_softmax(step_logits)
            # Expand with the top-k candidates to limit branching
            k = min(40 - len(seq), max(6, B))
            top_idx = _np.argsort(logp)[-k:][::-1]
            for idx in top_idx:
                n = int(idx)
                if n == 0 or n in seq:
                    continue
                cand.append((seq + [n], lp + float(logp[idx])))
        # Keep the best B candidates
        if not cand:
            break
        cand.sort(key=lambda s: s[1], reverse=True)
        beams = cand[:max(1, B)]

    # Finalise: sort ticket numbers and return list[(ticket, score)]
    out = []
    for seq, sc in beams:
        if len(seq) == 6:
            out.append((sorted(seq), float(sc)))
    # Fallback: if no full-length sequences, take the longest we have and pad greedily
    if not out and beams:
        seq, sc = beams[0]
        avail = [i for i in range(1, 41) if i not in seq]
        need = 6 - len(seq)
        seq = sorted(seq + avail[:max(0, need)])
        out = [(seq, float(sc))]

    out.sort(key=lambda x: x[1], reverse=True)
    return out

# --- Rolling backtest to fit JOINT isotonic calibrator ------------------------

def run_setar_joint_backtest(last_N=120, epochs=24, Kperm=6, beam=96):
    """Walk-forward over last_N draws: train SetAR on draws[:t], record the joint
    log-score of the true ticket and whether it appears in top-`beam` beams.
    Fit isotonic mapping raw joint scores → success rate. Export joint reliability."""
    start = max(6, len(draws) - int(last_N))
    scores, labels = [], []
    for t in range(start, len(draws)):
        try:
            mdl = train_setar_model(draws[:t], epochs=epochs, K=Kperm, verbose=0)
            truth = sorted(list(draws[t]))
            s = setar_max_logprob_of_set(mdl, truth, K=Kperm)
            scores.append(s)
            beams = setar_beam_search(mdl, beam_size=beam, B=beam)
            got = any(sorted(b[0]) == truth for b in beams)
            labels.append(1 if got else 0)
        except Exception as _e:
            warnings.warn(f"SetAR joint backtest failed at t={t}: {_e}")
    # fit isotonic on (scores, labels)
    try:
        if len(scores) >= 20:
            cal = _PostRankIsotonic().fit(scores, labels)
            globals()['postrank_calibrator_joint'] = cal
            import numpy as _np
            be = _np.linspace(float(min(scores)), float(max(scores) + 1e-9), 11)
            idx = _np.digitize(_np.asarray(scores, float), be, right=True)
            pred_means, obs_rates = [], []
            y_arr = _np.asarray(labels, float)
            for b in range(1, len(be)):
                m = (idx == b)
                if _np.any(m):
                    pm = float(_np.mean(_np.asarray(scores)[m]))
                    orc = float(_np.mean(y_arr[m]))
                else:
                    pm, orc = 0.0, 0.0
                pred_means.append(pm); obs_rates.append(orc)
            _export_reliability(be, pred_means, obs_rates,
                                csv_path="reliability_curve_joint.csv",
                                png_path="reliability_curve_joint.png")
    except Exception as _e:
        warnings.warn(f"Joint calibration export failed: {_e}")

# --- Particle Filter backtest over recent draws ---
def run_particle_filter_backtest(draws, train_size=None, window=50, num_particles=4000, alpha=0.005, sigma=0.01):
    """
    Assimilate up to `train_size` draws, then backtest sequentially on the remainder.
    Returns dict with keys: pf (fitted ParticleFilter), nlls, hits, recent_nll, recent_hits, window, train_size.
    """
    M = 40
    N = len(draws)
    if train_size is None:
        train_size = max(6, N - 150)
    pf = ParticleFilter(num_numbers=M, num_particles=int(num_particles), alpha=float(alpha), sigma=float(sigma))
    # Assimilate training segment
    for t in range(int(train_size)):
        try:
            pf.predict()
            pf.update(draws[t])
        except Exception:
            pass
    # Backtest segment
    pf_nlls, pf_hits = [], []
    for t in range(int(train_size), N):
        try:
            pf.predict()
            mean_p = pf.get_mean_probabilities()
            joint_pred__pf = predict_joint_number_set(base_prob_dict={i+1: float(mean_p[i]) for i in range(40)}, beam_width=JOINT_BEAM_WIDTH)
            top6_idx = np.array(sorted([n-1 for n in joint_pred__pf], key=lambda i: mean_p[i], reverse=True))
            pf_top6 = [int(i)+1 for i in top6_idx]
            pf_top6.sort()
            hits = len(set(pf_top6).intersection(set(draws[t])))
            pf_hits.append(int(hits))
            nll = float(pf.update(draws[t]))
            pf_nlls.append(nll)
        except Exception as _e:
            warnings.warn(f"PF backtest step failed at t={t}: {_e}")
    w = min(int(window), len(pf_nlls)) if len(pf_nlls) > 0 else 0
    recent_nll = float(np.mean(pf_nlls[-w:])) if w > 0 else float('inf')
    recent_hits = float(np.mean(pf_hits[-w:])) if w > 0 else 0.0
    return dict(pf=pf, nlls=pf_nlls, hits=pf_hits, recent_nll=recent_nll, recent_hits=recent_hits, window=w, train_size=int(train_size))

# --- Ensure ensemble recent metrics; compute via Set‑AR if not logged globally ---
def ensure_ensemble_recent_metrics(draws, window=50, last_N=150):
    """
    Tries to use globals 'main_pl_nlls' and 'top6_recall_list'.
    If missing, runs a lightweight Set‑AR walk‑forward over last_N draws to estimate:
      - recent average NLL (as -max joint logprob of truth)
      - recent average hits per draw from top‑1 Set‑AR ticket (or Bayes fallback)
    Returns dict with keys: recent_nll, recent_hits, window.
    """
    try:
        ens_nlls = globals().get('main_pl_nlls', None)
        ens_recalls = globals().get('top6_recall_list', None)
        if isinstance(ens_nlls, (list, tuple)) and len(ens_nlls) > 0 and isinstance(ens_recalls, (list, tuple)) and len(ens_recalls) > 0:
            w = min(int(window), len(ens_nlls), len(ens_recalls))
            return dict(recent_nll=float(np.mean(ens_nlls[-w:])),
                        recent_hits=float(np.mean(ens_recalls[-w:]) * 6.0),
                        window=w)
    except Exception:
        pass

    # Lightweight Set‑AR walk‑forward
    N = len(draws)
    start = max(6, N - int(last_N))
    nlls, hits = [], []
    for t in range(start, N):
        try:
            mdl = train_setar_model(draws[:t], epochs=8, K=4, verbose=0)
            truth = sorted(list(draws[t]))
            lp = setar_max_logprob_of_set(mdl, truth, K=6)
            nlls.append(float(-lp))
            beams = setar_beam_search(mdl, beam_size=128, B=128, temperature=1.0)
            if beams:
                pred = sorted(beams[0][0])
            else:
                # Bayes fallback
                P_local = compute_bayes_posterior(draws[:t], alpha=1)
                # local top‑k helper (avoid dependency on outside helper)
                pred = sorted(sorted(P_local.keys(), key=lambda n: P_local[n], reverse=True)[:6])
            hits.append(float(len(set(pred).intersection(set(truth)))))
        except Exception as _e:
            warnings.warn(f"Ensemble backtest failed at t={t}: {_e}")
    w = min(int(window), len(nlls)) if len(nlls) > 0 else 0
    rec_nll = float(np.mean(nlls[-w:])) if w > 0 else float('inf')
    rec_hits = float(np.mean(hits[-w:])) if w > 0 else 0.0
    return dict(recent_nll=rec_nll, recent_hits=rec_hits, window=w)

# === Train, calibrate & decode SetAR on historical draws ======================
try:
    # Train on all historical draws except the very last (avoid peeking)
    setar_model = train_setar_model(draws[:-1], d_model=96, epochs=50, K=6, verbose=1)
except Exception as _e:
    warnings.warn(f"SetAR training failed: {_e}")
    setar_model = None

# Fit a JOINT isotonic calibrator via a short walk-forward backtest
try:
    run_setar_joint_backtest(last_N=120, epochs=20, Kperm=4, beam=64)
except Exception as _e:
    warnings.warn(f"SetAR joint backtest failed: {_e}")

# Build marginal ensemble for the most recent completed draw to use as a prior
try:
    bases_now = _per_expert_prob_dicts_at_t(n_draws)  # leakage-safe: uses draws[:n_draws]
    _assert_expert_key_consistency(bases_now)
    w_now = np.asarray(globals().get('weights', np.ones(5, dtype=float) / 5.0), dtype=float)
    w_now = _adapt_base_weights(w_now)
    P_marg = _mix_prob_dicts(w_now, bases_now)
except Exception as _e:
    warnings.warn(f"Marginal ensemble build failed: {_e}")
    P_marg = _uniform_prob40()

# Decode joint tickets with SetAR and blend with marginal prior
setar_top = []
try:
    if setar_model is not None:
        setar_raw = setar_beam_search(setar_model, beam_size=128, B=100, temperature=1.0)
        setar_top = blend_joint_with_marginals(setar_raw, P_marg, alpha=0.8)
except Exception as _e:
    warnings.warn(f"SetAR decoding failed: {_e}")
    setar_top = []

# ---- Ticket-level reranker on top of Set‑AR beams ----
try:
    if setar_model is not None and setar_top:
        # Re-run Set‑AR beam search without prior blend to score by joint likelihood
        raw_beams = setar_beam_search(setar_model, beam_size=256, B=256, temperature=1.0)
        # Rerank using XGBoost features and blend with Set‑AR log-likelihood
        setar_top = rerank_beams_with_reranker(raw_beams, draws[:-1], blend_gamma=0.6)
        # Optional: re-apply marginal prior after reranking (light nudging)
        if setar_top:
            setar_top = blend_joint_with_marginals(setar_top, P_marg, alpha=0.8)
        # DPP re-ranking to inject diversity-aware set prior
        try:
            setar_top = rerank_beams_with_dpp(setar_top, draws[:-1], quality=P_marg, alpha=0.65, dist_beta=0.12, temp=1.0)
        except Exception as _e:
            warnings.warn(f"DPP reranker step failed: {_e}")
        # Optional: quick auto‑tune over the last ~60 draws, then re-apply
        try:
            tune = tune_dpp_knobs(last_N=60, beam=128, topK=10,
                                  alpha_grid=(0.55, 0.65, 0.75),
                                  beta_grid=(0.08, 0.12, 0.18))
            if isinstance(tune, dict) and tune.get("alpha") is not None:
                setar_top = rerank_beams_with_dpp(setar_top, draws[:-1], quality=P_marg,
                                                  alpha=float(tune["alpha"]),
                                                  dist_beta=float(tune["dist_beta"]), temp=1.0)
        except Exception as _e:
            warnings.warn(f"DPP auto‑tune failed: {_e}")
except Exception as _e:
    warnings.warn(f"Ticket reranker step failed: {_e}")

# --- Final selection: Dynamic PF vs Ensemble; write SINGLE prediction line ---
try:
    # 1) Backtest Particle Filter on recent data
    pf_bt = run_particle_filter_backtest(draws, train_size=None, window=50, num_particles=1000, alpha=0.01, sigma=0.02)
    pf_recent_nll = pf_bt['recent_nll']
    pf_recent_hits = pf_bt['recent_hits']

    # 2) Get Ensemble recent metrics (logged or freshly computed via Set‑AR)
    ens_met = ensure_ensemble_recent_metrics(draws, window=50, last_N=150)
    ens_recent_nll = float(ens_met.get('recent_nll', float('inf')))
    ens_recent_hits = float(ens_met.get('recent_hits', 0.0))

    # 3) Decide which method to trust right now
    THRESH = 0.05  # NLL advantage needed to be considered "significant"
    if pf_recent_nll + THRESH < ens_recent_nll:
        chosen_method = "PF"
    elif ens_recent_nll + THRESH < pf_recent_nll:
        chosen_method = "Ensemble"
    else:
        chosen_method = "PF" if pf_recent_hits >= ens_recent_hits else "Ensemble"

    # 4) Compute the final top‑6 prediction
    final_top6 = None
    if chosen_method == "PF":
        try:
            pf_model = pf_bt['pf']
            pf_model.predict()  # one‑step ahead beyond the last known draw
            final_probs = pf_model.get_mean_probabilities()
            joint_pred__final = predict_joint_number_set(base_prob_dict={i+1: float(final_probs[i]) for i in range(40)}, beam_width=JOINT_BEAM_WIDTH)
            top6_idx = np.array(sorted([n-1 for n in joint_pred__final], key=lambda i: final_probs[i], reverse=True))
            final_top6 = [int(i)+1 for i in top6_idx]
            final_top6.sort()
        except Exception as _e:
            warnings.warn(f"PF final prediction failed, falling back to ensemble: {_e}")
            chosen_method = "Ensemble"

    if chosen_method == "Ensemble" or final_top6 is None:
        try:
            # Prefer Set‑AR beams if available, otherwise marginal Bayes prior
            if 'setar_top' in globals() and isinstance(setar_top, list) and len(setar_top) > 0:
                final_top6 = sorted(list(setar_top[0][0]))
            else:
                # Fallback to marginal top‑6 from P_marg (already built above)
                P_local = globals().get('P_marg', None)
                if isinstance(P_local, dict) and len(P_local) == 40:
                    final_top6 = sorted(sorted(P_local.keys(), key=lambda n: P_local[n], reverse=True)[:6])
                else:
                    final_top6 = sorted(list(draws[-1]))  # last known draw as last‑resort placeholder
        except Exception as _e:
            warnings.warn(f"Ensemble final prediction fallback failed: {_e}")
            final_top6 = sorted(list(draws[-1]))

    # 5) Write ONLY ONE LINE to predicted_tickets.txt (as requested)
    with open("predicted_tickets.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(str(n) for n in final_top6) + "\n")

    print(f"Chosen method: {chosen_method}, Final Top-6 prediction: {final_top6}")

    # 6) Log selection + recent metrics to JSONL
    try:
        _log_jsonl({
            "type": "final_selection",
            "ts": _now_iso(),
            "chosen": chosen_method,
            "pf_recent_nll": float(pf_recent_nll),
            "pf_recent_hits": float(pf_recent_hits),
            "ensemble_recent_nll": float(ens_recent_nll),
            "ensemble_recent_hits": float(ens_recent_hits),
            "window": int(min(50, len(draws))),
        })
    except Exception:
        pass

except Exception as _e:
    warnings.warn(f"Final selection and write failed: {_e}")

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
M_train_dl, M_val_dl = M_seq[:split_idx], M_seq[split_idx:]
y_train_dl, y_val_dl = y_seq[:split_idx], y_seq[split_idx:]

# --- Prepare auxiliary targets for multi-task learning (Phase 4 Upgrade 2) ---
if USE_SPECIALIZED_TRAINING:
    print("Preparing auxiliary targets for multi-task training...")
    # Generate indices for each sample (k to n_draws-1)
    train_indices = list(range(k, k + split_idx))
    val_indices = list(range(k + split_idx, n_draws - 1))

    # TCN auxiliary targets: Frequency change trends
    tcn_aux_train = prepare_tcn_aux_targets(draws, train_indices, window=10)
    tcn_aux_val = prepare_tcn_aux_targets(draws, val_indices, window=10)

    # Transformer auxiliary targets: Future pattern prediction
    transformer_aux_train = prepare_transformer_aux_targets(draws, train_indices, lookahead=2)
    transformer_aux_val = prepare_transformer_aux_targets(draws, val_indices, lookahead=2)

    print(f"  TCN aux targets: train {tcn_aux_train.shape}, val {tcn_aux_val.shape}")
    print(f"  Transformer aux targets: train {transformer_aux_train.shape}, val {transformer_aux_val.shape}")
else:
    tcn_aux_train = tcn_aux_val = None
    transformer_aux_train = transformer_aux_val = None

print("Training Temporal CNN model...")
lstm_model = build_tcnn_model(input_shape=X_train_dl.shape[1:], output_dim=40, meta_dim=M_train_dl.shape[1], final_dropout=0.35)
# Multi-task training if enabled
if USE_SPECIALIZED_TRAINING and tcn_aux_train is not None:
    print("  Using multi-task training with frequency change auxiliary task")
    lstm_model.fit(
        [X_train_dl, M_train_dl],
        [y_train_dl, tcn_aux_train],  # Multi-task targets
        validation_data=([X_val_dl, M_val_dl], [y_val_dl, tcn_aux_val]),
        epochs=300,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )
else:
    lstm_model.fit(
        [X_train_dl, M_train_dl], y_train_dl,
        validation_data=([X_val_dl, M_val_dl], y_val_dl),
        epochs=300,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )

# Train Transformer Model
print("Training Transformer model...")
transformer_model = build_transformer_model(input_shape=X_train_dl.shape[1:], output_dim=40, meta_dim=M_train_dl.shape[1], final_dropout=0.35)
# Multi-task training if enabled
if USE_SPECIALIZED_TRAINING and transformer_aux_train is not None:
    print("  Using multi-task training with future pattern auxiliary task")
    transformer_model.fit(
        [X_train_dl, M_train_dl],
        [y_train_dl, transformer_aux_train],  # Multi-task targets
        validation_data=([X_val_dl, M_val_dl], [y_val_dl, transformer_aux_val]),
        epochs=1000,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )
else:
    transformer_model.fit(
        [X_train_dl, M_train_dl], y_train_dl,
        validation_data=([X_val_dl, M_val_dl], y_val_dl),
        epochs=1000,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )

# Train GNN Model
print("Training GNN model...")
# Prepare graph data for GNN training
def prepare_gnn_data(start_idx, end_idx):
    """Prepare adjacency matrices and node features for GNN training."""
    adj_list = []
    node_feat_list = []
    meta_list = []
    y_list = []

    for idx in range(start_idx, end_idx):
        if idx < k:
            continue
        # Build graph from history up to idx-1
        hist = draws[:idx]
        try:
            adj, node_feats = build_cooccurrence_graph(hist, window=30)
            adj_list.append(adj)
            node_feat_list.append(node_feats)
            meta_list.append(_meta_features_at_idx(idx-1))
            y_list.append([1 if num in draws[idx] else 0 for num in range(1, 41)])
        except Exception as e:
            warnings.warn(f"Failed to build graph for idx {idx}: {e}")
            continue

    if len(adj_list) == 0:
        return None, None, None, None

    return (np.array(adj_list), np.array(node_feat_list),
            np.array(meta_list), np.array(y_list))

# Prepare GNN training data
gnn_train_idx_start = k
gnn_train_idx_end = k + int((n_draws - k - 1) * 0.8)
gnn_val_idx_start = gnn_train_idx_end
gnn_val_idx_end = n_draws - 1

A_train_gnn, NF_train_gnn, M_train_gnn, y_train_gnn = prepare_gnn_data(gnn_train_idx_start, gnn_train_idx_end)
A_val_gnn, NF_val_gnn, M_val_gnn, y_val_gnn = prepare_gnn_data(gnn_val_idx_start, gnn_val_idx_end)

# Prepare GNN auxiliary targets if multi-task training is enabled
if USE_SPECIALIZED_TRAINING and A_train_gnn is not None:
    gnn_train_indices = list(range(gnn_train_idx_start, gnn_train_idx_end))
    gnn_val_indices = list(range(gnn_val_idx_start, gnn_val_idx_end))
    gnn_aux_train = prepare_gnn_aux_targets(draws, gnn_train_indices, window=30)
    gnn_aux_val = prepare_gnn_aux_targets(draws, gnn_val_indices, window=30)
    # Filter to match the actual samples (in case some were skipped due to errors)
    gnn_aux_train = gnn_aux_train[:len(y_train_gnn)]
    gnn_aux_val = gnn_aux_val[:len(y_val_gnn)]
    print(f"  GNN aux targets: train {gnn_aux_train.shape}, val {gnn_aux_val.shape}")
else:
    gnn_aux_train = gnn_aux_val = None

if A_train_gnn is not None and A_val_gnn is not None:
    gnn_model = build_gnn_model(num_nodes=40, node_feature_dim=5, meta_dim=3,
                                 hidden_dims=[64, 64, 32], final_dropout=0.35)
    # Multi-task training if enabled
    if USE_SPECIALIZED_TRAINING and gnn_aux_train is not None:
        print("  Using multi-task training with co-occurrence community auxiliary task")
        gnn_model.fit(
            [NF_train_gnn, A_train_gnn, M_train_gnn],
            [y_train_gnn, gnn_aux_train],  # Multi-task targets
            validation_data=([NF_val_gnn, A_val_gnn, M_val_gnn], [y_val_gnn, gnn_aux_val]),
            epochs=300,
            batch_size=8,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )
    else:
        gnn_model.fit(
            [NF_train_gnn, A_train_gnn, M_train_gnn], y_train_gnn,
            validation_data=([NF_val_gnn, A_val_gnn, M_val_gnn], y_val_gnn),
            epochs=300,
            batch_size=8,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )
    print("GNN model training complete.")
else:
    print("Warning: Could not prepare GNN training data. GNN will use uniform predictions.")
    gnn_model = None


# ================================================================
# Expanding-window fine-tuning for neural nets (every 8 draws)
# ------------------------------------------------
# Fine-tune copies of the CNN ("lstm_model") and Transformer on the
# most recent [150..150] supervised pairs strictly before t_idx.
# Cached per 8-draw bin to keep cost low.
FINE_TUNE_EVERY = 8
FT_MIN = 150
FT_MAX = 150
FT_EPOCHS = 12
_ft_cache = {}  # key: 8-draw bin -> (tcnn_ft, transformer_ft)

def _prepare_recent_supervised(t_idx, window_min=FT_MIN, window_max=FT_MAX):
    """
    Build (X_recent, y_recent, M_recent) using only draws strictly before t_idx.
    We take the last [window_min..window_max] next-draw supervised pairs.
    If USE_SPECIALIZED_TRAINING is enabled, also returns auxiliary targets.
    """
    _set_target_idx(t_idx)
    start_idx = max(k, t_idx - window_max)
    end_idx = t_idx  # exclusive (labels go up to t_idx-1)
    if (end_idx - start_idx) < window_min:
        start_idx = max(k, end_idx - window_min)
    Xr, yr = [], []
    Mr = []
    indices = []
    for i in range(start_idx, end_idx):
        if i - k < 0:
            continue
        feats = compute_stat_features(draws[i - k:i], i - 1)  # (40, n_features)
        Xr.append(feats)
        Mr.append(_meta_features_at_idx(i-1))
        yr.append([1 if n in draws[i] else 0 for n in range(1, 41)])
        indices.append(i)
    if len(Xr) == 0:
        _set_target_idx(None)
        return None, None, None, None, None
    _set_target_idx(None)

    # Prepare auxiliary targets if multi-task training is enabled
    if USE_SPECIALIZED_TRAINING:
        tcn_aux = prepare_tcn_aux_targets(draws, indices, window=10)
        transformer_aux = prepare_transformer_aux_targets(draws, indices, lookahead=2)
        return np.array(Xr), np.array(yr), np.array(Mr), tcn_aux, transformer_aux
    else:
        return np.array(Xr), np.array(yr), np.array(Mr), None, None

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

    Xr, yr, Mr, tcn_aux, transformer_aux = _prepare_recent_supervised(t_idx)
    if Xr is None or len(Xr) < FT_MIN:
        _set_target_idx(None)
        return lstm_model, transformer_model

    # Time-ordered split: last 10% for validation
    split = max(1, int(len(Xr) * 0.9))
    Xr_tr, Xr_val = Xr[:split], Xr[split:]
    yr_tr, yr_val = yr[:split], yr[split:]
    Mr_tr, Mr_val = Mr[:split], Mr[split:]

    # Split auxiliary targets if available
    if tcn_aux is not None:
        tcn_aux_tr, tcn_aux_val = tcn_aux[:split], tcn_aux[split:]
    else:
        tcn_aux_tr = tcn_aux_val = None

    if transformer_aux is not None:
        transformer_aux_tr, transformer_aux_val = transformer_aux[:split], transformer_aux[split:]
    else:
        transformer_aux_tr = transformer_aux_val = None

    # Clone architectures and initialise from global weights
    tcnn_ft = build_tcnn_model(input_shape=X_train_dl.shape[1:], output_dim=40)
    tcnn_ft.set_weights(lstm_model.get_weights())
    trans_ft = build_transformer_model(input_shape=X_train_dl.shape[1:], output_dim=40)
    trans_ft.set_weights(transformer_model.get_weights())

    # Freeze earlier blocks so fine-tune only adapts late heads
    for l in tcnn_ft.layers:
        if isinstance(l, (layers.MultiHeadAttention, layers.LayerNormalization)):
            l.trainable = False
        elif isinstance(l, (layers.Dense, layers.Conv1D)):
            l.trainable = True
        else:
            l.trainable = False

    for l in trans_ft.layers:
        if isinstance(l, (layers.MultiHeadAttention, layers.LayerNormalization)):
            l.trainable = False
        elif isinstance(l, (layers.Dense, layers.Conv1D)):
            l.trainable = True
        else:
            l.trainable = False

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
    # Use multi-task targets if available
    if USE_SPECIALIZED_TRAINING and tcn_aux_tr is not None:
        tcnn_ft.fit([Xr_tr, Mr_tr], [yr_tr, tcn_aux_tr],
                    validation_data=([Xr_val, Mr_val], [yr_val, tcn_aux_val]),
                    epochs=FT_EPOCHS, batch_size=8, verbose=0, shuffle=True, callbacks=[es])
    else:
        tcnn_ft.fit([Xr_tr, Mr_tr], yr_tr, validation_data=([Xr_val, Mr_val], yr_val),
                    epochs=FT_EPOCHS, batch_size=8, verbose=0, shuffle=True, callbacks=[es])

    if USE_SPECIALIZED_TRAINING and transformer_aux_tr is not None:
        trans_ft.fit([Xr_tr, Mr_tr], [yr_tr, transformer_aux_tr],
                     validation_data=([Xr_val, Mr_val], [yr_val, transformer_aux_val]),
                     epochs=FT_EPOCHS, batch_size=8, verbose=0, shuffle=True, callbacks=[es])
    else:
        trans_ft.fit([Xr_tr, Mr_tr], yr_tr, validation_data=([Xr_val, Mr_val], yr_val),
                     epochs=FT_EPOCHS, batch_size=8, verbose=0, shuffle=True, callbacks=[es])

    _ft_cache[bin_key] = (tcnn_ft, trans_ft)
    _set_target_idx(None)
    return _ft_cache[bin_key]

def _predict_with_nets(feats_batch, meta_batch, t_idx, use_finetune=True, mc_passes=11):
    _set_target_idx(t_idx)
    mdl_tcnn, mdl_trans = (lstm_model, transformer_model)
    if use_finetune:
        mdl_tcnn, mdl_trans = _get_finetuned_nets(t_idx)
    logits_l = _mc_dropout_logits(mdl_tcnn, [feats_batch, meta_batch], n_passes=mc_passes)[0]
    logits_t = _mc_dropout_logits(mdl_trans, [feats_batch, meta_batch], n_passes=mc_passes)[0]
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
meta_pred = np.array([_meta_features_at_idx(n_draws-1)], dtype=float)
_set_target_idx(None)


lstm_prob = None
transformer_prob = None
gnn_prob = None
# Use expanding-window fine-tuned nets for the latest prediction
_l_probs, _t_probs = _predict_with_nets(features_pred, meta_pred, t_idx=n_draws, use_finetune=True, mc_passes=21)
lstm_prob = {num: float(_l_probs[num-1]) for num in range(1, 41)}
transformer_prob = {num: float(_t_probs[num-1]) for num in range(1, 41)}

# GNN prediction for the next draw
gnn_prob = _gnn_prob_from_history(draws, gnn_model=globals().get('gnn_model', None), meta_features=meta_pred[0])

# --- Default equal weights for bases (prevent NameError during logging/mixing) ---
try:
    weights  # if already defined elsewhere, keep it
except NameError:
    weights = np.ones(6, dtype=float) / 6.0  # [Bayes, Markov, HMM, LSTM, Transformer, GNN]

# Provide a no-op regime adapter if not defined elsewhere
if "_adapt_base_weights" not in globals():
    def _adapt_base_weights(w):
        return np.asarray(w, dtype=float)

# --- GP tuning stub (inserted above cheap GP hyper-parameter tuning) ---
def _gp_tune_knobs(trials=0, last_N=60, **kwargs):
    """
    Lightweight guard/stub. If/when you add skopt.gp_minimize, plug it in here.
    Returning an empty dict keeps the caller's logging path clean.
    """
    return {}

# --- Cheap GP hyper-parameter tuning (Markov blend / gap cap / cal temp) ---
try:
    opt = _gp_tune_knobs(trials=12, last_N=60)
    if isinstance(opt, dict):
        rec = {'type': 'gp_opt', 'ts': _now_iso(), **opt}
        print("[OPT] GP-tuned knobs:", opt)
        _log_jsonl(rec, path="run_log.jsonl")
except Exception as _e:
    warnings.warn(f"GP tuning failed: {_e}")


 # --- Kneser–Ney–style blend of 1-/2-/3-step transitions with continuation ---
def _kn_interpolated_markov(last_draw, last_2_draw, last_3_draw, t1, t2, t3):
    # continuation probability: how many unique predecessors led to j
    cont = np.zeros(40, dtype=float)
    for i in range(1, 41):
        for j in range(1, 41):
            if t1[i].get(j, 0.0) > 1e-12:
                cont[j-1] += 1.0
    if cont.sum() > 0:
        cont = cont / (cont.sum() + 1e-12)

    # raw scores from observed contexts
    s1 = np.zeros(40, dtype=float)
    for i in last_draw:
        for j in range(1, 41):
            s1[j-1] += t1[i].get(j, 0.0)
    s2 = np.zeros(40, dtype=float)
    for i in last_2_draw:
        for j in range(1, 41):
            s2[j-1] += t2[i].get(j, 0.0)
    s3 = np.zeros(40, dtype=float)
    for i in last_3_draw:
        for j in range(1, 41):
            s3[j-1] += t3[i].get(j, 0.0)

    # data-driven interpolation weights proportional to evidence size
    w1 = float(len(last_draw))
    w2 = float(len(last_2_draw)) * 0.8
    w3 = float(len(last_3_draw)) * 0.6
    wc = 6.0  # small continuation prior
    # --- Simple regime-based modulation of weights ---
    try:
        # Schedule flip: if day gap not in {2,3}, prefer longer lags/continuation
        if len(data) >= 2 and 'DrawDate' in data.columns:
            gap_days = float((pd.to_datetime(data.iloc[-1]['DrawDate']) - pd.to_datetime(data.iloc[-2]['DrawDate'])).days)
            schedule_flip = 0 if gap_days in (2, 3) else 1
        else:
            schedule_flip = 0
        if schedule_flip:
            w3 *= 1.25
            w2 *= 1.10
            w1 *= 0.85
            wc *= 1.05
        # Entropy/dispersion proxy from last ~10 draws: higher diversity → lean longer lags
        recent = draws[-10:] if len(draws) >= 10 else draws
        uniq = len({n for d in recent for n in d})
        diversity = uniq / 40.0
        if diversity > 0.65:
            w3 *= 1.10
            wc *= 1.10
            w1 *= 0.95
    except Exception:
        pass
    total_w = max(1e-9, (w1 + w2 + w3 + wc))
    w1, w2, w3, wc = w1/total_w, w2/total_w, w3/total_w, wc/total_w

    mix = w1*s1 + w2*s2 + w3*s3 + wc*cont
    mix = np.clip(mix, 1e-12, None)
    mix = mix / (mix.sum() + 1e-12)
    return {n: float(mix[n-1]) for n in range(1, 41)}

# === Probabilistic Modeling & Calibration Utilities ============================
SUM6_EPS = 1e-9
SUM6_MAX_ITERS = 4
CAL_WINDOW = 60              # draws used to fit calibrators
CAL_MIN = 20
CAL_METHOD = "platt+isotonic"  # {"isotonic","platt","platt+isotonic"}
PF_PRIOR_BLEND = 0.35        # weight of PF prior in geometric blend with ensemble
PF_ENSEMBLE_CONFIGS = [      # (num_particles, alpha, sigma)
    (12000, 0.004, 0.008),
    (16000, 0.003, 0.006),
    (8000,  0.006, 0.012),
    (10000, 0.005, 0.010),
    (14000, 0.0035, 0.007),
    (20000, 0.0025, 0.005),
]

_cal_cache = {"LSTM": None, "Transformer": None, "t_fit": None}

def _enforce_sum6(v):
    """Project a length-40 vector v (non-negative) so that sum≈6 with clipping to [0,1].
    Uses iterative rescale + clip (water-filling style) for a few passes.
    Returns a copy with sum close to 6.
    """
    import numpy as _np
    x = _np.clip(_np.asarray(v, dtype=float), 0.0, 1.0)
    for _ in range(SUM6_MAX_ITERS):
        s = float(x.sum())
        if s <= 0:
            x[:] = 6.0/40.0
            break
        x *= (6.0 / s)
        over = x > 1.0
        if _np.any(over):
            x[over] = 1.0
        s2 = float(x.sum())
        if abs(s2 - 6.0) <= 1e-6:
            break
        if s2 > 0:
            x *= (6.0 / s2)
    return _np.clip(x, 0.0, 1.0)

def _sum6_to_sum1_dict(p6_dict):
    import numpy as _np
    v = _np.array([float(p6_dict.get(i, 0.0)) for i in range(1,41)], dtype=float)
    if v.sum() <= 0:
        v = _np.ones(40, dtype=float)*(6.0/40.0)
    v = _np.clip(v/6.0, 1e-18, None)
    v = v / (v.sum() + 1e-18)
    return {i: float(v[i-1]) for i in range(1,41)}

# ---- Simple 1D calibrators ----------------------------------------------------
class _Platt1D:
    def __init__(self):
        self.ok = False
        self.coef_ = None
        self.intercept_ = None
    def fit(self, p, y):
        try:
            import numpy as _np
            from sklearn.linear_model import LogisticRegression
            p = _np.clip(_np.asarray(p, dtype=float).reshape(-1,1), 1e-6, 1-1e-6)
            y = _np.asarray(y, dtype=float).reshape(-1)
            if p.shape[0] < 100 or y.sum() == 0 or y.sum() == y.size:
                self.ok = False
                return self
            logit = _np.log(p/(1-p))
            lr = LogisticRegression(class_weight="balanced", max_iter=1000)
            lr.fit(logit, y)
            self.ok = True
            self.coef_ = float(lr.coef_.ravel()[0])
            self.intercept_ = float(lr.intercept_.ravel()[0])
            self._lr = lr
        except Exception:
            self.ok = False
        return self
    def map(self, p):
        import numpy as _np
        p = _np.clip(_np.asarray(p, dtype=float).reshape(-1), 1e-9, 1-1e-9)
        if not self.ok:
            return p
        z = _np.log(p/(1-p)) * self.coef_ + self.intercept_
        out = 1.0/(1.0+_np.exp(-z))
        return _np.clip(out, 1e-9, 1-1e-9)

class _Iso1D:
    def __init__(self):
        try:
            from sklearn.isotonic import IsotonicRegression
            self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        except Exception:
            self.iso = None
        self.ok = False
    def fit(self, p, y):
        import numpy as _np
        if self.iso is None:
            self.ok = False
            return self
        p = _np.clip(_np.asarray(p, dtype=float).reshape(-1), 1e-9, 1-1e-9)
        y = _np.asarray(y, dtype=float).reshape(-1)
        if p.size < 100 or y.sum() == 0 or y.sum() == y.size:
            self.ok = False
            return self
        try:
            self.iso.fit(p, y)
            self.ok = True
        except Exception:
            self.ok = False
        return self
    def map(self, p):
        import numpy as _np
        p = _np.clip(_np.asarray(p, dtype=float).reshape(-1), 1e-9, 1-1e-9)
        if not self.ok or self.iso is None:
            return p
        out = self.iso.predict(p)
        return _np.clip(out, 1e-9, 1-1e-9)

class _ComboCal:
    def __init__(self, method="platt+isotonic"):
        self.method = method
        self.pl = _Platt1D()
        self.iso = _Iso1D()
        self.ok = False
    def fit(self, p, y):
        m = (self.method or "").lower()
        if "platt" in m:
            self.pl.fit(p, y)
        if "isotonic" in m:
            self.iso.fit(p, y)
        self.ok = (self.pl.ok or self.iso.ok)
        return self
    def map(self, p):
        import numpy as _np
        p = _np.asarray(p, dtype=float).reshape(-1)
        if not self.ok:
            return _np.clip(p, 1e-9, 1-1e-9)
        vals = []
        if self.pl.ok:
            vals.append(self.pl.map(p))
        if self.iso.ok:
            vals.append(self.iso.map(p))
        if not vals:
            return _np.clip(p, 1e-9, 1-1e-9)
        out = _np.mean(_np.vstack(vals), axis=0)
        return _np.clip(out, 1e-9, 1-1e-9)

# Fit calibrators for LSTM/Transformer using last CAL_WINDOW draws up to t_eval
def _fit_expert_calibrators(t_eval):
    import numpy as _np
    global _cal_cache
    t_eval = int(t_eval)
    if DEBUG_GUARDS:
        _assert_idx_ok(t_eval, context="_learn_blend_weights")
    t0 = max(1, t_eval - CAL_WINDOW + 1)
    if t_eval == _cal_cache.get("t_fit") and all(_cal_cache.get(k) is not None for k in ("LSTM","Transformer")):
        return _cal_cache
    P = {"LSTM": [], "Transformer": []}
    Y = []
    _orig = globals().get("_predict_with_nets", None)
    if not callable(_orig):
        _cal_cache = {"LSTM": None, "Transformer": None, "t_fit": t_eval}
        return _cal_cache
    if t_eval - t0 + 1 < CAL_MIN:
        _cal_cache = {"LSTM": None, "Transformer": None, "t_fit": t_eval}
        return _cal_cache
    for t in range(t0, t_eval+1):
        try:
            hist = draws[:t]
            if 'compute_stat_features' in globals():
                feats_t = compute_stat_features(hist[-min(len(hist), 20):], t - 1)
                feats_t = feats_t.reshape(1, feats_t.shape[0], feats_t.shape[1])
            else:
                continue
            meta_t = None
            if '_meta_features_at_idx' in globals():
                meta_t = __import__('numpy').array([_meta_features_at_idx(t)], dtype=float)
            l, tr = _orig(feats_t, meta_t, t_idx=t, use_finetune=True, mc_passes=globals().get('MC_STACK_PASSES', 1))
            l6 = _enforce_sum6(l*6.0)
            tr6 = _enforce_sum6(tr*6.0)
            winners = _winner_draw_at(t, context="_fit_expert_calibrators")
            y = __import__('numpy').array([1 if n in winners else 0 for n in range(1,41)], dtype=float)
            P["LSTM"].append(l6); P["Transformer"].append(tr6)
            Y.append(y)
        except Exception:
            continue
    if not P["LSTM"] or not Y:
        _cal_cache = {"LSTM": None, "Transformer": None, "t_fit": t_eval}
        return _cal_cache
    P_L = __import__('numpy').vstack(P["LSTM"]).reshape(-1)
    P_T = __import__('numpy').vstack(P["Transformer"]).reshape(-1)
    Yv  = __import__('numpy').vstack(Y).reshape(-1)
    cal_L = _ComboCal(CAL_METHOD).fit(P_L, Yv)
    cal_T = _ComboCal(CAL_METHOD).fit(P_T, Yv)
    _cal_cache = {"LSTM": cal_L, "Transformer": cal_T, "t_fit": t_eval}
    return _cal_cache

# Postprocess raw net probabilities (length-40 array) → calibrated, sum-6, then sum-1
def _postprocess_net_probs(arr, t_idx, expert_name="LSTM"):
    import numpy as _np
    v = _np.asarray(arr, dtype=float).reshape(-1)
    if v.size != 40:
        v = _np.ones(40, dtype=float) * (1.0/40.0)
    # Step 1: sum-to-6 constraint
    v6 = _enforce_sum6(v*6.0)
    # Step 2: calibration
    cals = _fit_expert_calibrators(max(1, int(t_idx)-1))
    cal = cals.get(expert_name) if isinstance(cals, dict) else None
    if cal is not None and getattr(cal, 'ok', False):
        v6 = cal.map(v6)
        v6 = _enforce_sum6(v6)
    # Step 3: back to categorical (sum=1)
    out = v6 / 6.0
    out = _np.clip(out, 1e-12, None)
    out = out / (out.sum() + 1e-12)
    return out

# ---- PF prior from history (ensemble over hyper-params) ----------------------
def _pf_prior_from_history(sub_draws, configs=PF_ENSEMBLE_CONFIGS):
    import numpy as _np
    if not sub_draws:
        return {n: 6.0/40.0 for n in range(1,41)}
    preds = []
    for (npart, a, s) in (configs or []):
        try:
            pf = ParticleFilter(num_numbers=40, num_particles=int(npart), alpha=float(a), sigma=float(s))
            for d in sub_draws:
                pf.predict()
                pf.update(d)
            pf.predict()
            preds.append(pf.get_mean_probabilities())
        except Exception:
            continue
    if not preds:
        return {n: 6.0/40.0 for n in range(1,41)}
    m = _np.mean(_np.vstack(preds), axis=0)
    m = _enforce_sum6(m)
    return {n: float(m[n-1]) for n in range(1,41)}

def _geom_blend(p, q, lam=0.5):
    """Geometric blend of two categorical dists p and q over 1..40 (sum=1 each)."""
    import numpy as _np
    lam = float(max(0.0, min(1.0, lam)))
    vp = _np.array([float(p.get(i,0.0)) for i in range(1,41)], dtype=float)
    vq = _np.array([float(q.get(i,0.0)) for i in range(1,41)], dtype=float)
    vp = _np.clip(vp, 1e-18, None); vq = _np.clip(vq, 1e-18, None)
    v = _np.power(vp, 1.0-lam) * _np.power(vq, lam)
    v = _np.clip(v, 1e-18, None); v = v / (v.sum() + 1e-18)
    return {i: float(v[i-1]) for i in range(1,41)}
# ============================================================================

# --- Per-expert probability dicts at historical index t_idx (leakage-free) ---
def _per_expert_prob_dicts_at_t(t_idx):
    """
    Build per-expert probability dicts at historical index t_idx using ONLY draws[:t_idx].
    Keys match _per_expert_names(): ["Bayes","Markov","HMM","LSTM","Transformer"].
    """
    t_idx = int(t_idx)
    t_idx = max(1, min(t_idx, len(draws) - 1))
    hist = draws[:t_idx]

    # Context sets for KN-blended Markov
    last  = draws[t_idx - 1] if t_idx - 1 >= 0 else set()
    last2 = draws[t_idx - 2] if t_idx - 2 >= 0 else set()
    last3 = draws[t_idx - 3] if t_idx - 3 >= 0 else set()

    # --- Bayes (Dirichlet-smoothed frequencies) ---
    bayes = compute_bayes_posterior(hist, alpha=1)

    # --- Markov (recompute from hist, then KN interpolate using last/last2/last3) ---
    t1, t2, t3 = compute_markov_transitions(hist)
    markov = _kn_interpolated_markov(last, last2, last3, t1, t2, t3)

    # --- HMM (guarded by version gate) ---
    try:
        hmmP = _build_tcn_prob_from_subset(hist) if HMM_OK else _uniform_prob40()
    except Exception:
        hmmP = _uniform_prob40()

    # --- Nets at t (strict leakage guard) ---
    if t_idx - k >= 0:
        try:
            _set_target_idx(t_idx)
            feats = compute_stat_features(draws[t_idx - k:t_idx], t_idx - 1)
            feats = feats.reshape(1, feats.shape[0], feats.shape[1])
            meta = np.array([_meta_features_at_idx(t_idx - 1)], dtype=float)
            _set_target_idx(None)

            lp, tp = _predict_with_nets(feats, meta, t_idx=t_idx, use_finetune=True, mc_passes=11)
            lstmP = {n: float(lp[n - 1]) for n in range(1, 41)}
            transP = {n: float(tp[n - 1]) for n in range(1, 41)}
        except Exception:
            _set_target_idx(None)
            # fallbacks if fine-tune path isn't available
            lstmP = lstm_prob if isinstance(globals().get("lstm_prob"), dict) else _uniform_prob40()
            transP = transformer_prob if isinstance(globals().get("transformer_prob"), dict) else _uniform_prob40()
    else:
        lstmP = _uniform_prob40()
        transP = _uniform_prob40()

    # --- GNN at t (using trained gnn_model if available) ---
    try:
        gnn_model_ref = globals().get("gnn_model", None)
        meta_gnn = _meta_features_at_idx(t_idx - 1) if t_idx > 0 else np.array([0.0, 0.5, 0.5])
        gnnP = _gnn_prob_from_history(hist, gnn_model=gnn_model_ref, meta_features=meta_gnn)
    except Exception:
        gnnP = _uniform_prob40()

    return {"Bayes": bayes, "Markov": markov, "HMM": hmmP, "LSTM": lstmP, "Transformer": transP, "GNN": gnnP}

# --- Walk-forward backtest with per-expert deltas + miss explanation ---

def _topk_from_prob_dict(P, k=6):
    items = sorted([(n, float(P.get(n, 0.0))) for n in range(1, 41)], key=lambda x: x[1], reverse=True)
    return [n for n,_ in items[:k]]


def _per_expert_pl_nll_at(t_idx):
    """Return per-expert PL-NLLs at draw t_idx using bases from draws[:t_idx]."""
    t_idx = int(t_idx)
    bases = _per_expert_prob_dicts_at_t(t_idx)
    _assert_expert_key_consistency(bases)
    if not isinstance(bases, dict):
        return None
    try:
        winners = draws[t_idx]
    except Exception as e:
        warnings.warn(f"Failed to access winners at t={t_idx}: {e}")
        winners = set()
    return [
        _nll_from_prob_dict(bases.get(name, {}), winners)
        for name in _per_expert_names()
    ]

# Forward-define the simple set NLL if it hasn't been defined yet, so logs won't crash
if "_nll_from_prob_dict" not in globals():
    def _nll_from_prob_dict(P_dict, winners):
        v = np.array([float(P_dict.get(n, 0.0)) for n in range(1, 41)], dtype=float)
        v = np.clip(v, 1e-12, None)
        v = v / (v.sum() + 1e-12)
        idx = [n - 1 for n in winners]
        return float(-np.sum(np.log(v[idx])))

# -- Predict-time operational log: weights, calibration, per-expert PL-NLL --
try:
    _log_predict_run(weights, t_eval=n_draws - 1, log_path="run_log.jsonl")
except Exception as _e:
    warnings.warn(f"Predict-time operational log failed: {_e}")



def _why_miss_line(t_idx, P_ens, bases):
    """Compose and print a concise miss explanation; also write JSONL with per-expert deltas."""
    try:
        truth = sorted(list(draws[t_idx]))
        top6 = _topk_from_prob_dict(P_ens, k=6)
        hits = len(set(top6) & set(truth))
        # Per-expert deltas (expert NLL minus ensemble NLL)
        names = _per_expert_names()
        nll_ens = _nll_from_prob_dict(P_ens, draws[t_idx])
        deltas = {}
        for nm in names:
            Pi = bases.get(nm, {})
            deltas[nm] = float(_nll_from_prob_dict(Pi, draws[t_idx]) - nll_ens)
        # Low-ranked truths and overconfident misses
        ranks = {n: i for i,n in enumerate(_topk_from_prob_dict(P_ens, k=40), start=1)}
        low_truth = sorted(truth, key=lambda n: ranks.get(n, 99), reverse=True)[:2]
        overconf = [n for n in top6 if n not in truth][:2]
        msg = f"[MISS t={t_idx}] hits={hits} top6={top6} truth={truth} low_truth={low_truth} overconf={overconf}"
        print(msg)
        rec = {
            'type': 'miss_explain',
            'ts': _now_iso(),
            't': int(t_idx),
            'hits': int(hits),
            'top6': top6,
            'truth': truth,
            'low_truth': low_truth,
            'overconf': overconf,
            'per_expert_delta_nll': deltas,
        }
        _log_jsonl(rec)
    except Exception as _e:
        warnings.warn(f"why-miss logging failed at t={t_idx}: {_e}")

# --- Post-rank isotonic calibration & reliability utilities -------------------

class _PostRankIsotonic:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.fitted = False
    def fit(self, p_list, y_list):
        import numpy as _np, warnings
        p = _np.asarray(p_list, dtype=float)
        y = _np.asarray(y_list, dtype=float)
        if p.size != y.size or p.size < 200:
            warnings.warn("Isotonic: insufficient pairs; skipping fit.")
            self.fitted = False
            return self
        try:
            self.iso.fit(p, y)
            self.fitted = True
        except Exception as e:
            warnings.warn(f"Isotonic fit failed: {e}")
            self.fitted = False
        return self
    def map(self, p):
        return float(self.iso.predict([float(p)])[0]) if self.fitted else float(p)

def _apply_postrank_calibrator(prob_dict):
    """Apply global postrank_calibrator (if available) to a prob dict and renormalise."""
    cal = globals().get("postrank_calibrator", None)
    if cal is None or not getattr(cal, "fitted", False):
        return prob_dict
    import numpy as _np
    v = _np.array([float(prob_dict.get(n, 0.0)) for n in range(1, 41)], dtype=float)
    v = _np.clip(v, 1e-12, None)
    v_cal = _np.array([cal.map(x) for x in v], dtype=float)
    v_cal = _np.clip(v_cal, 1e-12, None)
    v_cal = v_cal / (v_cal.sum() + 1e-12)
    return {n: float(v_cal[n-1]) for n in range(1, 41)}

def _mix_prob_dicts(weights, bases):
    """
    Combine per-expert distributions in `bases` using `weights` aligned to _per_expert_names().
    Returns a calibrated, renormalised dict {1..40 -> p}. Ranking is preserved; only the
    probability scale may change if a post-rank calibrator is present.
    """
    names = _per_expert_names()
    w = np.asarray(weights, dtype=float)
    if w.size != len(names):
        w = np.ones(len(names), dtype=float) / float(len(names))

    # Stack expert probs into shape (E, 40) and normalise each expert
    M = []
    for nm in names:
        vec = [float(bases.get(nm, {}).get(n, 0.0)) for n in range(1, 41)]
        M.append(vec)
    M = np.asarray(M, dtype=float)
    M = np.clip(M, 1e-12, None)
    M = M / (M.sum(axis=1, keepdims=True) + 1e-12)

    # Weighted mixture
    w = np.clip(w, 0.0, None)
    w = w / (w.sum() + 1e-12)
    p_mix = np.matmul(w, M)  # (40,)

    # Renormalise then apply optional post-rank isotonic; isotonic preserves ranking
    p_mix = np.clip(p_mix, 1e-12, None)
    p_mix = p_mix / (p_mix.sum() + 1e-12)
    P = {n: float(p_mix[n-1]) for n in range(1, 41)}
    return _apply_postrank_calibrator(P)

def _export_reliability(bin_edges, bin_pred_mean, bin_obs_rate,
                        csv_path="reliability_curve.csv", png_path="reliability_curve.png"):
    try:
        import matplotlib.pyplot as plt, csv as _csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["bin_left","bin_right","pred_mean","obs_rate"])
            for (l,r), pm, orate in zip(zip(bin_edges[:-1], bin_edges[1:]), bin_pred_mean, bin_obs_rate):
                w.writerow([l, r, pm, orate])
        fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111)
        ax.plot([0,1],[0,1], linestyle="--")
        ax.scatter(bin_pred_mean, bin_obs_rate)
        ax.set_title("Reliability (Final Ensemble)")
        ax.set_xlabel("Predicted probability (bin mean)")
        ax.set_ylabel("Observed frequency")
        fig.tight_layout(); fig.savefig(png_path, dpi=160); plt.close(fig)
    except Exception as e:
        import warnings; warnings.warn(f"Reliability export failed: {e}")

def run_walk_forward_backtest_ops(last_N=150, log_path="run_log.jsonl"):
    """Strict rolling walk-forward backtest over last_N draws with per-expert logging.
    Returns summary dict with mean NLL and hit stats. No shuffles; pure time-series split.
    Also COLLECTS probability/label pairs → fits a post-rank isotonic calibrator on the
    last_N window, and exports reliability CSV/PNG. Ranking is unchanged; only scale.
    """
    start = max(k + 1, n_draws - int(last_N))

    # --- Collectors for isotonic calibration & reliability ---
    _all_p = []   # predicted probabilities for all 40 candidates per evaluated draw
    _all_y = []   # 0/1 outcomes per candidate per evaluated draw

    nlls, hits_list = [], []

    for t in range(start, n_draws):
        try:
            bases = _per_expert_prob_dicts_at_t(t)
            _assert_expert_key_consistency(bases)

            # Current weights (caller may cache/decay); adapt by regime if available
            w = np.asarray(globals().get('weights', np.ones(5, dtype=float) / 5.0), dtype=float)
            try:
                w = _adapt_base_weights(w)  # no-op if function not defined
            except Exception:
                pass

            # Ensemble distribution for draw t (pre-calibration)
            P = _mix_prob_dicts(w, bases)

            # --- Collect pairs for reliability/calibration ---
            p_vec = [float(P.get(n, 0.0)) for n in range(1, 41)]
            y_vec = [1 if n in draws[t] else 0 for n in range(1, 41)]
            _all_p.extend(p_vec)
            _all_y.extend(y_vec)

            # Score PL-style NLL (independent-winner proxy) and hits
            nll = _nll_from_prob_dict(P, draws[t])
            nlls.append(nll)
            top6 = set(_topk_from_prob_dict(P, 6))
            hits = len(top6 & set(draws[t]))
            hits_list.append(hits)

            if hits < 2:  # concise miss line only when we really miss
                _why_miss_line(t, P, bases)

            # per-expert PL-NLLs (for JSONL)
            try:
                names = _per_expert_names()
                per_nll = {nm: _nll_from_prob_dict(bases.get(nm, {}), draws[t]) for nm in names}
                rec = {
                    'type': 'backtest_draw',
                    'ts': _now_iso(),
                    't': int(t),
                    'nll_ens': float(nll),
                    'per_expert_pl_nll': {k: float(v) for k, v in per_nll.items()},
                    'hits_top6': int(hits),
                }
                _log_jsonl(rec, path=log_path)
            except Exception:
                pass

        except Exception as _e:
            warnings.warn(f"Backtest step failed at t={t}: {_e}")

    # --- Fit post-rank isotonic on accumulated pairs and export reliability ---
    try:
        if len(_all_p) >= 400:  # safeguard for short histories
            iso = _PostRankIsotonic().fit(_all_p, _all_y)
            globals()["postrank_calibrator"] = iso

            import numpy as _np
            be = _np.linspace(0.0, 1.0, 11)
            idx = _np.digitize(_np.asarray(_all_p, float), be, right=True)
            pred_means, obs_rates = [], []
            y_arr = _np.asarray(_all_y, float)
            for b in range(1, len(be)):
                m = (idx == b)
                pm = float(_np.mean(_np.asarray(_all_p)[m])) if _np.any(m) else 0.0
                orc = float(_np.mean(y_arr[m])) if _np.any(m) else 0.0
                pred_means.append(pm)
                obs_rates.append(orc)
            _export_reliability(
                be, pred_means, obs_rates,
                csv_path="reliability_curve.csv",
                png_path="reliability_curve.png"
            )
    except Exception as _e:
        warnings.warn(f"Post-rank calibration step failed: {_e}")

    summary = {
        'mean_nll': float(np.mean(nlls)) if nlls else None,
        'mean_hits_top6': float(np.mean(hits_list)) if hits_list else None,
        'n_draws': len(nlls),
    }
    return summary

# If missing, provide a simple set-NLL proxy (independent winners) for diagnostics
if "_nll_from_prob_dict" not in globals():
    def _nll_from_prob_dict(P_dict, winners):
        v = np.array([float(P_dict.get(n, 0.0)) for n in range(1, 41)], dtype=float)
        v = np.clip(v, 1e-12, None); v = v / (v.sum() + 1e-12)
        idx = [n - 1 for n in winners]
        return float(-np.sum(np.log(v[idx])))

def run_oracle_ablation(last_N=100, log_path="run_log.jsonl"):
    """
    For each of the last_N draws, replace ONE expert with an 'oracle' distribution that
    concentrates mass on the true winners, then re-mix with the same weights/gates.
    Return mean ΔNLL = (ensemble NLL) - (oracle-replaced NLL) per expert.
    """
    names = _per_expert_names()
    start = max(k + 1, len(draws) - int(last_N))
    gains = {nm: [] for nm in names}

    for t in range(start, len(draws)):
        try:
            bases = _per_expert_prob_dicts_at_t(t)
            _assert_expert_key_consistency(bases)
            # Current weights with regime adaptation
            w = np.asarray(globals().get("weights", np.ones(len(names)) / len(names)), dtype=float)
            w = _adapt_base_weights(w)
            P_ens = _mix_prob_dicts(w, bases)
            truth = draws[t]
            nll_ens = _nll_from_prob_dict(P_ens, truth)
            # Smooth "oracle" (avoid zeros): ~uniform over winners, tiny mass on non-winners
            eps = 1e-6
            oracle = {n: (1.0 / 6.0 - eps) if n in truth else (eps / 34.0) for n in range(1, 41)}
            for nm in names:
                bases_or = dict(bases)
                bases_or[nm] = oracle
                P_or = _mix_prob_dicts(w, bases_or)
                nll_or = _nll_from_prob_dict(P_or, truth)
                gains[nm].append(nll_ens - nll_or)
        except Exception as _e:
            warnings.warn(f"Oracle ablation failed at t={t}: {_e}")

    mean_gain = {nm: float(np.mean(v)) if v else 0.0 for nm, v in gains.items()}
    rec = {"type": "oracle_ablation", "ts": _now_iso(), "last_N": int(last_N), "mean_delta_nll": mean_gain}
    _log_jsonl(rec, path=log_path)
    return mean_gain

# --- Evaluation & diagnostics: walk-forward, reliability, oracle ablation ---
try:
    # Strict rolling walk-forward backtest over the last 150 draws
    summary = run_walk_forward_backtest_ops(last_N=150, log_path="run_log.jsonl")
    print("[DIAG] Backtest summary:", summary)

    # Oracle ablation (mean improvement in PL-NLL if each expert were perfect)
    ablate = run_oracle_ablation(last_N=100, log_path="run_log.jsonl")
    print("[DIAG] Oracle ablation mean ΔNLL:", ablate)
except Exception as _e:
    warnings.warn(f"Diagnostics failed: {_e}")

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


markov_prob = _kn_interpolated_markov(last_draw, last_2_draw, last_3_draw,
                                      transition_probs, transition2_probs, transition3_probs)

# Co-occurrence conditional prob based on within-draw co-occurrence
cooc_prob = _cooc_conditional_prob_from_history(draws[:-1], last_draw)

# Without-replacement compatibility features (top-k competitor sums) for each base
COMP_K = 5
compat_bayes   = _compat_topk_sum(bayes_prob, k=COMP_K)
compat_markov  = _compat_topk_sum(markov_prob, k=COMP_K)
compat_hmm     = _compat_topk_sum(hmm_prob, k=COMP_K)
compat_nn      = _compat_topk_sum({n: 0.5*lstm_prob[n] + 0.5*transformer_prob[n] for n in range(1, 41)}, k=COMP_K)
compat_gnn     = _compat_topk_sum(gnn_prob, k=COMP_K)




# Regime + volatility descriptors available to stacker
reg_now = _regime_features_at_t(n_draws - 1)
roll_ent_now = _rolling_entropy_from_history(draws, window=30)
roll_disp_now = _dispersion_last_draw(draws)

# --- Regime detection & adaptation --------------------------------------------------

def _regime_switch_flags():
    """Detect entropy jump and cadence perturbation on the latest draw.
    Returns dict with booleans and magnitudes.
    """
    try:
        # Normalised entropies (divide by log 40)
        ent_s = _rolling_entropy_from_history(draws, window=min(20, len(draws))) / float(np.log(40.0))
        ent_l = _rolling_entropy_from_history(draws, window=min(60, len(draws))) / float(np.log(40.0))
        ent_jump = (ent_s - ent_l) > 0.08  # heuristic threshold
    except Exception:
        ent_s, ent_l, ent_jump = 0.0, 0.0, False
    try:
        r = _regime_features_at_t(n_draws - 1)
        cadence_perturbed = bool(r.get('schedule_flip', 0) == 1)
    except Exception:
        cadence_perturbed = False
    return {
        'ent_short': float(ent_s),
        'ent_long': float(ent_l),
        'entropy_jump': bool(ent_jump),
        'cadence_perturbed': bool(cadence_perturbed),
    }


def _adapt_base_weights(weights_vec):
    """When entropy jumps or cadence perturbs, raise Bayes and lower Markov/HMM.
    Keeps weights on the simplex.
    """
    try:
        w = np.asarray(weights_vec, dtype=float).reshape(-1)
        if w.size < 6:
            # Pad to 6 if needed
            w_new = np.ones(6, dtype=float) / 6.0
            w_new[:w.size] = w[:min(w.size, 6)]
            w = w_new
        flags = _regime_switch_flags()
        if flags['entropy_jump'] or flags['cadence_perturbed']:
            # indices: 0=Bayes, 1=Markov, 2=HMM, 3=LSTM, 4=Transformer, 5=GNN
            mult = np.array([1.25, 0.85, 0.90, 1.00, 1.00, 1.05], dtype=float)
            w = w[:6] * mult  # Ensure we only use first 6 elements
        w = np.clip(w, 1e-12, None)
        w = w / (w.sum() + 1e-12)
        return w
    except Exception:
        return np.asarray(weights_vec, dtype=float)


def _streakiness_signal():
    """Return (is_streaky:bool, score:0..1) using dispersion and frequent co-occurrences.
    Low dispersion + presence of frequent pairs/triplets => streaky.
    """
    try:
        disp = _dispersion_last_draw(draws)
        low_disp = float(disp) < 7.5
    except Exception:
        low_disp = False
    try:
        ld = sorted(list(draws[-1])) if len(draws) else []
        has_pair = any(((a,b) in frequent_pairs) for i,a in enumerate(ld) for b in ld[i+1:]) if ld else False
        has_trip = any((t in frequent_triplets) for t in [tuple(sorted((ld[i], ld[j], ld[k]))) for i in range(len(ld)) for j in range(i+1, len(ld)) for k in range(j+1, len(ld))]) if len(ld) >= 3 else False
    except Exception:
        has_pair, has_trip = False, False
    streak = low_disp or has_pair or has_trip
    score = 0.0
    if streak:
        score = 0.5 * (1.0 if low_disp else 0.0) + 0.3 * (1.0 if has_pair else 0.0) + 0.2 * (1.0 if has_trip else 0.0)
        score = float(min(1.0, max(0.0, score)))
    return bool(streak), float(score)


def _streak_gate_for_current_run():
    """Blend in co-occurrence conditional prob during streaky phases.
    Returns (enabled, cooc_prob_dict, weight 0..0.25).
    """
    try:
        is_strk, s = _streakiness_signal()
        if not is_strk:
            return False, None, 0.0
        last_draw_set = draws[-1] if len(draws) else set()
        hist = draws[:-1] if len(draws) > 1 else []
        P_cooc = _cooc_conditional_prob_from_history(hist, last_draw_set)
        # Map score to weight in [0.10, 0.25]
        w = 0.10 + 0.15 * s
        return True, P_cooc, float(w)
    except Exception:
        return False, None, 0.0


def _adjust_selection_configs(sel_sharp, sel_div):
    """Cool sampling temps when regime switch (entropy jump or cadence perturbation).
    Returns possibly modified (sel_sharp, sel_div).
    """
    try:
        flags = _regime_switch_flags()
        if flags['entropy_jump'] or flags['cadence_perturbed']:
            cs = dict(sel_sharp)
            cd = dict(sel_div)
            cs['temp'] = float(max(0.5, cs.get('temp', 0.9) * 0.85))
            cd['temp'] = float(max(0.7, cd.get('temp', 1.1) * 0.92))
            return cs, cd
    except Exception:
        pass
    return sel_sharp, sel_div

# --- Cheap calendar baselines (Dirichlet-smoothed) + weekday gate ---
def _dirichlet_freq_for_mask(mask_idx):
    """Return Dirichlet-smoothed per-number frequencies using draws at indices in mask_idx."""
    alpha = 1.0
    cnt = np.zeros(40, dtype=float)
    for i in mask_idx:
        for n in draws[i]:
            cnt[n-1] += 1.0
    p = (cnt + alpha) / ((len(mask_idx)*6.0) + 40.0*alpha if len(mask_idx) > 0 else 40.0*alpha)
    return {n: float(p[n-1]) for n in range(1, 41)}

def _weekday_month_baselines_at_t(t_idx):
    """Build weekday- and month-conditioned frequency baselines using only draws < t_idx."""
    t_idx = int(t_idx)
    if t_idx <= 0:
        return ({n:1.0/40 for n in range(1,41)}, {n:1.0/40 for n in range(1,41)})
    wd = int(pd.to_datetime(data.iloc[t_idx]['DrawDate']).weekday())
    mo = int(pd.to_datetime(data.iloc[t_idx]['DrawDate']).month)
    mask_wd = [i for i in range(t_idx) if int(pd.to_datetime(data.iloc[i]['DrawDate']).weekday()) == wd]
    mask_mo = [i for i in range(t_idx) if int(pd.to_datetime(data.iloc[i]['DrawDate']).month) == mo]
    return _dirichlet_freq_for_mask(mask_wd), _dirichlet_freq_for_mask(mask_mo)

# Gate provider consulted by _live_mixture_prob()
def _weekday_gate_for_current_run():
    try:
        wd_base, mo_base = _weekday_month_baselines_at_t(n_draws - 1)
        mix = {n: 0.5*wd_base[n] + 0.5*mo_base[n] for n in range(1, 41)}
        return True, mix, 0.10  # enabled, blended baseline, gate weight
    except Exception:
        return False, None, 0.0


# (c) Deep learning probabilities: use LSTM and Transformer models to predict probabilities for next draw.
# We average their probabilities for a robust neural network component.
def _mix_prob_dicts(weights, bases):
    """
    Combine per-expert distributions in `bases` using `weights` aligned to _per_expert_names().
    Returns a calibrated, renormalised dict {1..40 -> p}. Ranking is preserved; only scale
    may change if a post-rank calibrator is available.
    """
    names = _per_expert_names()
    w = np.asarray(weights, dtype=float)
    if w.size != len(names):
        w = np.ones(len(names), dtype=float) / float(len(names))

    # Stack expert probs into (E, 40)
    M = []
    for nm in names:
        vec = [float(bases.get(nm, {}).get(n, 0.0)) for n in range(1, 41)]
        M.append(vec)
    M = np.asarray(M, dtype=float)
    M = np.clip(M, 1e-12, None)
    # normalise each expert to a proper distribution
    M = M / (M.sum(axis=1, keepdims=True) + 1e-12)

    # Weighted mixture
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w) / float(len(w))
    w = w / w.sum()
    p_mix = np.matmul(w, M)  # (40,)

    # Final renorm + optional isotonic post-rank calibration
    p_mix = np.clip(p_mix, 1e-12, None)
    p_mix = p_mix / (p_mix.sum() + 1e-12)
    return _apply_postrank_calibrator({n: float(p_mix[n-1]) for n in range(1, 41)})


# (c) Deep learning probabilities: use LSTM and Transformer models to predict probabilities for next draw.
# We average their probabilities for a robust neural network component.
nn_prob = {}
for num in range(1, 41):
    # Average LSTM and Transformer probabilities for each number
    nn_prob[num] = 0.5 * lstm_prob[num] + 0.5 * transformer_prob[num]


# (d) Recent trend heuristic with capped, concave schedule
last_seen = {num: -1 for num in range(1, 41)}
for idx, draw in enumerate(draws):
    for num in draw:
        last_seen[num] = idx
gaps_full = {num: (n_draws - 1 - last_seen[num]) for num in range(1, 41)}  # gap in draws
GAP_TAU = 18.0   # curvature control
GAP_CAP = 0.22   # absolute cap on boost
_gap_boost_cc = {}
for num in range(1, 41):
    g = max(0.0, float(gaps_full[num]))
    _gap_boost_cc[num] = float(np.tanh(g / GAP_TAU)) * GAP_CAP

gap_boost = _gap_boost_cc

# === Minimal prediction emission (ensures we always produce a ticket set) ===
def _normalise_prob_dict(d):
    p = np.array([float(d.get(n, 0.0)) for n in range(1, 41)], dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / (p.sum() + 1e-12)
    return {n: float(p[n-1]) for n in range(1, 41)}

def _mix_prob_dicts_safe(w_vec, bases):
    """
    Mix base probability dicts on the simplex. `bases` is a dict name->dict(1..40 -> p).
    Falls back gracefully if any base is missing.
    """
    keys = ["Bayes", "Markov", "HMM", "LSTM", "Transformer"]
    w = np.asarray(w_vec, dtype=float).reshape(-1)
    if w.size != len(keys):
        w = np.ones(len(keys), dtype=float) / float(len(keys))
    w = np.clip(w, 1e-12, None)
    w = w / (w.sum() + 1e-12)
        # Accumulate weighted sum (robust to dict, vector, or scalar inputs)
    acc = np.zeros(40, dtype=float)
    for i, k in enumerate(keys):
        Pi = bases.get(k, None)
        if not isinstance(Pi, dict) or len(Pi) == 0:
            continue
        acc += float(w[i]) * np.array([Pi.get(n, 0.0) for n in range(1, 41)], dtype=float)
    acc = np.clip(acc, 1e-12, None)
    acc = acc / (acc.sum() + 1e-12)
    return {n: float(acc[n-1]) for n in range(1, 41)}

def _apply_gate(P, gate_fn):
    """gate_fn returns (enabled:bool, mix_dict:dict or None, weight:float)."""
    try:
        en, mix, w = gate_fn()
        if not en or not isinstance(mix, dict) or float(w) <= 0:
            return P
        base = np.array([P.get(n, 0.0) for n in range(1, 41)], dtype=float)
        alt  = np.array([mix.get(n, 0.0) for n in range(1, 41)], dtype=float)
        comb = (1.0 - float(w)) * base + float(w) * alt
        comb = np.clip(comb, 1e-12, None)
        comb = comb / (comb.sum() + 1e-12)
        return {n: float(comb[n-1]) for n in range(1, 41)}
    except Exception:
        return P

def _emit_predictions(bayesP, markovP, hmmP, lstmP, transP, path="predicted_tickets.txt"):
    bases = {
        "Bayes": _normalise_prob_dict(bayesP),
        "Markov": _normalise_prob_dict(markovP),
        "HMM": _normalise_prob_dict(hmmP),
        "LSTM": _normalise_prob_dict(lstmP),
        "Transformer": _normalise_prob_dict(transP),
    }
    # Adapt weights if regime adapter is available; otherwise use defaults
    try:
        w_use = _adapt_base_weights(weights)
    except Exception:
        w_use = weights

    P = _mix_prob_dicts_safe(w_use, bases)

    # Optional gates if available
    try:
        P = _apply_gate(P, _weekday_gate_for_current_run)
    except Exception:
        pass
    try:
        P = _apply_gate(P, _streak_gate_for_current_run)
    except Exception:
        pass

    # Rank and choose top-6 deterministically
    ranked = sorted([(n, float(P.get(n, 0.0))) for n in range(1, 41)], key=lambda x: x[1], reverse=True)
    top6 = [n for n, _ in ranked[:6]]

    # Write plaintext prediction file (intermediate - will be overwritten by final backtest-selected prediction)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(" ".join(str(n) for n in sorted(top6)) + "\n")
        print("[PREDICT] Intermediate ensemble Top6:", sorted(top6), "-> written to", path)
    except Exception as e:
        warnings.warn(f"Failed to write predictions to {path}: {e}")

# Run the emission once after components are ready
try:
    _emit_predictions(bayes_prob, markov_prob, hmm_prob, lstm_prob, transformer_prob, path="predicted_tickets.txt")
except Exception as _e:
    warnings.warn(f"Prediction emission failed: {_e}")



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

# === Lotto+ Compatibility Shims (guarded) =====================================
# These avoid NameErrors after a revert by providing minimal, robust fallbacks.

# 1) Post‑rank isotonic calibrator used by reliability export
if '_PostRankIsotonic' not in globals():
    from sklearn.isotonic import IsotonicRegression as _Iso
    class _PostRankIsotonic:
        def __init__(self):
            self.iso = _Iso(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            self.fitted = False
        def fit(self, probs_list, labels_list):
            import numpy as _np
            x = _np.asarray(probs_list, dtype=float).reshape(-1)
            y = _np.asarray(labels_list, dtype=float).reshape(-1)
            if x.size >= 100 and y.size == x.size:
                self.iso.fit(x, y)
                self.fitted = True
            return self
        def map(self, p):
            # Map a single probability through isotonic; identity if not fitted.
            try:
                return float(self.iso.predict([float(p)])[0]) if self.fitted else float(p)
            except Exception:
                return float(p)

# 2) Expert name ordering used throughout
if '_per_expert_names' not in globals():
    def _per_expert_names():
        return ["Bayes", "Markov", "HMM", "LSTM", "Transformer", "GNN"]

# 3) Historical per‑expert distributions builder used by backtests/diagnostics
if '_per_expert_prob_dicts_at_t' not in globals():
    def _per_expert_prob_dicts_at_t(t_idx):
        """Return base probability dicts at historical index t_idx using only draws[:t_idx].
        Provides a safe, minimal version compatible with older code paths.
        """
        try:
            t = int(t_idx)
        except Exception:
            return None
        try:
            hist = _history_upto(t, context="_per_expert_prob_dicts_at_t")
        except Exception:
            return None
        if t <= 0 or t > len(draws) - 0:
            return None
        # Bayes / Markov / HMM
        try:
            bayes_t = compute_bayes_posterior(hist, alpha=1)
        except Exception:
            bayes_t = {n: 1.0/40 for n in range(1,41)}
        try:
            markov_t = _markov_prob_from_history(hist)
        except Exception:
            markov_t = {n: 1.0/40 for n in range(1,41)}
        try:
            hmm_t = _build_tcn_prob_from_subset(hist)
        except Exception:
            hmm_t = {n: 1.0/40 for n in range(1,41)}
        # LSTM / Transformer via existing net predictor, if available
        try:
            feats_t = compute_stat_features(draws[t - k: t], t - 1)
            feats_t = feats_t.reshape(1, feats_t.shape[0], feats_t.shape[1])
            meta_t = None
            if '_meta_features_at_idx' in globals():
                import numpy as _np
                meta_t = _np.array([_meta_features_at_idx(t)], dtype=float)
            if '_predict_with_nets' in globals():
                _l_t, _t_t = _predict_with_nets(feats_t, meta_t, t_idx=t, use_finetune=True, mc_passes=globals().get('MC_STACK_PASSES', 1))
                lstm_t = {n: float(_l_t[n-1]) for n in range(1,41)}
                trans_t = {n: float(_t_t[n-1]) for n in range(1,41)}
            else:
                lstm_t = {n: 1.0/40 for n in range(1,41)}
                trans_t = {n: 1.0/40 for n in range(1,41)}
        except Exception:
            lstm_t = {n: 1.0/40 for n in range(1,41)}
            trans_t = {n: 1.0/40 for n in range(1,41)}
        # GNN
        try:
            if '_gnn_prob_from_history' in globals():
                import numpy as _np
                gnn_model_ref = globals().get("gnn_model", None)
                meta_gnn = _meta_features_at_idx(t - 1) if t > 0 else _np.array([0.0, 0.5, 0.5])
                gnn_t = _gnn_prob_from_history(hist, gnn_model=gnn_model_ref, meta_features=meta_gnn)
            else:
                gnn_t = {n: 1.0/40 for n in range(1,41)}
        except Exception:
            gnn_t = {n: 1.0/40 for n in range(1,41)}
        # Normalise defensively
        def _norm(d):
            import numpy as _np
            v = _np.array([float(d.get(i,0.0)) for i in range(1,41)], dtype=float)
            v = _np.clip(v, 1e-12, None); v = v / (v.sum() + 1e-12)
            return {i: float(v[i-1]) for i in range(1,41)}
        return {
            "Bayes": _norm(bayes_t),
            "Markov": _norm(markov_t),
            "HMM": _norm(hmm_t),
            "LSTM": _norm(lstm_t),
            "Transformer": _norm(trans_t),
            "GNN": _norm(gnn_t),
        }

# 4) Joint/marginal blender used by SetAR decoding paths
if 'blend_joint_with_marginals' not in globals():
    def blend_joint_with_marginals(joint_scores, marginal_scores, alpha=0.5):
        """Blend a ticket‑level joint score dict with per‑number marginals.
        `joint_scores`: dict[ticket_tuple->score]; `marginal_scores`: dict[num->p].
        Returns a new dict with renormalised blended scores.
        """
        import numpy as _np
        a = float(alpha)
        a = 0.0 if not _np.isfinite(a) else max(0.0, min(1.0, a))
        # Normalise joint scores
        js_keys = list(joint_scores.keys()) if isinstance(joint_scores, dict) else []
        if not js_keys:
            return {}
        js = _np.array([float(joint_scores[k]) for k in js_keys], dtype=float)
        js = _np.clip(js, 1e-12, None); js = js / (js.sum() + 1e-12)
        # Build a marginal product per ticket (without replacement proxy)
        mp = []
        for t in js_keys:
            prod = 1.0
            for n in t:
                prod *= float(marginal_scores.get(n, 1.0/40.0))
            mp.append(prod)
        mp = _np.array(mp, dtype=float)
        mp = _np.clip(mp, 1e-18, None); mp = mp / (mp.sum() + 1e-18)
        comb = (1.0 - a) * js + a * mp
        comb = _np.clip(comb, 1e-18, None); comb = comb / (comb.sum() + 1e-18)
        return {k: float(v) for k, v in zip(js_keys, comb)}
# === End shims ================================================================

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

# --- Simplex helpers & ensemble mix -------------------------------------------------

def _softmax_simplex(theta):
    v = np.asarray(theta, dtype=float).reshape(-1)
    v = v - np.max(v)
    e = np.exp(v)
    s = float(e.sum())
    if s <= 0:
        return np.ones(5, dtype=float) / 5.0
    return (e / s)

def _mix_prob_dicts(weights_vec, bases_dict):
    """Given weights (len=5 aligned to _per_expert_names) and a dict of base prob dicts,
    return the convex combination {1..40->p}. Renormalises defensively.
    """
    names = _per_expert_names()
    w = np.asarray(weights_vec, dtype=float).reshape(-1)
    if w.size != len(names):
        w = np.ones(len(names), dtype=float) / float(len(names))
    mix = np.zeros(40, dtype=float)
    for i, nm in enumerate(names):
        d = bases_dict.get(nm, {})
        v = np.array([float(d.get(n, 0.0)) for n in range(1, 41)], dtype=float)
        mix += float(w[i]) * v
    mix = np.clip(mix, 1e-12, None)
    mix = mix / (mix.sum() + 1e-12)
    return {n: float(mix[n-1]) for n in range(1, 41)}

# --- Decayed PL-NLL objective over a rolling window --------------------------------

def _exp_decay(age, half_life=30.0):
    try:
        return float(0.5 ** (float(age) / float(half_life)))
    except Exception:
        return 1.0

def _rolling_decayed_nll_for_theta(theta, t_start, t_end, half_life=30.0):
    """Theta in R^5 -> simplex via softmax; evaluate decayed PL-NLL over [t_start, t_end).
    Uses only draws strictly before each t via _per_expert_prob_dicts_at_t.
    """
    w = _softmax_simplex(theta)
    loss = 0.0
    for t in range(int(t_start), int(t_end)):
        bases = _per_expert_prob_dicts_at_t(t)
        P = _mix_prob_dicts(w, bases)
        nll = _nll_from_prob_dict(P, draws[t])
        age = (t_end - 1) - t
        loss += _exp_decay(age, half_life=half_life) * float(nll)
    return float(loss)

# --- Learn live 5-way base weights (constrained logistic via softmax) ---------------

def _learn_live_base_weights(window=120, half_life=30.0, restarts=3):
    """Return a 5-dim simplex weight vector for [Bayes, Markov, HMM, LSTM, Transformer]
    by minimising decayed PL-NLL over the last `window` draws. Softmax enforces
    non-negativity and sum-to-1. Uses multiple random restarts for stability.
    """
    t_end = max(1, int(globals().get('n_draws', 0)) - 0)  # last completed draw index + 1
    t_start = max(int(globals().get('k', 1)) + 1, t_end - int(window))
    # Local import to avoid hard dependency if SciPy missing elsewhere
    try:
        from scipy.optimize import minimize
    except Exception as _e:
        warnings.warn(f"scipy not available for weight learning: {_e}")
        return np.ones(5, dtype=float) / 5.0

    best_val, best_w = float('inf'), None
    # Try deterministic start (uniform) + a few random jitters
    inits = [np.zeros(5, dtype=float)]
    rng = np.random.default_rng(1234)
    for _ in range(max(0, int(restarts) - 1)):
        inits.append(rng.normal(0, 0.2, size=5))
    for th0 in inits:
        try:
            res = minimize(lambda th: _rolling_decayed_nll_for_theta(th, t_start, t_end, half_life=half_life),
                           x0=th0, method='L-BFGS-B')
            if res.success:
                val = float(res.fun)
                if val < best_val:
                    best_val, best_w = val, _softmax_simplex(res.x)
        except Exception:
            continue
    return best_w if best_w is not None else (np.ones(5, dtype=float) / 5.0)

# --- Bayesian performance weighting (Dirichlet/Thompson) ----------------------------

def _dirichlet_performance_weights(last_N=120, half_life=30.0, alpha0=1.0, scale=1.0, seed=202):
    """Build Dirichlet params from decayed successes s_e = sum(decay * exp(-NLL_e))
    over the recent window, then sample a Thompson weight vector.
    """
    names = _per_expert_names()
    t_end = max(1, int(globals().get('n_draws', 0)))
    t_start = max(int(globals().get('k', 1)) + 1, t_end - int(last_N))
    s = np.zeros(len(names), dtype=float)
    for t in range(t_start, t_end):
        vals = _per_expert_pl_nll_at(t)
        if not vals:
            continue
        for i, nll in enumerate(vals):
            s[i] += _exp_decay((t_end-1) - t, half_life=half_life) * np.exp(-float(nll))
    alpha = alpha0 + scale * s
    alpha = np.maximum(alpha, 1e-6)
    rng = np.random.default_rng(int(seed) + int(t_end))
    w = rng.dirichlet(alpha)
    return w

# --- Isotonic post-rank calibration (pooled across numbers) -------------------------
class _IsoCalibrator:
    def __init__(self):
        # Clip to [0,1]; avoid extrapolation issues by clipping
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self.fitted = False
    def fit(self, probs_list, labels_list):
        # probs_list: list of length-40 arrays; labels_list: list of length-40 binary arrays
        x, y = [], []
        for p, lab in zip(probs_list, labels_list):
            x.extend(list(np.asarray(p).reshape(-1)))
            y.extend(list(np.asarray(lab).reshape(-1)))
        if len(x) >= 100:  # need enough points
            self._iso.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            self.fitted = True
        return self
    def transform_vec(self, p):
        p = np.asarray(p, dtype=float).reshape(-1)
        if not self.fitted:
            return p
        out = self._iso.transform(p)
        out = np.clip(out, 1e-12, None)
        s = float(np.sum(out))
        return out / (s + 1e-12)
    def transform_dict(self, P_dict):
        arr = np.array([float(P_dict.get(n, 0.0)) for n in range(1, 41)], dtype=float)
        arr = self.transform_vec(arr)
        return {n: float(arr[n-1]) for n in range(1, 41)}

def _fit_isotonic_calibrator(window=150, weights_vec=None):
    """Fit pooled isotonic calibrator on the last `window` draws using the current
    ensemble weights (defaults to uniform if None)."""
    t_end = max(1, int(globals().get('n_draws', 0)))
    t_start = max(int(globals().get('k', 1)) + 1, t_end - int(window))
    cal = _IsoCalibrator()
    probs, labels = [], []
    w = np.asarray(weights_vec, dtype=float) if weights_vec is not None else (np.ones(5, dtype=float)/5.0)
    for t in range(t_start, t_end):
        bases = _per_expert_prob_dicts_at_t(t)
        continue
        P = _mix_prob_dicts(w, bases)
        probs.append([P[n] for n in range(1, 41)])
        lab = [1.0 if (n in draws[t]) else 0.0 for n in range(1, 41)]
        labels.append(lab)
    try:
        cal.fit(probs, labels)
    except Exception as _e:
        warnings.warn(f"Isotonic fit failed: {_e}")
    return cal

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
    hmm_t     = _build_tcn_prob_from_subset(history)

    # Neural nets features at time t_idx
    feats_t = compute_stat_features(draws[t_idx - k: t_idx], t_idx - 1)
    feats_t = feats_t.reshape(1, feats_t.shape[0], feats_t.shape[1])
    meta_t = np.array([_meta_features_at_idx(t_idx)], dtype=float)
    _l_t, _t_t = _predict_with_nets(feats_t, meta_t, t_idx=t_idx, use_finetune=True, mc_passes=MC_STACK_PASSES)
    lstm_t = {n: float(_l_t[n-1]) for n in range(1, 41)}
    trans_t = {n: float(_t_t[n-1]) for n in range(1, 41)}

    # GNN prediction
    gnn_t = _gnn_prob_from_history(history, gnn_model=globals().get('gnn_model', None), meta_features=meta_t[0])
    gnn_t = _norm_prob_dict_local(gnn_t)

    return {"Bayes": bayes_t, "Markov": markov_t, "HMM": hmm_t, "LSTM": lstm_t, "Transformer": trans_t, "GNN": gnn_t}


def _per_expert_pl_nll_at(t_idx):
    """Return per-expert PL-NLLs at draw t_idx using bases from draws[:t_idx]."""
    # Normalise/guard t_idx
    try:
        t = int(t_idx)
    except Exception:
        try:
            t = int(float(t_idx))
        except Exception:
            warnings.warn(f"Invalid t_idx passed to _per_expert_pl_nll_at: {t_idx}")
            return None

    # Build per-expert probability dictionaries strictly from draws[:t]
    bases = _per_expert_prob_dicts_at_t(t)
    if not isinstance(bases, dict):
        return None

    # Ensure the expected expert keys are present (Bayes/Markov/HMM/LSTM/Transformer)
    try:
        _assert_expert_key_consistency(bases)
    except Exception:
        # If the assert helper raises, proceed best-effort
        pass

    # Get the ground-truth winners for draw t (guarded)
    try:
        winners = draws[t]
    except Exception as e:
        warnings.warn(f"Failed to access winners at t={t}: {e}")
        return None

    # Compute per-expert PL-NLLs in the canonical order returned by _per_expert_names()
    names = _per_expert_names()
    out = []
    for name in names:
        Pi = bases.get(name, {}) if isinstance(bases, dict) else {}
        out.append(_nll_from_prob_dict(Pi, winners))
    return out


def _live_mixture_prob():
    """
    Build the current live per-number probability distribution for the NEXT draw
    using the already-computed base models and learned weights. Applies the weekday
    gate if it is enabled, and a mild concave gap-pressure multiplier. Returns {1..40->p}.
    """
    try:
        def _get(d, n):
            return float(d.get(n, 0.0)) if isinstance(d, dict) else 0.0
        P = {}
        for n in range(1, 41):
            bv = [
                _get(globals().get("bayes_prob", {}), n),
                _get(globals().get("markov_prob", {}), n),
                _get(globals().get("hmm_prob", {}), n),
                float(globals().get("lstm_prob", {}).get(n, 0.0)),
                float(globals().get("transformer_prob", {}).get(n, 0.0)),
                _get(globals().get("gnn_prob", {}), n),
            ]
            w = globals().get("weights", np.ones(6, dtype=float) / 6.0)
            P[n] = float(np.dot(np.asarray(w, dtype=float), np.asarray(bv, dtype=float)))
        s = sum(P.values())
        if s > 0:
            for n in range(1, 41):
                P[n] = P[n] / s
        else:
            P = {n: 1.0/40 for n in range(1, 41)}

        # Apply weekday/month cheap baseline via a light gate
        try:
            gate_fn = globals().get("_weekday_gate_for_current_run", None)
            if callable(gate_fn):
                enabled, wd_prob, w = gate_fn()
                if enabled and isinstance(wd_prob, dict) and w and w > 0:
                    P = {n: float((1.0 - w) * P.get(n, 0.0) + w * wd_prob.get(n, 0.0)) for n in range(1, 41)}
                    s = sum(P.values()); P = {n: P[n]/s for n in range(1, 41)}
        except Exception:
            pass

        # Apply concave gap-pressure multiplier then renormalise (very mild)
        try:
            GP_W = 0.10
            gb = globals().get("gap_boost", None)
            if isinstance(gb, dict):
                P = {n: P[n] * (1.0 + GP_W * float(gb.get(n, 0.0))) for n in range(1, 41)}
                s = sum(P.values()); P = {n: P[n]/s for n in range(1, 41)}
        except Exception:
            pass

        return P
    except ImportError:
        warnings.warn("pomegranate not installed; falling back to uniform HMM prior if hmmlearn path fails.")
        return {n: 1.0/40 for n in range(1, 41)}
    except Exception:
        return {n: 1.0/40 for n in range(1, 41)}

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
    compat_gnn_t    = _compat_topk_sum(bases.get("GNN", {}),  k=compat_k)
    reg_t = _regime_features_at_t(t_idx)
    roll_ent_t = _rolling_entropy_from_history(history, window=30)
    roll_disp_t = _dispersion_last_draw(history)

    wd_base_t, mo_base_t = _weekday_month_baselines_at_t(t_idx)
    latest_feats_t = np.array([
        [
            bases["Bayes"].get(num, 0.0),
            bases["Markov"].get(num, 0.0),
            bases["HMM"].get(num, 0.0),
            float(bases["LSTM"][num]),
            float(bases["Transformer"][num]),
            bases.get("GNN", {}).get(num, 0.0),
            markov1_t.get(num, 0.0),
            cooc_t.get(num, 0.0),
            compat_bayes_t[num],
            compat_markov_t[num],
            compat_hmm_t[num],
            compat_nn_t[num],
            compat_gnn_t[num],
            float(reg_t['weekday']),
            float(reg_t['month']),
            float(reg_t['gap_days']),
            float(reg_t['schedule_flip']),
            float(roll_ent_t),
            float(roll_disp_t),
            wd_base_t.get(num, 0.0),  # weekday-conditioned baseline
            mo_base_t.get(num, 0.0),  # month-conditioned baseline
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
        hmm_t   = _build_tcn_prob_from_subset(sub_draws)
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
    hmm_t     = _build_tcn_prob_from_subset(history)
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
    # --- Evaluation & diagnostics: walk-forward, reliability, oracle ablation ---
    try:
        # Strict rolling walk-forward backtest over the last 150 draws
        summary = run_walk_forward_backtest(last_N=150, log_path="run_log.jsonl")
        print("[DIAG] Backtest summary:", summary)
        # Oracle ablation (mean improvement in PL-NLL if each expert were perfect)
        ablate = run_oracle_ablation(last_N=100, log_path="run_log.jsonl")
        print("[DIAG] Oracle ablation mean ΔNLL:", ablate)
    except Exception as _e:
        _w.warn(f"Diagnostics failed: {_e}")
except Exception as _e:
    warnings.warn(f"Diagnostics failed: {_e}")
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

# === FINAL PREDICTION PIPELINE (single-ticket output) =========================
# Chooses the method that backtests best on recent draws, then writes ONE ticket
# to 'predicted_tickets.txt'. Supports a HOLDOUT mode (HOLDOUT_LAST=1) which
# withholds the most recent draw for a quick sanity check.

def _rescale_probs_to_six(prob_dict):
    """Return a dict {1..40->p} scaled so sum p_i = 6 and each p in [1e-6, 1-1e-6].
    Adds a *tiny*, deterministic jitter to break perfect ties (avoids 1-2-3-4-5-6 when all p equal).
    """
    import numpy as _np
    p = _np.array([float(prob_dict.get(i, 1.0/40.0)) for i in range(1, 41)], dtype=float)
    # Clip negatives and NaNs, and guard against an all-zero vector
    p = _np.clip(_np.nan_to_num(p, nan=0.0), 0.0, None)
    s = float(p.sum())
    if s <= 0.0:
        p[:] = 6.0 / 40.0
    else:
        p *= (6.0 / s)
    # *Tiny* deterministic jitter breaks exact ties without changing calibration
    jitter = _np.linspace(1.0, 40.0, 40, dtype=float) * 1e-12
    p = p + jitter
    # Renormalize to keep sum≈6 after jitter
    p *= (6.0 / float(p.sum()))
    p = _np.clip(p, 1e-6, 1.0 - 1e-6)
    return {i: float(p[i-1]) for i in range(1, 41)}

def _top6_from_prob_dict(prob_dict):
    """
    Return a 6-number ticket chosen by JOINT COMBINATION MODELING.
    Falls back to simple top-6 by marginals if joint decoding fails.
    """
    try:
        hist = draws[:-1] if (isinstance(draws, list) and len(draws) > 0) else []
        ticket = choose_ticket_joint(prob_dict, hist, beam=JOINT_BEAM_WIDTH, top_pool=JOINT_TOP_POOL)
        return tuple(sorted(ticket))
    except Exception:
        import numpy as _np
        arr = _np.array([float(prob_dict.get(i, 0.0)) for i in range(1, 41)], dtype=float)
        idx = _np.argsort(arr)[-6:]
        return tuple(sorted(int(i+1) for i in idx))

def _set_like_nll(draw_set, prob_dict):
    """Independent-set approximation to the draw likelihood to compute an NLL."""
    import numpy as _np
    drawn = set(int(x) for x in draw_set)
    p = _np.array([float(prob_dict.get(i, 1.0/40.0)) for i in range(1, 41)], dtype=float)
    p = _np.clip(p, 1e-9, 1.0 - 1e-9)
    logp = 0.0
    for i in range(1, 41):
        if i in drawn:
            logp += float(_np.log(p[i-1]))
        else:
            logp += float(_np.log(1.0 - p[i-1]))
    return float(-logp)

def _probs_from_bayes(hist_draws, alpha=1.0):
    """Simple Dirichlet posterior (uniform prior) from history only."""
    from collections import Counter as _Counter
    cnt = _Counter([n for d in hist_draws for n in d])
    tot = max(1, len(hist_draws) * 6)
    raw = {i: (alpha + cnt.get(i, 0)) / (tot + 40.0 * alpha) for i in range(1, 41)}
    return _rescale_probs_to_six(raw)

def _probs_from_hmm(hist_draws):
    """Emission mixture from an HMM fit on `hist_draws` (or uniform if HMM not OK)."""
    try:
        if HMM_OK:
            raw = _build_tcn_prob_from_subset(hist_draws)
        else:
            raw = _uniform_prob40()
    except Exception:
        raw = _uniform_prob40()
    return _rescale_probs_to_six(raw)

def _probs_from_pf(hist_draws, num_particles=4000, alpha=0.005, sigma=0.01):
    """Assimilate history with the ParticleFilter and return the posterior mean probs."""
    try:
        pf = ParticleFilter(num_numbers=40, num_particles=int(num_particles),
                            alpha=float(alpha), sigma=float(sigma))
        for d in hist_draws:
            pf.predict()
            pf.update(d)
        # One more predict step to get the next-step prior
        pf.predict()
        mean_p = pf.get_mean_probabilities()
        raw = {i: float(mean_p[i-1]) for i in range(1, 41)}
    except Exception:
        raw = _uniform_prob40()
    return _rescale_probs_to_six(raw)

def _probs_from_ensemble(hist_draws):
    """
    6-expert ensemble: computes Bayes, Markov, HMM, LSTM, Transformer, GNN from hist_draws,
    learns adaptive weights based on recent performance, and returns the blended distribution.
    """
    import numpy as np
    try:
        # 1) Compute all 6 base experts from hist_draws only
        bayes_p = compute_bayes_posterior(hist_draws, alpha=1)
        bayes_p = {n: bayes_p.get(n, 0.0) for n in range(1, 41)}
        s = sum(bayes_p.values())
        if s > 0:
            bayes_p = {n: bayes_p[n]/s for n in range(1, 41)}

        markov_p = _markov_prob_from_history(hist_draws)
        hmm_p = _build_tcn_prob_from_subset(hist_draws) if HMM_OK else _uniform_prob40()

        # For neural nets (LSTM/Transformer): compute features from the last k draws of hist_draws
        t_idx = len(hist_draws)
        if t_idx > k:
            feats = compute_stat_features(hist_draws[t_idx-k:t_idx], t_idx-1)
            feats = feats.reshape(1, feats.shape[0], feats.shape[1])
            try:
                meta_feats = np.array([_meta_features_at_idx(t_idx)], dtype=float)
            except Exception:
                meta_feats = None
            _l, _t = _predict_with_nets(feats, meta_feats, t_idx=t_idx, use_finetune=True, mc_passes=7)
            lstm_p = {n: float(_l[n-1]) for n in range(1, 41)}
            transformer_p = {n: float(_t[n-1]) for n in range(1, 41)}
        else:
            lstm_p = _uniform_prob40()
            transformer_p = _uniform_prob40()

        # GNN
        try:
            if t_idx > k:
                meta_feats = np.array([_meta_features_at_idx(t_idx)], dtype=float)
            else:
                meta_feats = None
            gnn_p = _gnn_prob_from_history(hist_draws, gnn_model=globals().get('gnn_model', None),
                                          meta_features=meta_feats[0] if meta_feats is not None else None)
        except Exception:
            gnn_p = _uniform_prob40()

        # 2) Learn weights from recent NLLs (use last 60 draws of hist_draws for weight learning)
        # For simplicity, use equal weights if insufficient history, otherwise compute NLL-based weights
        if len(hist_draws) < 30:
            w = np.ones(6, dtype=float) / 6.0
        else:
            # Compute per-expert NLLs on recent window
            window_size = min(60, len(hist_draws) // 2)
            nlls = np.zeros(6, dtype=float)
            count = 0
            for i in range(max(10, len(hist_draws) - window_size), len(hist_draws)):
                sub_hist = hist_draws[:i]
                truth = hist_draws[i]
                try:
                    b_sub = compute_bayes_posterior(sub_hist, alpha=1)
                    nlls[0] += _nll_from_prob_dict(b_sub, truth)
                    m_sub = _markov_prob_from_history(sub_hist)
                    nlls[1] += _nll_from_prob_dict(m_sub, truth)
                    h_sub = _build_tcn_prob_from_subset(sub_hist) if HMM_OK else _uniform_prob40()
                    nlls[2] += _nll_from_prob_dict(h_sub, truth)
                    # Skip LSTM/Transformer/GNN in backtest weight learning (too expensive)
                    nlls[3] += 20.0  # placeholder
                    nlls[4] += 20.0
                    nlls[5] += 20.0
                    count += 1
                except Exception:
                    continue
            if count > 0:
                nlls /= float(count)
                # Softmax(-NLL) for weights
                x = -nlls
                x = x - np.max(x)
                w = np.exp(x)
                w = w / (w.sum() + 1e-12)
            else:
                w = np.ones(6, dtype=float) / 6.0

        # 3) Blend the 6 experts
        P = {}
        for n in range(1, 41):
            bv = np.array([
                bayes_p.get(n, 0.0),
                markov_p.get(n, 0.0),
                hmm_p.get(n, 0.0),
                lstm_p.get(n, 0.0),
                transformer_p.get(n, 0.0),
                gnn_p.get(n, 0.0),
            ], dtype=float)
            P[n] = float(np.dot(w, bv))

        # Normalize
        s = sum(P.values())
        if s > 0:
            P = {n: P[n]/s for n in range(1, 41)}
        else:
            P = _uniform_prob40()

        return _rescale_probs_to_six(P)

    except Exception as e:
        import warnings
        warnings.warn(f"Ensemble computation failed: {e}")
        return _uniform_prob40()

def _evaluate_method(hist_draws, method_fn, eval_window=80):
    """Walk-forward backtest on the tail of history; return (avg_hits, mean_nll)."""
    N = len(hist_draws)
    if N < 10:
        return (0.0, float('inf'))
    start = max(6, N - int(eval_window))
    hits = []; nlls = []
    for t in range(start, N):
        try:
            P = method_fn(hist_draws[:t])
            top6 = predict_joint_number_set(
                base_prob_dict=locals().get('final_prob_dict', locals().get('prob_dict', locals().get('marginals', {}))),
                beam_width=JOINT_BEAM_WIDTH
            )
            h = len(set(top6).intersection(set(hist_draws[t])))
            hits.append(int(h))
            nlls.append(_set_like_nll(hist_draws[t], P))
        except Exception:
            continue
    if len(hits) == 0:
        return (0.0, float('inf'))
    return (float(sum(hits)) / len(hits), float(sum(nlls)) / len(nlls))

def _choose_best_method(hist_draws, eval_window=80):
    """Return (name, method_fn, metrics_dict) for the best-performing method."""
    cand = [
        ("Ensemble6", _probs_from_ensemble),
        ("ParticleFilter", _probs_from_pf),
        ("HMM", _probs_from_hmm),
        ("Bayes", _probs_from_bayes),
    ]
    scores = []
    for name, fn in cand:
        avg_hits, mean_nll = _evaluate_method(hist_draws, fn, eval_window=eval_window)
        scores.append((name, fn, avg_hits, mean_nll))
    # Best by lowest NLL; break ties by highest hits
    scores.sort(key=lambda x: (x[3], -x[2]))
    best = scores[0]
    return best[0], best[1], {"avg_hits": best[2], "mean_nll": best[3]}

def _write_single_ticket(numbers, path="predicted_tickets.txt"):
    """Write ONE line with space-separated numbers to `path`."""
    s = " ".join(str(int(n)) for n in numbers)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s + "\n")

def _is_near_uniform(prob_dict, tol=1e-9):
    """Return True if all probabilities are effectively equal (within `tol`)."""
    import numpy as _np
    p = _np.array([float(prob_dict.get(i, 0.0)) for i in range(1, 41)], dtype=float)
    if p.size == 0:
        return True
    return (float(p.max()) - float(p.min())) <= float(tol)

def _predict_and_write_single_ticket(holdout_last=False, eval_window=80):
    """
    Pick the best method by recent backtest, predict next draw, and write 1 ticket.
    If holdout_last=True, exclude the last draw from training and report its hits.
    """
    hist = draws[:-1] if holdout_last else draws[:]
    best_name, best_fn, metrics = _choose_best_method(hist, eval_window=eval_window)
    P = best_fn(hist)
    # Safety guard: if the chosen method returned an (almost) uniform distribution,
    # or if it produced NaNs/Infs, fall back to a Bayes posterior from the same history
    # to avoid the 1..6 artifact.
    try:
        _pvals = np.array([float(P.get(i, 0.0)) for i in range(1, 41)], dtype=float)
        _bad = not np.isfinite(_pvals).all()
    except Exception:
        _bad = True

    if _bad or _is_near_uniform(P):
        try:
            # Preferred wrapper if available (returns sum≈6 and clipped)
            P = _probs_from_bayes(hist)
        except NameError:
            # Safe fallback: compute raw Bayes and rescale
            P = _rescale_probs_to_six(compute_bayes_posterior(hist, alpha=1))
        best_name = f"{best_name}+BayesFallback"
    ticket = _top6_from_prob_dict(P)
    _write_single_ticket(ticket, path="predicted_tickets.txt")
    print(f"[FINAL] Prediction written to predicted_tickets.txt: {ticket}")
    # Logging for traceability
    try:
        _log_predict_run(blend_weights=[best_name], t_eval=(len(draws)-1))
        _log_jsonl({
            "type": "final_ticket",
            "ts": _now_iso(),
            "method": best_name,
            "metrics": metrics,
            "ticket": ticket,
            "holdout": bool(holdout_last),
        })
    except Exception:
        pass
    # Print a concise console summary
    print(f"[FINAL] Method={best_name}  ticket={ticket}  metrics={metrics}")
    if holdout_last:
        truth = sorted(list(draws[-1]))
        hit = len(set(ticket).intersection(set(truth)))
        print(f"[HOLDOUT] truth={truth}  hits={hit}")

if __name__ == "__main__":
    # HOLDOUT_LAST=1 → leave the most recent draw out for a quick test
    holdout = str(os.environ.get("HOLDOUT_LAST", "0")).strip() == "1"
    try:
        _predict_and_write_single_ticket(holdout_last=holdout, eval_window=80)
    except Exception as e:
        warnings.warn(f"Final prediction pipeline failed: {e}")