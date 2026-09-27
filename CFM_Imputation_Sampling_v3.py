"""
CFM_Imputation_Sampling_v3.py
=============================
Generates model posterior samples for the IMPUTATION question:

    for every model feature ("target"), and for every conditioning pattern
    that does NOT contain that target, draw N_DRAWS model samples for
    N_PER_SUBSET real Gaia sources, in two flavours:

      have    : sources where the target IS measured.  The target is
                withheld from the model's INPUT but is present in its
                OUTPUT mask, so the draws are a genuine posterior for a
                quantity we can check against truth.  Fully
                in-distribution: mask_generator draws the input and
                output masks independently, so "input 0 / output 1" is a
                configuration the model saw constantly during training.

      missing : sources where the target is NOT measured.  The output
                mask bit for the target is FORCED ON.  This is the
                extrapolation the experiment is about: the column-wise
                configuration is one the model saw in training, but it
                never saw it *for a source where that column was absent*.
                Read the caveats in the write-up before using these.

Outputs, per (target, flavour, pattern), two pickled-dict .npy files:

    {TAG}_pattern{digits}_pred{code}_{flavour}_samples_{stamp}.npy
        'samples'          (n, N_DRAWS, dim_phys) float32, PHYSICAL units
        'output_mask_phys' (n, dim_phys) bool
        'input_mask_phys'  (n, dim_phys) bool
        'source_id'        (n,) int64
        + metadata

    {TAG}_pattern{digits}_pred{code}_{flavour}_truth_{stamp}.npy
        'truth'            (n, dim_phys) float32, PHYSICAL units,
                           NaN wherever the value is not measured
        'valid'            (n, dim_phys) bool
        'source_id'        (n,) int64
        + metadata

Both files are ROW-ALIGNED and share 'source_id', so they can be checked
against each other.

Naming.  Features are numbered by their position in the training feature
list (l=0, b=1, parallax=2, pml=3, pmb=4, G=5, BP=6, RP=7, RV=8).  The
pattern string is the sorted conditioned-feature codes concatenated;
'NONE' means the unconditional pattern.  So

    pattern01_pred2      : given l and b, predict parallax
    pattern0156_pred8    : given l, b, G, BP, predict radial_velocity
    patternNONE_pred2    : unconditional baseline for parallax

NOTHING IS ZEROED.  Columns whose output-mask bit is 0 hold whatever the
integrator produced for a channel the loss never trained; they are saved
verbatim and 'output_mask_phys' tells you which those are.

Data source.  This script streams the REAL validation split of the Gaia
HDF5 catalog.  It does NOT use a Pure_Sample: a cached Pure_Sample is a
~1% subsample in normalized model space, which is both too small for the
rare flavours and the wrong space.  The catalog is read directly here
(rather than through HDF5IterableDataset) for one reason: that class does
not emit source_id.  The validation-split definition is taken from
source_id_dataset_creator_070726.partition_code, which is documented to
use the same random_index boundaries as
HDF5IterableDataset._partition_selection(p_splits=100), and the mapping is
asserted at startup.

Pattern-exact subsets.  Every source in a subset satisfies its pattern
exactly: all conditioned features are measured.  There is therefore no
mask fallback anywhere in the output.  Because that intersection can be
small (RV is measured for ~2% of sources) each (target, flavour) keeps two
reservoirs -- a general one and one restricted to RV-measured sources --
and patterns that condition on RV draw from the latter.  Where fewer than
N_PER_SUBSET sources are available the file is still written, with
status='short' and n_used recorded.  Where the intersection is
structurally empty (e.g. target=parallax/missing are 2-parameter
solutions, which have no proper motion, so no pattern conditioning on
pml/pmb can be satisfied) status='empty' and no file is written.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

# ===========================================================================
# Configuration
# ===========================================================================
SETUP_SCRIPT = 'YOUR PATH/CFM_Sampling_hdf5_070726_patched_v2.py'
CFM_LIB_DIR = 'YOUR PATH'
OUT_ROOT = Path('YOUR PATH/imputation_samples')



def _env_int(name, default):
    v = os.environ.get(name)
    return default if v in (None, '') else int(v)

N_PER_SUBSET  = _env_int('CFM_N_PER_SUBSET', 1000)
N_DRAWS       = _env_int('CFM_N_DRAWS', 100)
MAX_SCAN_ROWS = _env_int('CFM_MAX_SCAN_ROWS', 0) or None   # 0/unset -> full split
MODEL_TAG     = os.environ.get('CFM_MODEL_TAG', 'POS_PM_setA_0710_RK50')

# Short, human-chosen tag for the output directory and filenames. The full
# MODEL_NAME (~200 chars) goes in the manifest, not in every filename.
#MODEL_TAG = 'POS_PM_setA_0710_RK50'



#N_PER_SUBSET = 1000      # sources per (target, flavour, pattern)
#N_DRAWS = 100            # model samples per source
POOL_SIZE = 25000        # reservoir capacity per (target, flavour, pool)
RK4_STEPS = 50
BATCH_SOURCES = 250      # sources per forward pass (x N_DRAWS rows)

FLAVOURS = ('have', 'missing')

# Conditioning patterns. The target is never in its own pattern, so each
# pattern is used for every target it does not contain.
BESPOKE_PERMUTATIONS = [
    ['l', 'b'],
    ['l', 'b', 'parallax'],
    ['l', 'b', 'pml', 'pmb'],
    ['l', 'b', 'parallax', 'pml', 'pmb'],
    ['l', 'b', 'phot_g_mean_mag'],
    ['l', 'b', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
    ['l', 'b', 'parallax', 'pml', 'pmb',
     'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
    ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
    ['parallax', 'pml', 'pmb'],
    ['l', 'b', 'radial_velocity'],
    ['l', 'b', 'parallax', 'pml', 'pmb', 'radial_velocity'],
]
ADD_UNCONDITIONAL = True   # patternNONE: zero-information baseline
ADD_DROP_ONE = True        # every feature except one
ADD_KEEP_ONE = True        # exactly one feature

MAX_SCAN_ROWS = None       # None = full validation split
SEED = 20260812
VERBOSE_EVERY = 100        # progress print every N catalog files

# ===========================================================================
# Feature numbering for filenames. Fixed by design: the feature set is
# frozen for the remainder of training, and a silent reordering would
# silently rename every output file. Asserted against the checkpoint below.
# ===========================================================================
EXPECTED_FEATURES = ['l', 'b', 'parallax', 'pml', 'pmb', 'phot_g_mean_mag',
                     'phot_bp_mean_mag', 'phot_rp_mean_mag', 'radial_velocity']
FEATURE_CODE = {name: i for i, name in enumerate(EXPECTED_FEATURES)}

# random_index partition code (source_id_dataset_creator_070726) for the
# validation split of each training set.
SPLIT = 'test'                                    # 'val' | 'test'
VAL_CODE_FOR_SET  = {'A': 1, 'B': 4, 'M': 6}
TEST_CODE_FOR_SET = {'A': 2, 'B': 5, 'M': 7}
SPLIT_CODE_FOR_SET = {'val': VAL_CODE_FOR_SET, 'test': TEST_CODE_FOR_SET}

sys.path.insert(0, CFM_LIB_DIR)
from source_id_dataset_creator_070726 import (   # noqa: E402
    partition_code, pm_gal_transform, PARTITION_NAMES,
)


def pattern_digits(cond):
    """Sorted feature codes, concatenated. 'NONE' for the empty pattern."""
    if len(cond) == 0:
        return 'NONE'
    return ''.join(str(FEATURE_CODE[f]) for f in sorted(cond, key=FEATURE_CODE.get))


def pattern_label(cond, feature_names):
    if len(cond) == 0:
        return 'unconditional'
    if len(cond) == len(feature_names) - 1:
        missing = [f for f in feature_names if f not in cond]
        return f'drop {missing[0]}'
    if len(cond) == 1:
        return f'only {cond[0]}'
    return ', '.join(cond)


def build_pattern_list(feature_names):
    """Deduplicated on frozenset; order within a pattern is irrelevant."""
    out, seen = [], set()

    def push(p):
        key = frozenset(p)
        if key in seen:
            return
        seen.add(key)
        out.append(list(p))

    if ADD_UNCONDITIONAL:
        push([])
    for p in BESPOKE_PERMUTATIONS:
        bad = [f for f in p if f not in feature_names]
        if bad:
            print(f'  WARNING: pattern {p} contains unknown features {bad}; skipped.')
            continue
        push(p)
    if ADD_DROP_ONE:
        for f in feature_names:
            push([g for g in feature_names if g != f])
    if ADD_KEEP_ONE:
        for f in feature_names:
            push([f])
    return out


# ===========================================================================
# Import the sampling setup (model, normalizer, denormalizer, mask helpers)
# ===========================================================================
def load_setup(path):
    """
    Execute CFM_Sampling_hdf5_*.py up to the sampling loop, so the model,
    the normalizer/denormalizer closures and the mask helpers are built by
    exactly the code that produced the training-time samples.

    Three source-level overrides are applied before exec:
      * PURE_SAMPLE_OVERRIDE -> '' (we never want a cached subsample)
      * SAVE_SAMPLES         -> False (the prefix must not write anything)
      * the create_test_loader() call is neutralised, so importing the
        setup neither torch.loads a Pure_Sample nor triggers a full
        catalog scan of its own.
    """
    path = Path(path)
    src = path.read_text()

    marker = 'sample_metrics = {}'
    if src.count('\n' + marker) != 1:
        raise RuntimeError(f'setup marker {marker!r} is not unique in {path}')
    prefix = src.split('\n' + marker)[0]

    patched = []
    for line in prefix.split('\n'):
        if line.startswith('PURE_SAMPLE_OVERRIDE'):
            patched.append("PURE_SAMPLE_OVERRIDE = ''")
        elif line.startswith('SAVE_SAMPLES'):
            patched.append('SAVE_SAMPLES = False')
        elif line.startswith('SAVE_TEST_SET'):
            patched.append('SAVE_TEST_SET = False')
        elif line.startswith('x_test, test_loader'):
            patched.append('x_test, test_loader = None, None'
                           '  # neutralised by CFM_Imputation_Sampling_v3')
        else:
            patched.append(line)
    prefix = '\n'.join(patched)

    ns = {'__name__': '__cfm_setup__', '__file__': str(path)}
    cwd = os.getcwd()
    sys.path.insert(0, str(path.parent))
    try:
        os.chdir(path.parent)
        exec(compile(prefix, str(path), 'exec'), ns)
    finally:
        os.chdir(cwd)

    need = ['model', 'feature_names', 'feature_dict', 'dim', 'dim_phys',
            'device', 'normalizer', 'denormalizer', 'expand_mask',
            'phys_names_to_model_indices', 'sample_properties_distribution_rk4',
            'MODEL_PARAM_DICT', 'MODEL_NAME', 'hdf5_filelist', 'L_CUSTOM']
    missing = [k for k in need if k not in ns]
    if missing:
        raise RuntimeError(f'setup did not define: {missing}')
    return ns


# ===========================================================================
# Reservoir sampling (Algorithm R), vectorised.
#
# The scalar form -- for each eligible row, r = rng.integers(0, seen); if
# r < capacity, replace slot r -- is correct but is a Python loop, which is
# fatal over a ~70M-row split. This does the same draws in batch. Duplicate
# target slots within one batch are resolved last-write-wins, which is what
# the sequential algorithm would do.
# ===========================================================================
class Reservoir:

    def __init__(self, capacity, n_feat, n_astro, rng):
        self.cap = int(capacity)
        self.feat = np.empty((self.cap, n_feat), dtype=np.float64)
        self.astro = np.empty((self.cap, n_astro), dtype=np.float64)
        self.valid = np.empty((self.cap, n_feat), dtype=bool)
        self.sid = np.empty(self.cap, dtype=np.int64)
        self.n = 0        # slots filled
        self.seen = 0     # eligible rows encountered
        self.rng = rng

    def add(self, feat, astro, valid, sid):
        m = feat.shape[0]
        if m == 0:
            return

        # Phase 1: fill empty slots in order.
        take = min(self.cap - self.n, m)
        if take > 0:
            s = slice(self.n, self.n + take)
            self.feat[s] = feat[:take]
            self.astro[s] = astro[:take]
            self.valid[s] = valid[:take]
            self.sid[s] = sid[:take]
            self.n += take
            self.seen += take

        rest = m - take
        if rest == 0:
            return

        # Phase 2: for the i-th remaining row the overall 1-indexed
        # position is seen + i, so accept with probability cap/(seen+i)
        # and replace a uniformly chosen slot.
        k = self.seen + np.arange(1, rest + 1, dtype=np.int64)
        r = self.rng.integers(0, k)
        acc = r < self.cap
        self.seen += rest
        if not acc.any():
            return

        slots = r[acc]
        rows = np.nonzero(acc)[0] + take

        # Keep only the LAST write to each slot (sequential semantics).
        _, first_in_reversed = np.unique(slots[::-1], return_index=True)
        keep = slots.size - 1 - first_in_reversed
        slots, rows = slots[keep], rows[keep]

        self.feat[slots] = feat[rows]
        self.astro[slots] = astro[rows]
        self.valid[slots] = valid[rows]
        self.sid[slots] = sid[rows]

    def finalize(self):
        return dict(feat=self.feat[:self.n].copy(),
                    astro=self.astro[:self.n].copy(),
                    valid=self.valid[:self.n].copy(),
                    sid=self.sid[:self.n].copy(),
                    n=self.n, seen=self.seen)


# ===========================================================================
# One streaming pass over the validation split
# ===========================================================================
def stream_pools(hdf5_filelist, feature_names, set_type, targets, rng, split=SPLIT):
    """
    Build, per (target, flavour), two reservoirs:
        'gen' : any source of that flavour
        'rv'  : that flavour AND radial_velocity measured
    The 'rv' pool exists because RV is measured for ~2% of sources, so a
    general pool cannot supply N_PER_SUBSET sources for a pattern that
    conditions on RV. It is not built for target='radial_velocity', where
    no pattern can contain RV.

    pml/pmb are NOT computed here. Only the ~POOL_SIZE retained rows ever
    need them, so ra/dec/pmra/pmdec are carried through the reservoir and
    the astropy transform runs once at the end. On a 70M-row split that
    removes the dominant CPU cost of the scan.
    """
    codes = SPLIT_CODE_FOR_SET[split]
    if set_type not in codes:
        raise RuntimeError(f'unknown SET_TYPE {set_type!r}')
    sel_code = codes[set_type]
    print(f"split = partition code {sel_code} "
          f"({PARTITION_NAMES[sel_code]}) for set '{set_type}'")

    expected = ('validation' if split == 'val' else 'test') + f' {set_type}'
    if PARTITION_NAMES[sel_code] != expected:
        raise RuntimeError(
            f'partition code {sel_code} is {PARTITION_NAMES[sel_code]!r}, '
            f'expected {expected!r}')
            
    has_pm = ('pml' in feature_names) or ('pmb' in feature_names)
    astro_cols = ['ra', 'dec', 'pmra', 'pmdec'] if has_pm else []
    disk_cols = [f for f in feature_names if f not in ('pml', 'pmb')]
    read_cols = list(dict.fromkeys(disk_cols + astro_cols))

    j_rv = feature_names.index('radial_velocity') if 'radial_velocity' in feature_names else None
    idx_of = {f: i for i, f in enumerate(feature_names)}
    j_pml = idx_of.get('pml')
    j_pmb = idx_of.get('pmb')

    pools = {}
    for t in targets:
        for fl in FLAVOURS:
            pools[(t, fl, 'gen')] = Reservoir(POOL_SIZE, len(feature_names),
                                              len(astro_cols), rng)
            if j_rv is not None and t != 'radial_velocity':
                pools[(t, fl, 'rv')] = Reservoir(POOL_SIZE, len(feature_names),
                                                 len(astro_cols), rng)

    scanned = 0
    t0 = time.time()
    for fi, fpath in enumerate(hdf5_filelist):
        if MAX_SCAN_ROWS is not None and scanned >= MAX_SCAN_ROWS:
            print(f'  MAX_SCAN_ROWS reached after {fi} files.')
            break
        try:
            fh = h5py.File(fpath, 'r')
        except OSError as e:
            print(f'  WARNING: cannot open {fpath}: {e}')
            continue
        try:
            ri = fh['random_index'][:]
            sel = (partition_code(ri) == sel_code)
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue

            sid = np.asarray(fh['source_id'][:])[sel].astype(np.int64)

            raw = {}
            for c in read_cols:
                if c not in fh:
                    raise KeyError(f'column {c!r} missing from {fpath}')
                raw[c] = np.asarray(fh[c][:])[sel].astype(np.float64)
        finally:
            fh.close()

        feat = np.full((n_sel, len(feature_names)), np.nan, dtype=np.float64)
        valid = np.zeros((n_sel, len(feature_names)), dtype=bool)
        for c in disk_cols:
            j = idx_of[c]
            feat[:, j] = raw[c]
            valid[:, j] = np.isfinite(raw[c])

        if has_pm:
            astro = np.column_stack([raw[c] for c in astro_cols])
            # Matches HDF5IterableDataset: pml/pmb exist iff all four
            # astrometric inputs to the transform are finite.
            pm_ok = np.isfinite(astro).all(axis=1)
            if j_pml is not None:
                valid[:, j_pml] = pm_ok
            if j_pmb is not None:
                valid[:, j_pmb] = pm_ok
        else:
            astro = np.empty((n_sel, 0), dtype=np.float64)

        for t in targets:
            jt = idx_of[t]
            have = valid[:, jt]
            for fl in FLAVOURS:
                base = have if fl == 'have' else ~have
                if not base.any():
                    continue
                pools[(t, fl, 'gen')].add(feat[base], astro[base],
                                          valid[base], sid[base])
                key_rv = (t, fl, 'rv')
                if key_rv in pools:
                    sub = base & valid[:, j_rv]
                    if sub.any():
                        pools[key_rv].add(feat[sub], astro[sub],
                                          valid[sub], sid[sub])

        scanned += n_sel
        if VERBOSE_EVERY and (fi + 1) % VERBOSE_EVERY == 0:
            print(f'  {fi + 1} files, {scanned:,} validation sources, '
                  f'{time.time() - t0:.0f}s')

    print(f'  scan complete: {scanned:,} validation sources in '
          f'{time.time() - t0:.0f}s')

    out = {k: v.finalize() for k, v in pools.items()}

    # Deferred galactic proper-motion transform, on retained rows only.
    if has_pm:
        for key, p in out.items():
            if p['n'] == 0:
                continue
            ok = np.isfinite(p['astro']).all(axis=1)
            if not ok.any():
                continue
            pml, pmb = pm_gal_transform(p['astro'][ok, 0], p['astro'][ok, 1],
                                        p['astro'][ok, 2], p['astro'][ok, 3])
            if j_pml is not None:
                p['feat'][ok, j_pml] = pml
            if j_pmb is not None:
                p['feat'][ok, j_pmb] = pmb

    return out, scanned


# ===========================================================================
# Sampling for one (target, flavour, pattern)
# ===========================================================================
def sample_one(ns, feat, valid, cond, target, rng):
    """
    feat  (n, dim_phys) physical, NaN where not measured
    valid (n, dim_phys) bool
    Returns (samples_phys (n, N_DRAWS, dim_phys), in_mask_phys, out_mask_phys)
    """
    model = ns['model']
    device = ns['device']
    dim = ns['dim']
    dim_phys = ns['dim_phys']
    feature_names = ns['feature_names']

    n = feat.shape[0]

    # Normalizer/denormalizer operate on physical (n, dim_phys) input and
    # would propagate NaN into columns that are then multiplied by a zero
    # mask (0 * NaN = NaN), poisoning the network input. Zero the unmeasured
    # cells first; the mask is what carries their absence.
    safe = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    x_norm = ns['normalizer'](torch.as_tensor(safe, dtype=torch.float32))
    v_model = ns['expand_mask'](torch.as_tensor(valid.astype(np.float32)))

    if not torch.isfinite(x_norm).all():
        raise RuntimeError('non-finite value in normalized model input')

    # Input mask: exactly the pattern. Subsets are pattern-exact, so
    # cmask * v_model == cmask; assert rather than silently fall back.
    cmask = torch.zeros((n, dim), dtype=torch.float32)
    if len(cond) > 0:
        cidx = ns['phys_names_to_model_indices'](cond)
        cmask[:, cidx] = 1.0
    in_mask = cmask * v_model
    if not torch.equal(in_mask, cmask):
        raise RuntimeError('conditioned feature is unmeasured for some source; '
                           'pattern-exact selection failed')

    # Output mask: everything measured, plus the target forced on. For
    # 'have' the target bit is already 1 and this is a no-op; for 'missing'
    # this is the intended extrapolation.
    tgt_cols = ns['phys_names_to_model_indices']([target])
    out_mask = v_model.clone()
    out_mask[:, tgt_cols] = 1.0

    in_mask_d = in_mask.to(device)
    out_mask_d = out_mask.to(device)

    draws_model = np.empty((n, N_DRAWS, dim), dtype=np.float32)
    for s in range(0, n, BATCH_SOURCES):
        e = min(n, s + BATCH_SOURCES)
        # NOTE: sample_properties_distribution_rk4 returns a NumPy array,
        # not a tensor -- no .detach()/.cpu() here.
        draws_model[s:e] = ns['sample_properties_distribution_rk4'](
            model, x_norm[s:e].to(device), in_mask_d[s:e], out_mask_d[s:e],
            N_DRAWS, num_steps=RK4_STEPS)

    # Denormalize every draw. Doing it here (rather than downstream) is what
    # keeps anyone from taking a per-source median in normalized space,
    # where l is a (cos, sin) pair and a componentwise median is not the
    # median direction.
    samples_phys = np.empty((n, N_DRAWS, dim_phys), dtype=np.float32)
    for s in range(N_DRAWS):
        samples_phys[:, s, :] = ns['denormalizer'](draws_model[:, s, :])

    n_bad = int((~np.isfinite(samples_phys)).sum())
    if n_bad:
        print(f'      note: {n_bad} non-finite sample cells '
              f'({100.0 * n_bad / samples_phys.size:.3f}%) left as-is')

    # Report masks back in physical space.
    in_phys = np.zeros((n, dim_phys), dtype=bool)
    out_phys = valid.copy()
    jt = feature_names.index(target)
    out_phys[:, jt] = True
    for f in cond:
        in_phys[:, feature_names.index(f)] = True

    return samples_phys, in_phys, out_phys


# ===========================================================================
def main():
    rng = np.random.default_rng(SEED)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    print('[1/4] loading sampling setup')
    ns = load_setup(SETUP_SCRIPT)
    feature_names = ns['feature_names']
    dim_phys = ns['dim_phys']

    if list(feature_names) != EXPECTED_FEATURES:
        raise RuntimeError(
            'feature list does not match the frozen naming scheme.\n'
            f'  checkpoint: {list(feature_names)}\n'
            f'  expected  : {EXPECTED_FEATURES}\n'
            'Filenames encode feature POSITION, so update EXPECTED_FEATURES / '
            'FEATURE_CODE deliberately before rerunning.')

    set_type = ns['MODEL_PARAM_DICT']['SET_TYPE']
    print(f"  model: {ns['MODEL_NAME']}")
    print(f'  dim_phys={dim_phys} dim={ns["dim"]} L_CUSTOM={ns["L_CUSTOM"]} '
          f'SET_TYPE={set_type}')

    targets = list(feature_names)
    patterns = build_pattern_list(feature_names)
    print(f'  {len(patterns)} unique conditioning patterns, {len(targets)} targets')

    out_dir = OUT_ROOT / f'{MODEL_TAG}_{SPLIT}'
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_code = SPLIT_CODE_FOR_SET[SPLIT][set_type]
    print(f'[2/4] streaming the {PARTITION_NAMES[sel_code]} split')
    pools, scanned = stream_pools(np.atleast_1d(ns['hdf5_filelist']),
                                  feature_names, set_type, targets, rng)
    for t in targets:
        for fl in FLAVOURS:
            g = pools[(t, fl, 'gen')]
            msg = f'  {t:<20} {fl:<8} pool {g["n"]:>6} / seen {g["seen"]:,}'
            if (t, fl, 'rv') in pools:
                msg += f'   rv-pool {pools[(t, fl, "rv")]["n"]:>6}'
            print(msg)

    print('[3/4] sampling')
    manifest = {
        'timestamp': stamp, 'model_name': ns['MODEL_NAME'],
        'model_tag': MODEL_TAG,
        'feature_names': list(feature_names), 'feature_code': FEATURE_CODE,
        'n_per_subset': N_PER_SUBSET, 'n_draws': N_DRAWS,
        'rk4_steps': RK4_STEPS, 'pool_size': POOL_SIZE, 'seed': SEED,
        'set_type': set_type, 'split': SPLIT,
        'partition_code': sel_code,
        'scanned_rows': int(scanned),
        'sample_space': 'physical', 'truth_space': 'physical',
        'entries': [],
    }
    n_written = n_short = n_empty = 0

    for target in targets:
        jt = feature_names.index(target)
        for flavour in FLAVOURS:
            for cond in patterns:
                if target in cond:
                    continue
                use_rv = 'radial_velocity' in cond
                pool = pools.get((target, flavour, 'rv' if use_rv else 'gen'))
                pat = pattern_digits(cond)
                code = FEATURE_CODE[target]
                base = f'{MODEL_TAG}_pattern{pat}_pred{code}_{flavour}'

                entry = {
                    'target': target, 'target_code': code, 'flavour': flavour,
                    'pattern_digits': pat, 'cond_features': list(cond),
                    'pattern_label': pattern_label(cond, feature_names),
                    'pool': 'rv' if use_rv else 'gen',
                }

                if pool is None or pool['n'] == 0:
                    entry.update(status='empty', n_pool=0, n_eligible=0, n_used=0)
                    manifest['entries'].append(entry)
                    n_empty += 1
                    continue

                cidx = [feature_names.index(f) for f in cond]
                elig = (pool['valid'][:, cidx].all(axis=1) if cidx
                        else np.ones(pool['n'], dtype=bool))
                # Flavour is guaranteed by pool construction; re-check cheaply.
                elig &= pool['valid'][:, jt] if flavour == 'have' else ~pool['valid'][:, jt]
                n_elig = int(elig.sum())

                entry.update(n_pool=int(pool['n']), n_pool_seen=int(pool['seen']),
                             n_eligible=n_elig)

                if n_elig == 0:
                    entry.update(status='empty', n_used=0)
                    manifest['entries'].append(entry)
                    n_empty += 1
                    print(f'  {base}: EMPTY (no source satisfies the pattern)')
                    continue

                idx = np.nonzero(elig)[0]
                if idx.size > N_PER_SUBSET:
                    idx = np.sort(rng.choice(idx, size=N_PER_SUBSET, replace=False))
                    status = 'ok'
                else:
                    status = 'short'
                n_used = idx.size

                feat = pool['feat'][idx]
                valid = pool['valid'][idx]
                sid = pool['sid'][idx]

                samples, in_phys, out_phys = sample_one(
                    ns, feat, valid, cond, target, rng)

                truth = feat.astype(np.float32)
                truth[~valid] = np.nan

                meta = dict(entry)
                meta.update(status=status, n_used=int(n_used),
                            model_name=ns['MODEL_NAME'], model_tag=MODEL_TAG,
                            timestamp=stamp, feature_names=list(feature_names),
                            n_draws=N_DRAWS, rk4_steps=RK4_STEPS,
                            set_type=set_type)

                s_path = out_dir / f'{base}_samples_{stamp}.npy'
                t_path = out_dir / f'{base}_truth_{stamp}.npy'
                np.save(s_path, dict(samples=samples,
                                     output_mask_phys=out_phys,
                                     input_mask_phys=in_phys,
                                     source_id=sid, **meta),
                        allow_pickle=True)
                np.save(t_path, dict(truth=truth, valid=valid,
                                     source_id=sid, **meta),
                        allow_pickle=True)

                entry.update(status=status, n_used=int(n_used),
                             samples_file=s_path.name, truth_file=t_path.name)
                manifest['entries'].append(entry)
                n_written += 1
                if status == 'short':
                    n_short += 1
                flag = '  [SHORT]' if status == 'short' else ''
                print(f'  {base}: n={n_used} of {n_elig} eligible{flag}')

    print('[4/4] writing manifest')
    mpath = out_dir / f'imputation_manifest_{MODEL_TAG}_{stamp}.json'
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nwrote {n_written} sample/truth pairs to {out_dir}')
    print(f'  {n_short} short (fewer than {N_PER_SUBSET} sources), '
          f'{n_empty} structurally empty')
    print(f'  manifest: {mpath}')
    if n_written == 0:
        raise SystemExit('no files written')


if __name__ == '__main__':
    main()
