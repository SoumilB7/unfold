"""F13 attempt isolation (EXECUTION_PLAN P2a.3; dossier model-benchmark/_reruns/p2a3_f13). INTENT §1b rule 3: a path
that raises credits nothing. Every attempt records into its OWN recorder; when it raises, that recording is discarded
(never unioned, never drawn, never compared). Soumil 2026-10-13: the state is put back after EVERY attempt (failed or
completed, F13-2/3), so each attempt and each independent pass starts from the state being tested, and each attempt runs
from its own recorded seed (F13-4). Whatever cannot be put back is reported (state_not_restored -> a G2 unknown). Each
attempt is written to a ledger entry (entry point, inputs, flags, seed, ok / error, what could not be restored)."""
import random, itertools, contextlib, signal
import torch
import numpy as np
import dag
from dag import DagRecorder, math_attention, clear_library_caches, is_reg

_SLOTS = ("_parameters", "_buffers", "_modules")
_CONFIG_TYPES = ("PreTrainedConfig", "PretrainedConfig", "GenerationConfig")   # library config objects a forward may change in place


def _shapes(kw):
    return {k: (list(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__) for k, v in kw.items()}


def _is_config(v):
    return any(c.__name__ in _CONFIG_TYPES for c in type(v).__mro__)


def _fp(v, tracked, depth=0):
    """A structural fingerprint of a Python value: what an attempt could have changed inside a container or object
    attribute (keys, lengths, scalars, which tensors / objects are held). A tensor the snapshot restores by value is
    identified by object and shape; any other tensor also by its version counter (a write to it cannot be put back).
    Large containers are summarised by their length and identity."""
    if isinstance(v, torch.Tensor):
        return ("T", id(v), tuple(v.shape)) if id(v) in tracked else ("T", id(v), tuple(v.shape), v._version)
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return v
    if isinstance(v, torch.nn.Module) or depth >= 3:
        return ("O", type(v).__name__, id(v))
    if isinstance(v, dict):
        if len(v) > 2000:
            return ("D", id(v), len(v))
        return ("D", tuple((repr(k), _fp(x, tracked, depth + 1)) for k, x in sorted(v.items(), key=lambda kv: repr(kv[0]))))
    if isinstance(v, (list, tuple)):
        if len(v) > 2000:
            return ("L", id(v), len(v))
        return ("L", type(v).__name__, tuple(_fp(x, tracked, depth + 1) for x in v))
    if isinstance(v, (set, frozenset)):
        return ("S", len(v), tuple(sorted(repr(x) for x in v)[:2000]))
    return ("O", type(v).__name__, id(v))


class Snapshot:
    """The state an attempt can change on the models it runs, and the extra objects given (a pipeline, a scheduler):
      - module attributes (rebinding, new / deleted attributes) and the parameter / buffer / sub-module slots;
      - every tensor held by a module (parameters, buffers, tensor attributes, tensors inside list / tuple / dict
        attributes): its storage (replaced storage is put back) and, unless it lives in shared zero storage (writes
        there are refused or copied-on-write by the recorder), its VALUE (writes through .data included);
      - containers held as attributes (dict / list / set: e.g. attention-bias caches) by their contents;
      - library config objects (PretrainedConfig, GenerationConfig, nested sub-configs) and the extra objects by
        their attributes; the torch / numpy / random RNG states.
    After restore() a structural fingerprint of every attribute is compared with the one taken here; any difference
    left (e.g. a write inside a nested container) is reported as not restored. Known blind spot: a plain helper object
    held as an attribute (not a library config, not a container) is fingerprinted by identity only, so a change of
    its own attributes is neither restored nor reported (likewise torch.Generator / numpy state held in objects)."""
    def __init__(self, *models, objects=()):
        self.mods, self.tens, self.conts, self.objs = [], {}, [], []
        seen = set()
        for model in models:
            for mn, mod in model.named_modules():
                d = vars(mod)
                attrs = {k: v for k, v in d.items() if "hook" not in k}
                self.mods.append((mn, mod, attrs, dict(mod._parameters), dict(mod._buffers), dict(mod._modules)))
                for k, t in list(mod._parameters.items()) + list(mod._buffers.items()):
                    self._tensor(f"{mn}.{k}" if mn else k, t)
                for k, v in attrs.items():
                    if k in _SLOTS:
                        continue
                    name = f"{mn}.{k}" if mn else k
                    self._value(name, v, seen)
        for i, o in enumerate(objects):
            if o is not None:
                self._object(f"<object {i}: {type(o).__name__}>", o, seen)
        self.rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        self.fp0 = self._fingerprint()

    def _tensor(self, name, t):
        if t is None or not isinstance(t, torch.Tensor) or id(t) in self.tens or t.is_meta:
            return
        val = None if is_reg(t) else t.detach().clone()
        self.tens[id(t)] = (name, t, t.data, t._version, val)

    def _value(self, name, v, seen):
        if isinstance(v, torch.Tensor):
            self._tensor(name, v)
        elif isinstance(v, (dict, list, set)) and id(v) not in seen:
            seen.add(id(v))
            self.conts.append((name, v, dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else set(v))))
            for x in (v.values() if isinstance(v, dict) else v):
                if isinstance(x, torch.Tensor):
                    self._tensor(f"{name}[]", x)
        elif isinstance(v, tuple):
            for x in v:
                if isinstance(x, torch.Tensor):
                    self._tensor(f"{name}[]", x)
        elif _is_config(v):
            self._object(name, v, seen)

    def _object(self, name, o, seen):
        if id(o) in seen or isinstance(o, torch.nn.Module) or not hasattr(o, "__dict__"):
            return
        seen.add(id(o))
        attrs = dict(vars(o))
        self.objs.append((name, o, attrs))
        for k, v in attrs.items():
            self._value(f"{name}.{k}", v, seen)

    def _fingerprint(self):
        out = {}
        for mn, mod, _, _, _, _ in self.mods:
            for k, v in vars(mod).items():
                if "hook" not in k and k not in _SLOTS:
                    out[f"{mn}.{k}"] = _fp(v, self.tens)
        for name, o, _ in self.objs:
            out[name] = _fp(vars(o), self.tens)
        return out

    def restore(self):
        lost = []
        for mn, mod, attrs, params, bufs, subs in self.mods:
            d = vars(mod)
            for k in [k for k in d if "hook" not in k and k not in attrs]:
                del d[k]
            d.update(attrs)
            for slot, saved in ((mod._parameters, params), (mod._buffers, bufs), (mod._modules, subs)):
                if slot.keys() != saved.keys() or any(slot[k] is not v for k, v in saved.items()):
                    slot.clear(); slot.update(saved)
        for name, o, attrs in self.objs:
            try:
                d = vars(o)
                for k in [k for k in d if k not in attrs]:
                    del d[k]
                d.update(attrs)
            except Exception:
                lost.append(name)
        for name, c, saved in self.conts:
            try:
                if isinstance(c, dict):
                    if len(c) != len(saved) or any(k not in c or c[k] is not v for k, v in saved.items()):
                        c.clear(); c.update(saved)
                elif isinstance(c, list):
                    if len(c) != len(saved) or any(a is not b for a, b in zip(c, saved)):
                        c[:] = saved
                elif c != saved:
                    c.clear(); c.update(saved)
            except Exception:                                    # (an immutable container cannot have changed)
                lost.append(name)
        for name, t, data0, ver0, val in self.tens.values():
            swapped = t.data.data_ptr() != data0.data_ptr() or t.shape != data0.shape or t.dtype != data0.dtype
            if swapped:
                t.data = data0                                   # storage replaced (copy-on-write, .data = ...): put back
            if val is not None:
                try:
                    if not torch.equal(t.data, val):             # value written (in place or through .data)
                        with torch.no_grad():
                            t.data.copy_(val)
                except Exception:
                    lost.append(name)
            elif not swapped and t._version != ver0:
                lost.append(name)                                # a write into shared zero storage itself got through
        torch.set_rng_state(self.rng[0]); np.random.set_state(self.rng[1]); random.setstate(self.rng[2])
        fp1 = self._fingerprint()
        lost += [k for k in sorted(set(self.fp0) | set(fp1)) if self.fp0.get(k) != fp1.get(k)]
        return lost


_SEEDS = itertools.count()


def _seed(seed):
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 32)); random.seed(seed)


_no_alarm = dag._alarm_blocked      # the restore and the ledger write never run a stage time limit midway (delivered after)


@contextlib.contextmanager
def isolated(models, ledger, desc, inputs=None, objects=()):
    """One attempt's state window (Soumil 2026-10-13, F13-2/3/4): snapshot, run from the attempt's own recorded seed,
    put the state back afterwards whether the attempt raised or completed. The recorder's run-level event lists
    (non-finite results, zero-storage writes, copy-on-write) keep only the events of attempts that completed.
    Yields the ledger entry."""
    models = models if isinstance(models, (list, tuple)) else [models]
    clear_library_caches()
    snap = Snapshot(*models, objects=objects)
    seed = next(_SEEDS)
    entry = dict(desc, seed=seed)
    if inputs is not None:
        entry["inputs"] = _shapes(inputs)
    events = [(lst, len(lst)) for lst in (dag.NONFINITE_EVENTS, dag.ZSW_EVENTS, dag.COW_EVENTS)]
    _seed(seed)
    try:
        yield entry
        entry["ok"] = True
    except BaseException as e:
        entry.update(ok=False, error=f"{type(e).__name__}: {str(e)[:160]}")
        raise
    finally:
        def _drop_events():
            for lst, n in events:
                del lst[n:]                                 # a discarded attempt leaves no recorder events
        try:
            with _no_alarm():
                lost = snap.restore()
                if lost:
                    entry["state_not_restored"] = lost[:20]
                if not entry.get("ok"):
                    _drop_events()
        except BaseException as e:                          # a time limit delivered right after the restore
            if entry.get("ok"):
                entry.update(ok=False, error=f"{type(e).__name__}: {str(e)[:160]}"); _drop_events()
            raise
        finally:
            ledger.append(entry)


def run(models, rec_model, rec_inputs, call, ledger, desc, recorder=DagRecorder, objects=()):
    """One attempt with its own recorder. Returns (recorder, output) of a call that completed; the caller credits ONLY
    this recorder. On an exception the recorder is discarded (hooks removed) and the exception re-raised. Either way
    the state is put back afterwards (isolated)."""
    rec = recorder(rec_model, rec_inputs)
    try:
        with isolated(models, ledger, desc, rec_inputs, objects):
            with torch.no_grad(), math_attention(), rec:
                out = call()
    except BaseException:
        rec.remove_hooks()
        raise
    return rec, out


def unrestored(ledger):
    """Every name an attempt changed and that could not be put back (a G2 unknown for the run)."""
    return sorted({n for e in ledger for n in e.get("state_not_restored", [])})


def ledger_record(ledger, cap=400):
    """The ledger as written into the result: every entry (capped, with the count)."""
    return {"attempts": len(ledger), "failed": sum(1 for e in ledger if not e.get("ok")), "entries": ledger[:cap]}
