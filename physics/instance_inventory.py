"""Bounded, network-disabled meta-device instance inventory.

The parent process never imports the requested framework class.  A fresh child
process imports and constructs it under ``torch.device("meta")``, writes one
typed result file, and exits.  A timeout, memory ceiling, network attempt,
import error, constructor error, or serialization error is a typed failure;
there is deliberately no fallback model or familiar structure.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any, Mapping


SCHEMA_VERSION = 1
_CAPTURE_LIMIT = 65536
_FAILURE_KINDS = frozenset({
    "NetworkRefused", "TimeoutExpired", "MemoryLimitExceeded",
    "ImportFailed", "ConfigurationFailed", "ConstructionFailed",
    "SerializationFailed", "WorkerFailed", "ExecutionFailed",
    "ExecutionUnresolved",
})
_ENVIRONMENT_KEYS = frozenset({
    "python", "platform", "hash_seed", "network", "hf_hub_offline",
    "transformers_offline", "diffusers_offline",
})


def _is_sha256(value: str) -> bool:
    return (len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


class NetworkRefused(RuntimeError):
    """Raised inside the child when installed code attempts network access."""


@dataclasses.dataclass(frozen=True)
class BuildRequest:
    config: Mapping[str, Any]
    framework: str
    factory_module: str
    factory_qualname: str
    factory_method: str = "__call__"
    config_module: str | None = None
    config_qualname: str | None = None
    config_method: str | None = None
    build_flags: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    import_paths: tuple[str, ...] = ()
    timeout_seconds: float = 120.0
    memory_limit_bytes: int = 16 * 1024**3
    label: str = "unnamed"

    def __post_init__(self) -> None:
        if self.framework not in {"custom", "transformers", "diffusers"}:
            raise ValueError("framework must be custom, transformers, or diffusers")
        for value in (self.factory_module, self.factory_qualname):
            if not value or not isinstance(value, str):
                raise ValueError("factory module and qualname are required")
        if self.factory_method not in {"__call__", "from_config", "_from_config"}:
            raise ValueError("unsupported constructor/factory method")
        if self.timeout_seconds <= 0 or self.memory_limit_bytes <= 0:
            raise ValueError("timeout and memory limit must be positive")
        if self.build_flags.get("trust_remote_code"):
            raise ValueError("remote custom code is outside the S6 contract")
        # Prove the payload can cross the process boundary before launching it.
        json.dumps(self.to_dict(), sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["config"] = dict(self.config)
        row["build_flags"] = dict(self.build_flags)
        row["import_paths"] = list(self.import_paths)
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "BuildRequest":
        return cls(**dict(row))


@dataclasses.dataclass(frozen=True)
class Failure:
    kind: str
    stage: str
    detail: str

    def __post_init__(self) -> None:
        if self.kind not in _FAILURE_KINDS or not self.stage or not self.detail:
            raise ValueError("failure must have a closed kind, stage, and detail")


@dataclasses.dataclass(frozen=True)
class PackageVersion:
    package: str
    version: str

    def __post_init__(self) -> None:
        if not self.package or not self.version:
            raise ValueError("package provenance requires a name and version")


@dataclasses.dataclass(frozen=True)
class SourceFile:
    module: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.module or not self.path or not _is_sha256(self.sha256):
            raise ValueError("source file requires a SHA-256 digest")


@dataclasses.dataclass(frozen=True)
class ResolvedClass:
    module: str
    qualname: str

    def __post_init__(self) -> None:
        if not self.module or not self.qualname:
            raise ValueError("class provenance requires module and qualname")


@dataclasses.dataclass(frozen=True)
class Provenance:
    packages: tuple[PackageVersion, ...]
    source_files: tuple[SourceFile, ...]
    config_sha256: str
    resolved_class: ResolvedClass
    requested_factory: str
    constructor_used: str
    build_flags: Mapping[str, Any]
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        if (not _is_sha256(self.config_sha256)
                or not self.packages or not self.source_files):
            raise ValueError("provenance is incomplete")
        if len(set(self.packages)) != len(self.packages):
            raise ValueError("package provenance must be unique")
        if len(set(self.source_files)) != len(self.source_files):
            raise ValueError("source-file provenance must be unique")
        if not self.requested_factory or not self.constructor_used:
            raise ValueError("factory provenance is required")
        if not _ENVIRONMENT_KEYS <= set(self.environment):
            raise ValueError("deterministic environment record is incomplete")
        json.dumps(dict(self.build_flags), sort_keys=True)


@dataclasses.dataclass(frozen=True)
class ParameterShape:
    name: str
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool

    def __post_init__(self) -> None:
        if (not self.name or not self.dtype
                or any(not isinstance(dim, int) or dim < 0 for dim in self.shape)
                or not isinstance(self.requires_grad, bool)):
            raise ValueError("parameter shape record is invalid")


@dataclasses.dataclass(frozen=True)
class ModuleNode:
    path: str
    class_ref: ResolvedClass
    origin_module: str
    mro_entries: tuple[ResolvedClass, ...]
    children: tuple[str, ...]
    parameters: tuple[ParameterShape, ...]
    init_attributes: Mapping[str, Any]
    guarded_none_children: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if self.origin_module != self.class_ref.module:
            raise ValueError("module origin must agree with its exact class")
        if not self.mro_entries or self.mro_entries[0] != self.class_ref:
            raise ValueError("module MRO must begin with its exact class")
        if len(set(self.children)) != len(self.children):
            raise ValueError("direct child names must be unique")
        names = tuple(row.name for row in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("direct parameter names must be unique")


@dataclasses.dataclass(frozen=True)
class RepetitionGroup:
    parent_path: str
    signature_sha256: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not _is_sha256(self.signature_sha256) or len(self.members) < 2
                or len(set(self.members)) != len(self.members)):
            raise ValueError("a repetition group needs a signature and two members")


@dataclasses.dataclass(frozen=True)
class ParameterAliasGroup:
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.names) < 2 or tuple(sorted(set(self.names))) != self.names:
            raise ValueError("alias names must be unique, sorted, and repeated")


@dataclasses.dataclass(frozen=True)
class InstanceInventory:
    schema_version: int
    provenance: Provenance
    modules: tuple[ModuleNode, ...]
    repetition_groups: tuple[RepetitionGroup, ...]
    parameter_aliases: tuple[ParameterAliasGroup, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.modules:
            raise ValueError("inventory must contain a root module")
        paths = tuple(node.path for node in self.modules)
        if paths[0] != "" or len(paths) != len(set(paths)):
            raise ValueError("module paths must have one unique root")
        path_set = set(paths)
        for node in self.modules:
            for child in node.children:
                child_path = f"{node.path}.{child}".lstrip(".")
                if child_path not in path_set:
                    raise ValueError("module child is missing from the exact tree")
        for group in self.repetition_groups:
            if group.parent_path not in path_set or not set(group.members) <= path_set:
                raise ValueError("repetition group cites a foreign module path")
        parameter_names = {
            f"{node.path}.{parameter.name}".lstrip(".")
            for node in self.modules for parameter in node.parameters
        }
        if any(not set(group.names) <= parameter_names
               for group in self.parameter_aliases):
            raise ValueError("parameter alias cites a foreign parameter")


@dataclasses.dataclass(frozen=True)
class InventoryResult:
    status: str
    inventory: InstanceInventory | None = None
    failure: Failure | None = None
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if self.status == "ok":
            if self.inventory is None or self.failure is not None:
                raise ValueError("ok result requires only an inventory")
        elif self.status == "failed":
            if self.failure is None or self.inventory is not None:
                raise ValueError("failed result requires only a failure")
        else:
            raise ValueError("result status must be ok or failed")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "InventoryResult":
        inv = row.get("inventory")
        failure = row.get("failure")
        return cls(
            status=str(row["status"]),
            inventory=_inventory_from_dict(inv) if inv else None,
            failure=Failure(**failure) if failure else None,
            stdout=str(row.get("stdout", "")),
            stderr=str(row.get("stderr", "")),
        )


def _inventory_from_dict(row: Mapping[str, Any]) -> InstanceInventory:
    p = row["provenance"]
    provenance = Provenance(
        packages=tuple(PackageVersion(**x) for x in p["packages"]),
        source_files=tuple(SourceFile(**x) for x in p["source_files"]),
        config_sha256=p["config_sha256"],
        resolved_class=ResolvedClass(**p["resolved_class"]),
        requested_factory=p["requested_factory"],
        constructor_used=p["constructor_used"],
        build_flags=p["build_flags"],
        environment=p["environment"],
    )
    modules = []
    for node in row["modules"]:
        modules.append(ModuleNode(
            path=node["path"], class_ref=ResolvedClass(**node["class_ref"]),
            origin_module=node["origin_module"],
            mro_entries=tuple(ResolvedClass(**x) for x in node["mro_entries"]),
            children=tuple(node["children"]),
            parameters=tuple(ParameterShape(
                name=x["name"], shape=tuple(x["shape"]), dtype=x["dtype"],
                requires_grad=x["requires_grad"]) for x in node["parameters"]),
            init_attributes=node["init_attributes"],
            guarded_none_children=tuple(node["guarded_none_children"]),
        ))
    return InstanceInventory(
        schema_version=row["schema_version"], provenance=provenance,
        modules=tuple(modules),
        repetition_groups=tuple(RepetitionGroup(
            parent_path=x["parent_path"], signature_sha256=x["signature_sha256"],
            members=tuple(x["members"])) for x in row["repetition_groups"]),
        parameter_aliases=tuple(ParameterAliasGroup(tuple(x["names"]))
                                for x in row["parameter_aliases"]),
    )


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def _resolve(module_name: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        if part == "<locals>" or part.startswith("__"):
            raise AttributeError(f"unimportable qualname {qualname!r}")
        value = getattr(value, part)
    return value


def _network_guard(event: str, _args: tuple[Any, ...]) -> None:
    denied = {
        "socket.bind", "socket.connect", "socket.getaddrinfo",
        "socket.gethostbyaddr", "socket.gethostbyname", "socket.sendto",
    }
    if event == "subprocess.Popen":
        executable = Path(str(_args[0])).name.lower() if _args else ""
        if executable in {"curl", "wget", "ftp", "ssh", "scp", "sftp"}:
            raise NetworkRefused("network client process refused")
    if event in denied or event.startswith("urllib.Request"):
        raise NetworkRefused(f"network audit event refused: {event}")


def _install_network_guard() -> None:
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "NO_PROXY": "*", "no_proxy": "*", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "0",
    })
    sys.addaudithook(_network_guard)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "network": os.environ.get("UNFOLD_NETWORK_SANDBOX", "audit-hook-refused"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "diffusers_offline": os.environ.get("DIFFUSERS_OFFLINE", ""),
    }


def _config_object(request: BuildRequest) -> Any:
    # Constructors are allowed to normalize/pop nested config structures. Give
    # them a deep JSON copy so the request remains the immutable raw-value
    # authority used by the provenance hash.
    config = json.loads(json.dumps(request.config))
    if request.config_module is None:
        return config
    target = _resolve(request.config_module, request.config_qualname or "")
    if request.config_method == "for_model":
        model_type = config.pop("model_type")
        return target.for_model(model_type, **config)
    if request.config_method:
        return getattr(target, request.config_method)(config)
    return target(**config)


def _construct(request: BuildRequest) -> tuple[Any, str]:
    import torch

    factory = _resolve(request.factory_module, request.factory_qualname)
    config = _config_object(request)
    method = request.factory_method
    with torch.device("meta"):
        if method == "__call__":
            model = factory(config, **dict(request.build_flags))
            used = f"{request.factory_module}.{request.factory_qualname}(config)"
        else:
            model = getattr(factory, method)(config, **dict(request.build_flags))
            used = f"{request.factory_module}.{request.factory_qualname}.{method}(config)"
    return model, used


def _class_ref(cls: type) -> ResolvedClass:
    return ResolvedClass(cls.__module__, cls.__qualname__)


def _package_for(module_name: str) -> PackageVersion:
    top = module_name.split(".", 1)[0]
    if top == "builtins":
        return PackageVersion("python", platform.python_version())
    names = sorted(importlib.metadata.packages_distributions().get(top, ()))
    for name in names:
        try:
            return PackageVersion(name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        module = importlib.import_module(top)
        version = str(getattr(module, "__version__", "local"))
    except (ImportError, AttributeError):
        version = "local"
    return PackageVersion(top, version)


def _source_for_class(cls: type) -> SourceFile | None:
    try:
        source = inspect.getsourcefile(cls)
    except (TypeError, OSError):
        return None
    if not source or not Path(source).is_file():
        return None
    path = Path(source)
    return SourceFile(cls.__module__, path.name,
                      hashlib.sha256(path.read_bytes()).hexdigest())


def _safe_value(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= 2:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v, depth + 1) for v in value[:64]]
    if isinstance(value, dict):
        return {str(k): _safe_value(v, depth + 1)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))[:64]}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": str(value) if type(value).__module__ == "torch" else None}


def _guarded_none(cls: type, module_path: str, instance: Any) -> tuple[dict[str, Any], ...]:
    try:
        lines, start = inspect.getsourcelines(cls.__init__)
        filename = inspect.getsourcefile(cls.__init__)
        source = textwrap.dedent("".join(lines))
        tree = ast.parse(source)
    except (TypeError, OSError, IndentationError, SyntaxError):
        return ()
    rows: list[dict[str, Any]] = []

    class GuardedNoneVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.guards: list[tuple[str, bool]] = []

        def visit_If(self, node: ast.If) -> None:
            predicate = ast.get_source_segment(source, node.test) or ast.unparse(node.test)
            self.guards.append((predicate, True))
            for statement in node.body:
                self.visit(statement)
            self.guards.pop()
            self.guards.append((predicate, False))
            for statement in node.orelse:
                self.visit(statement)
            self.guards.pop()

        def _record(self, node: ast.AST, targets: list[ast.AST],
                    guards: tuple[tuple[str, bool], ...]) -> None:
            if not guards:
                return
            for target in targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and getattr(instance, target.attr, object()) is None):
                    rows.append({
                        "path": f"{module_path}.{target.attr}".lstrip("."),
                        "attribute": target.attr, "value": None,
                        "source_file": Path(filename).name if filename else "",
                        "line": start + node.lineno - 1,
                        "guards": [
                            {"predicate": predicate, "branch": branch}
                            for predicate, branch in guards
                        ],
                    })

        def _assignment(self, node: ast.AST, targets: list[ast.AST],
                        value: ast.AST | None) -> None:
            if isinstance(value, ast.Constant) and value.value is None:
                self._record(node, targets, tuple(self.guards))
            elif isinstance(value, ast.IfExp):
                predicate = (ast.get_source_segment(source, value.test)
                             or ast.unparse(value.test))
                if isinstance(value.body, ast.Constant) and value.body.value is None:
                    self._record(node, targets, (*self.guards, (predicate, True)))
                if isinstance(value.orelse, ast.Constant) and value.orelse.value is None:
                    self._record(node, targets, (*self.guards, (predicate, False)))

        def visit_Assign(self, node: ast.Assign) -> None:
            self._assignment(node, node.targets, node.value)
            self.generic_visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self._assignment(node, [node.target], node.value)
            if node.value is not None:
                self.generic_visit(node.value)

    GuardedNoneVisitor().visit(tree)
    return tuple(sorted(rows, key=lambda row: (row["path"], row["line"])))


def _direct_parameter_shapes(module: Any) -> tuple[ParameterShape, ...]:
    return tuple(ParameterShape(name, tuple(int(x) for x in p.shape), str(p.dtype),
                                bool(p.requires_grad))
                 for name, p in module.named_parameters(recurse=False,
                                                        remove_duplicate=False))


def _signature(module: Any) -> str:
    row = {
        "class": [type(module).__module__, type(module).__qualname__],
        "children": [(name, type(child).__module__, type(child).__qualname__)
                     for name, child in module.named_children()],
        "parameters": [(p.name, p.shape, p.dtype) for p in _direct_parameter_shapes(module)],
    }
    return hashlib.sha256(_canon(row)).hexdigest()


def inventory_model(model: Any, request: BuildRequest, constructor_used: str) -> InstanceInventory:
    modules: list[ModuleNode] = []
    repetitions: list[RepetitionGroup] = []
    classes: set[type] = set()
    for path, module in model.named_modules(remove_duplicate=False):
        cls = type(module)
        classes.update(cls.__mro__)
        attrs = {name: _safe_value(value) for name, value in sorted(vars(module).items())
                 if name not in {"_modules", "_parameters", "_buffers"}}
        modules.append(ModuleNode(
            path=path, class_ref=_class_ref(cls), origin_module=cls.__module__,
            mro_entries=tuple(_class_ref(c) for c in cls.__mro__),
            children=tuple(name for name, _ in module.named_children()),
            parameters=_direct_parameter_shapes(module),
            init_attributes=attrs,
            guarded_none_children=_guarded_none(cls, path, module),
        ))
        groups: dict[str, list[str]] = {}
        for name, child in module.named_children():
            groups.setdefault(_signature(child), []).append(f"{path}.{name}".lstrip("."))
        for signature, members in groups.items():
            if len(members) > 1:
                repetitions.append(RepetitionGroup(path, signature, tuple(members)))
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    alias_rows = tuple(ParameterAliasGroup(tuple(sorted(names)))
                       for names in aliases.values() if len(set(names)) > 1)
    packages = tuple(sorted({_package_for(c.__module__) for c in classes},
                            key=lambda row: (row.package, row.version)))
    sources = tuple(sorted({row for c in classes if (row := _source_for_class(c))},
                           key=lambda row: (row.module, row.path, row.sha256)))
    provenance = Provenance(
        packages=packages, source_files=sources,
        config_sha256=hashlib.sha256(_canon(dict(request.config))).hexdigest(),
        resolved_class=_class_ref(type(model)),
        requested_factory=f"{request.factory_module}.{request.factory_qualname}",
        constructor_used=constructor_used, build_flags=dict(request.build_flags),
        environment=_environment(),
    )
    return InstanceInventory(SCHEMA_VERSION, provenance, tuple(modules),
                             tuple(repetitions), alias_rows)


def _failure(kind: str, stage: str, exc: BaseException) -> InventoryResult:
    return InventoryResult("failed", failure=Failure(
        kind, stage, f"{type(exc).__name__}: {str(exc)[:2000]}"))


def _worker(request_path: Path, result_path: Path) -> int:
    _install_network_guard()
    try:
        request = BuildRequest.from_dict(json.loads(request_path.read_text()))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        result = _failure("ConfigurationFailed", "request", exc)
    else:
        for path in reversed(request.import_paths):
            sys.path.insert(0, path)
        try:
            model, used = _construct(request)
            result = InventoryResult("ok", inventory=inventory_model(model, request, used))
        except NetworkRefused as exc:
            result = _failure("NetworkRefused", "construct", exc)
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            result = _failure("ImportFailed", "import", exc)
        except MemoryError as exc:
            result = _failure("MemoryLimitExceeded", "construct", exc)
        except (TypeError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            result = _failure("ConstructionFailed", "construct", exc)
        except BaseException as exc:  # child boundary: every library failure is typed
            result = _failure("WorkerFailed", "construct", exc)
    try:
        result_path.write_text(json.dumps(result.to_dict(), sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        fallback = _failure("SerializationFailed", "serialize", exc)
        try:
            result_path.write_text(json.dumps(fallback.to_dict(), sort_keys=True))
        except OSError:
            pass
        return 2


def _kill_group(process: subprocess.Popen) -> None:
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()


def _network_isolated_command(command: list[str], env: dict[str, str]) -> list[str]:
    """Wrap the entire child process in the host's network-denial boundary."""
    sandbox = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if sandbox:
        env["UNFOLD_NETWORK_SANDBOX"] = "macos-sandbox-exec+python-audit-hook"
        return [sandbox, "-p", "(version 1) (allow default) (deny network*)",
                *command]
    # The audit hook and offline variables still block in-process Python
    # clients. The recorded value makes the missing OS boundary explicit; a
    # later productionisation unit must add the host-specific wrapper rather
    # than silently calling this equivalent to the macOS pilot.
    env["UNFOLD_NETWORK_SANDBOX"] = "python-audit-hook-only"
    return command


def _communicate_bounded(process: subprocess.Popen, *, timeout: float,
                         memory_limit: int) -> tuple[str, str, str | None]:
    """Supervise wall time/RSS while continuously draining bounded output.

    Reading concurrently is part of the isolation contract: an imported
    constructor that writes more than an OS pipe buffer must not deadlock the
    worker and masquerade as a timeout.
    """
    tails = {"stdout": "", "stderr": ""}

    def drain(name: str, stream: Any) -> None:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            tails[name] = (tails[name] + chunk)[-_CAPTURE_LIMIT:]

    threads = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    def finish(reason: str | None) -> tuple[str, str, str | None]:
        process.wait()
        for thread in threads:
            thread.join(timeout=2)
        return tails["stdout"], tails["stderr"], reason

    try:
        import psutil
    except ImportError as exc:
        _kill_group(process)
        return finish(f"monitor_unavailable:{exc}")
    started = time.monotonic()
    root = psutil.Process(process.pid)
    while process.poll() is None:
        if time.monotonic() - started > timeout:
            _kill_group(process)
            return finish("timeout")
        try:
            processes = (root, *root.children(recursive=True))
            rss = sum(item.memory_info().rss for item in processes
                      if item.is_running())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss = 0
        if rss > memory_limit:
            _kill_group(process)
            return finish(f"memory:{rss}")
        time.sleep(0.02)
    return finish(None)


def inventory_in_subprocess(request: BuildRequest) -> InventoryResult:
    """Run one inventory with a wall timeout, address-space cap, and no network."""
    with tempfile.TemporaryDirectory(prefix="unfold-s6-") as tmp:
        root = Path(tmp)
        request_path, result_path = root / "request.json", root / "result.json"
        request_path.write_text(json.dumps(request.to_dict(), sort_keys=True))
        env = os.environ.copy()
        env.update({"PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false"})
        cmd = [sys.executable, "-m", "physics.instance_inventory", "--worker",
               str(request_path), str(result_path)]
        cmd = _network_isolated_command(cmd, env)
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
            env=env, start_new_session=True)
        stdout, stderr, termination = _communicate_bounded(
            process, timeout=request.timeout_seconds,
            memory_limit=request.memory_limit_bytes)
        if termination == "timeout":
            return InventoryResult("failed", failure=Failure(
                "TimeoutExpired", "construct",
                f"exceeded {request.timeout_seconds:g}s wall timeout"),
                stdout=stdout[-_CAPTURE_LIMIT:], stderr=stderr[-_CAPTURE_LIMIT:])
        if termination and termination.startswith("memory:"):
            peak = int(termination.split(":", 1)[1])
            return InventoryResult("failed", failure=Failure(
                "MemoryLimitExceeded", "construct",
                f"process-tree RSS {peak} exceeded {request.memory_limit_bytes} bytes"),
                stdout=stdout[-_CAPTURE_LIMIT:], stderr=stderr[-_CAPTURE_LIMIT:])
        if termination:
            return InventoryResult("failed", failure=Failure(
                "ConfigurationFailed", "memory_monitor", termination),
                stdout=stdout[-_CAPTURE_LIMIT:], stderr=stderr[-_CAPTURE_LIMIT:])
        if result_path.exists():
            try:
                result = InventoryResult.from_dict(json.loads(result_path.read_text()))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                result = _failure("SerializationFailed", "parent_decode", exc)
        else:
            memory_hint = (process.returncode is not None and process.returncode < 0) or any(
                word in stderr.lower() for word in ("memory", "cannot allocate", "malloc"))
            result = InventoryResult("failed", failure=Failure(
                "MemoryLimitExceeded" if memory_hint else "WorkerFailed", "worker_exit",
                f"worker exited {process.returncode} without a typed result"))
        return dataclasses.replace(result, stdout=stdout[-_CAPTURE_LIMIT:],
                                   stderr=stderr[-_CAPTURE_LIMIT:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("REQUEST", "RESULT"))
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("this module's CLI is an internal subprocess worker")
    return _worker(Path(args.worker[0]), Path(args.worker[1]))


if __name__ == "__main__":
    raise SystemExit(main())
