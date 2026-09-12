"""Independent stdlib-only reverse-patch and AST audit; no project imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reverse_patch(after, patch):
    source = after.splitlines(keepends=True)
    old, cursor, i = [], 0, 0
    while i < len(patch):
        line = patch[i]
        if not line.startswith("@@"):
            i += 1
            continue
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        target = int(match[3]) - 1 if int(match[3]) else 0
        old.extend(source[cursor:target])
        cursor = target
        i += 1
        while i < len(patch) and not patch[i].startswith("@@"):
            line = patch[i]
            if line.startswith(" "):
                assert source[cursor] == line[1:]
                old.append(line[1:])
                cursor += 1
            elif line.startswith("+"):
                assert source[cursor] == line[1:]
                cursor += 1
            elif line.startswith("-"):
                old.append(line[1:])
            else:
                raise ValueError("unexpected patch record")
            i += 1
    old.extend(source[cursor:])
    return "".join(old)


class Imports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("test_support.s9_fixtures."):
            node.module = "test_" + node.module.rsplit(".", 1)[-1]
        return node


def normalized(node):
    return ast.dump(Imports().visit(copy.deepcopy(node)), include_attributes=False)


def symbols(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
    return out


def main():
    root = Path.cwd()
    held = root / "verification/receipts/S9-A-schema-candidate-fff99e22/new-resume/fixture-isolation"
    manifest_bytes = (held / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    segments = {}
    path = None
    for line in (held / "changes.patch").read_text().splitlines(keepends=True):
        if line.startswith("--- before/"):
            path = line[len("--- before/"):].strip()
            segments[path] = []
        elif line.startswith("+++ after/"):
            assert line[len("+++ after/"):].strip() == path
        else:
            segments[path].append(line)
    before, after = {}, {}
    pins, tests = [], []
    for row in manifest["files"]:
        name = row["path"]
        data = (held / "source" / name).read_bytes()
        assert digest(data) == row["sha256"]
        original = reverse_patch(data.decode(), segments[name]).encode()
        assert (digest(original) if original else None) == row["before_sha256"], name
        before[name], after[name] = ast.parse(original), ast.parse(data)
        pins.append({"path": name, "after_pin": True, "reconstructed_before_pin": True,
                     "current_matches_held": (root / name).read_bytes() == data})
        if name.startswith("tests/"):
            oldtests = {key: value for key, value in symbols(before[name]).items()
                        if key.startswith("test_") or key.startswith("Test")}
            newtests = {key: value for key, value in symbols(after[name]).items()
                        if key.startswith("test_") or key.startswith("Test")}
            assert set(oldtests) == set(newtests), name
            for key in oldtests:
                assert normalized(oldtests[key]) == normalized(newtests[key]), (name, key)
            tests.append({"path": name, "unchanged_test_definitions": len(oldtests)})
    moved = []
    for move in manifest["moves"]:
        old = symbols(before[move["source"]])
        new = symbols(after[move["destination"]])
        for name in move["definitions"]:
            same = normalized(old[name]) == normalized(new[name])
            if not same:
                assert name == "_CORPUS", (move, name)
                assert ast.unparse(new[name].value) == "Path(__file__).resolve().parents[2] / 'tests' / 'sable_test_corpus'"
                assert (root / move["destination"]).resolve().parents[2] / "tests" / "sable_test_corpus" == root / "tests" / "sable_test_corpus"
            moved.append({"source": move["source"], "destination": move["destination"],
                          "name": name, "same_ast_except_import_paths": same,
                          "corpus_relocation_only": not same})
    result = {"scope": "Saved-source AST audit, not runtime or importability approval",
              "manifest_sha256": digest(manifest_bytes), "files": pins, "tests": tests,
              "moves": moved, "test_definition_count": sum(row["unchanged_test_definitions"] for row in tests)}
    destination = held.parent / "reviews/fixture-isolation/result.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"files": len(pins), "test_definitions": result["test_definition_count"],
                      "moved_definitions": len(moved), "result_sha256": digest(destination.read_bytes())}))


if __name__ == "__main__":
    main()
