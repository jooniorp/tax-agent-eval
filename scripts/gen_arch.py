"""Generate docs/architecture.md: a Mermaid import graph of src/tax_agent.

Run: uv run python scripts/gen_arch.py
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "docs" / "architecture.md"

START = "<!-- gen:start -->"  # only the text between these markers is regenerated
END = "<!-- gen:end -->"


def module_name(path: Path, src: Path) -> str:
    parts = path.relative_to(src).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def build_edges(src: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (modules to draw, import edges). Empty package __init__ files are left out."""
    files = sorted((src / "tax_agent").rglob("*.py"))
    modules = {module_name(f, src) for f in files}
    packages = {module_name(f, src) for f in files if f.name == "__init__.py"}
    edges: set[tuple[str, str]] = set()
    for f in files:
        name = module_name(f, src)
        package = name if f.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:  # relative import: anchor at the importing package
                    anchor = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                    base = ".".join(anchor + ([node.module] if node.module else []))
                # `from pkg import mod` depends on pkg.mod, not on pkg itself
                subs = [f"{base}.{a.name}" for a in node.names if f"{base}.{a.name}" in modules]
                targets = subs or [base]
            else:
                continue
            edges.update((name, t) for t in targets if t in modules and t != name)
    drawn = (modules - packages) | {m for edge in edges for m in edge}
    return drawn, edges


def render(nodes: set[str], edges: set[tuple[str, str]]) -> str:
    node_id = lambda m: m.replace(".", "_")  # noqa: E731
    groups: dict[str, list[str]] = {}
    for m in sorted(nodes):
        groups.setdefault(".".join(m.split(".")[1:-1]), []).append(m)
    lines = ["graph LR"]
    for group, mods in sorted(groups.items()):
        indent = "  "
        if group:
            lines.append(f'  subgraph pkg_{group.replace(".", "_")}["{group}/"]')
            indent = "    "
        lines += [f'{indent}{node_id(m)}["{m.rpartition(".")[2]}"]' for m in mods]
        if group:
            lines.append("  end")
    lines += [f"  {node_id(a)} --> {node_id(b)}" for a, b in sorted(edges)]
    return "\n".join(lines)


def splice(doc: str, block: str) -> str:
    """Replace the generated block between the markers; keep everything outside them."""
    head, found_start, rest = doc.partition(START)
    _, found_end, tail = rest.partition(END)
    if not (found_start and found_end):
        raise SystemExit(f"{START} / {END} markers not found in {OUT.name}")
    return f"{head}{START}\n{block}\n{END}{tail}"


def undocumented(nodes: set[str], doc: str) -> list[str]:
    """Modules that the hand-written part never mentions as `module` (without the package prefix)."""
    return sorted(m for m in nodes if f"`{m.removeprefix('tax_agent.')}`" not in doc)


def main() -> None:
    nodes, edges = build_edges(SRC)
    block = "```mermaid\n" + render(nodes, edges) + "\n```"
    text = splice(OUT.read_text(encoding="utf-8"), block)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(nodes)} modules, {len(edges)} imports)")
    if missing := undocumented(nodes, text):
        print("설명이 없는 모듈 (docs/architecture.md의 '모듈별 역할'에 추가): " + ", ".join(missing))


if __name__ == "__main__":
    main()
