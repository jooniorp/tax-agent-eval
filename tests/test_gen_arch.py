import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen_arch.py"
spec = importlib.util.spec_from_file_location("gen_arch", SCRIPT)
gen_arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_arch)


def write(root: Path, rel: str, code: str = "") -> None:
    path = root / "tax_agent" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")


def test_edges_cover_absolute_relative_and_from_package_imports(tmp_path):
    write(tmp_path, "__init__.py")
    write(tmp_path, "a.py", "from tax_agent import b\n")  # `from pkg import module`
    write(tmp_path, "b.py", "from tax_agent.pkg.c import X\n")  # absolute
    write(tmp_path, "d.py", "from . import b\n")  # relative
    write(tmp_path, "pkg/__init__.py")
    write(tmp_path, "pkg/c.py", "from .. import a\n")  # relative, one level up

    nodes, edges = gen_arch.build_edges(tmp_path)

    assert edges == {
        ("tax_agent.a", "tax_agent.b"),
        ("tax_agent.b", "tax_agent.pkg.c"),
        ("tax_agent.d", "tax_agent.b"),
        ("tax_agent.pkg.c", "tax_agent.a"),
    }
    assert "tax_agent" not in nodes and "tax_agent.pkg" not in nodes  # empty __init__ left out
    text = gen_arch.render(nodes, edges)
    assert 'subgraph pkg_pkg["pkg/"]' in text
    assert "tax_agent_a --> tax_agent_b" in text


def test_splice_replaces_only_the_generated_block():
    doc = f"hand-written top\n{gen_arch.START}\nOLD GRAPH\n{gen_arch.END}\nhand-written bottom\n"
    out = gen_arch.splice(doc, "NEW GRAPH")
    assert out == f"hand-written top\n{gen_arch.START}\nNEW GRAPH\n{gen_arch.END}\nhand-written bottom\n"
    assert gen_arch.splice(out, "NEW GRAPH") == out  # running twice changes nothing


def test_undocumented_reports_modules_missing_from_the_doc():
    nodes = {"tax_agent.tools", "tax_agent.mock_api.source", "tax_agent.agent"}
    doc = "| `tools` | ... |\n| `mock_api.source` | ... |\n"
    assert gen_arch.undocumented(nodes, doc) == ["tax_agent.agent"]
