import json

from tax_agent.evaluation import compare

MODEL = "claude-haiku-5-5"


def row(item_id, variant, topic, out, seconds, group="basic", passed=True, **extra):
    return {"id": item_id, "variant": variant, "topic": topic, "group": group, "model": MODEL, "passed": passed,
            "tokens_in": 1000, "tokens_out": out, "seconds": seconds, **extra}


def test_item_means_average_the_runs_and_skip_fields_a_run_does_not_have():
    rows = [
        row("a", "calculator", "card", 100, 2.0, reasoning_tokens=10, model_calls=3),
        row("a", "calculator", "card", 300, 4.0),  # saved before the new fields existed
    ]
    means = compare.item_means(rows, MODEL, "calculator", "basic")["a"]
    assert (means["runs"], means["tokens_out"], means["seconds"]) == (2, 200, 3.0)
    assert (means["reasoning_tokens"], means["model_calls"]) == (10, 3)  # mean over the runs that have them
    assert means["model_seconds"] is None  # no run has it


def test_change_is_a_signed_percentage_or_a_dash():
    assert compare.change(100, 145) == "+45%" and compare.change(100, 85) == "-15%"
    assert compare.change(None, 5) == "-" and compare.change(0, 5) == "-" and compare.change(5, None) == "-"


def test_the_report_compares_the_variants_by_group_and_by_topic():
    rows = [
        row("card-1", "calculator", "card", 1000, 8.0, reasoning_tokens=300, model_calls=3, model_seconds=7.0),
        row("card-1", "no_calculator", "card", 2000, 12.0, reasoning_tokens=1200, model_calls=2, model_seconds=11.5),
        row("rent-1", "calculator", "rent", 700, 5.0, reasoning_tokens=200, model_calls=3, model_seconds=4.0),
        row("rent-1", "no_calculator", "rent", 600, 4.0, reasoning_tokens=300, model_calls=2, model_seconds=3.8),
    ]
    text = compare.report(rows)
    assert f"{MODEL} · group basic (2 items)" in text
    assert "out tokens +53%" in text and "seconds +23%" in text  # 2600 vs 1700 tokens, 16 s vs 13 s
    card_line = next(line for line in text.splitlines() if line.strip().startswith("card"))
    rent_line = next(line for line in text.splitlines() if line.strip().startswith("rent"))
    assert "+100%" in card_line and "+50%" in card_line  # the long calculation costs the control more
    assert "-14%" in rent_line and "-20%" in rent_line  # the short one costs it less


def test_a_group_with_only_one_variant_is_listed_without_a_comparison():
    text = compare.report([row("x", "calculator", "card", 500, 3.0, group="sweep")])
    assert "group sweep (1 items)" in text and "control vs calculator" not in text


def test_rescore_judges_the_stored_answers_again_and_flags_the_verdicts_that_changed(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    items = [{"id": "edge-01", "expected": {"unsupported": True}}, {"id": "card-01", "expected": {"amount": 5_300_000}}]
    dataset.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    rows = [
        row("edge-01", "calculator", "edge", 1, 1.0, passed=False, answer_ok=False, rules_ok=True,
            answer="교육비는 기능이 없어서 알려드릴 수 없습니다."),  # was missed by the first wording rule
        row("card-01", "calculator", "card", 1, 1.0, passed=True, answer_ok=True, rules_ok=True, answer="530만 원입니다."),
        row("card-01", "no_calculator", "card", 1, 1.0, passed=False, answer_ok=False, rules_ok=True, answer="195만 원입니다."),
        row("card-01", "no_calculator", "card", 1, 1.0, passed=False, answer_ok=False, rules_ok=True, error="timeout", answer=""),
    ]
    out = compare.rescore(rows, str(dataset))
    assert [r["passed"] for r in out] == [True, True, False, False]
    assert [r.get("rescored") for r in out] == [True, False, False, None]  # an errored run is left alone
    assert rows[0]["passed"] is False  # the input rows are not modified


def test_load_rows_reads_the_saved_files_and_filters_by_prefix_and_model(tmp_path):
    def save(name, model, variant):
        record = {"model": model, "variant": variant, "results": [{"id": "a", "topic": "card", "passed": True,
                  "tokens_in": 1, "tokens_out": 1, "seconds": 1.0}]}
        (tmp_path / name).write_text(json.dumps(record), encoding="utf-8")

    save("20261114-1.json", MODEL, "no_calculator")
    save("20261115-1.json", MODEL, "calculator")
    save("20261114-2.json", "other-model", "calculator")
    rows = compare.load_rows(str(tmp_path), prefix="20261114")
    assert sorted((r["model"], r["variant"]) for r in rows) == [(MODEL, "no_calculator"), ("other-model", "calculator")]
    assert [r["model"] for r in compare.load_rows(str(tmp_path), model=MODEL)] == [MODEL, MODEL]
    assert [r["variant"] for r in compare.load_rows(str(tmp_path), since="20261114-2")] == ["calculator", "calculator"]  # drops the first file
    assert compare.report([]) == "no saved runs found"


def test_the_report_adds_a_line_for_the_combined_variant_against_the_calculator():
    rows = [
        row("card-1", "calculator", "card", 1000, 8.0, model_calls=4),
        row("card-1", "no_calculator", "card", 1500, 9.0, model_calls=3),
        row("card-1", "combined", "card", 800, 5.0, model_calls=2),
        row("sweep-1", "calculator", "card", 1000, 8.0, group="sweep"),
        row("sweep-1", "combined", "card", 800, 4.0, group="sweep"),
    ]
    text = compare.report(rows)
    assert "control vs calculator: " in text and "out tokens +50%" in text
    combined = next(line for line in text.splitlines() if line.startswith("combined vs calculator"))
    assert "out tokens -20%" in combined and "seconds -38%" in combined and "model calls -50%" in combined
    sweep = text[text.index("group sweep"):]
    assert "combined vs calculator" in sweep and "control vs calculator" not in sweep  # only the variants that ran
    assert next(line for line in sweep.splitlines() if line.strip().startswith("sweep-1")).count("%") == 2


def test_the_same_variant_under_two_models_is_compared_with_the_default_model_as_baseline():
    other = "gpt-6.2-sol"
    rows = [
        row("card-1", "combined", "card", 1000, 8.0, model_calls=2),
        row("card-1", "combined", "card", 2000, 6.0, model_calls=2, passed=False),
        row("rent-1", "combined", "rent", 500, 4.0, model_calls=2),
    ]
    rows += [{**r, "model": other, "tokens_out": r["tokens_out"] // 2, "seconds": r["seconds"] * 2, "passed": True} for r in rows]
    text = compare.report(rows)
    section = text[text.index("== models · group basic · variant combined"):]
    lines = section.splitlines()
    assert lines[2].startswith(MODEL) and lines[3].startswith(other)  # the default model comes first
    assert f"{other} vs {MODEL}: " in section and "out tokens -50%" in section and "seconds +100%" in section
    assert f"{MODEL} " in text and "== models" not in text.replace(section, "")  # only one such part
    only_one_model = compare.report(rows[:3])
    assert "== models" not in only_one_model


def test_rescore_also_judges_the_checks_that_read_only_the_answer_text_again(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    item = {"id": "med-01", "expected": {"amount": 0},
            "checks": [{"type": "answer_contains", "all": ["보험금|돌려받"]}, {"type": "tool_called", "name": "compute_deduction"}]}
    dataset.write_text(json.dumps(item), encoding="utf-8")
    checks = [{"name": "answer_contains", "ok": False, "detail": "missing"}, {"name": "tool_called", "ok": True, "detail": "calls"}]
    wrong = row("med-01", "combined", "medical", 1, 1.0, passed=False, answer_ok=True, rules_ok=False,
                answer="돌려받은 금액을 빼면 공제는 0원입니다.", turn_answers=["돌려받은 금액을 빼면 공제는 0원입니다."], checks=checks)
    (fixed,) = compare.rescore([wrong], str(dataset))
    assert fixed["passed"] and fixed["rules_ok"] and fixed["rescored"]
    assert [c["ok"] for c in fixed["checks"]] == [True, True]  # the tool check keeps its stored result
    still = {**wrong, "answer": "공제는 0원입니다.", "turn_answers": ["공제는 0원입니다."]}
    assert not compare.rescore([still], str(dataset))[0]["passed"]  # nothing about the payout: still a failure


def test_rescore_checks_the_amount_the_calculator_gave_again_from_the_stored_detail(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"id": "med-04", "expected": {"amount": 375_000}, "checks": ["standard_calc"]}), encoding="utf-8")
    checks = [{"name": "tool_called", "ok": True, "detail": "calls"},
              {"name": "answer_uses_tool_amount", "ok": False, "detail": "tool said [375000]; answer has [370000]"}]
    answer = "의료비 세액공제액은 37만 5천 원입니다."
    failed = row("med-04", "combined", "medical", 1, 1.0, passed=False, answer_ok=False, rules_ok=False,
                 answer=answer, turn_answers=[answer], checks=checks)
    fixed = compare.rescore([failed], str(dataset))[0]
    assert fixed["passed"] and fixed["rescored"] and fixed["checks"][1]["ok"]
    wrong = {**failed, "answer": "공제액은 38만 원입니다.", "turn_answers": ["공제액은 38만 원입니다."]}
    assert not compare.rescore([wrong], str(dataset))[0]["passed"]  # a really different amount stays a failure
