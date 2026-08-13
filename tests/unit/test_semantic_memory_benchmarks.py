import json

from examples.semantic_memory_benchmarks import (
    load_pararel,
    load_ripple,
    load_wikibigedit,
    manifest_summary,
    stream_checkpoints,
)


def test_pararel_uses_counterfactual_target_and_holds_out_templates(tmp_path):
    patterns = tmp_path / "patterns"; facts = tmp_path / "facts"
    patterns.mkdir(); facts.mkdir()
    (patterns / "P1.jsonl").write_text('\n'.join((json.dumps({"pattern":"[X] lives in [Y]."}), json.dumps({"pattern":"The home of [X] is [Y]."}))))
    (facts / "P1.jsonl").write_text('\n'.join((json.dumps({"sub_label":"Alice","obj_label":"Paris","uuid":"a"}), json.dumps({"sub_label":"Bob","obj_label":"Rome","uuid":"b"}))))
    cases = load_pararel(patterns, facts)
    assert cases[0].target != cases[0].metadata["original_target"]
    assert "[Y]" not in cases[0].prompt
    assert cases[0].queries[0].prompt != cases[0].prompt
    assert cases[0].queries[0].role == "should_propagate"


def test_ripple_preserves_condition_gating_and_boundary_roles(tmp_path):
    path = tmp_path / "random.json"
    testcase = {"test_queries":[{"prompt":"alias?","answers":[{"value":"new","aliases":["n"]}]}], "test_condition":"AND", "condition_queries":[{"prompt":"known?","answers":[{"value":"yes","aliases":[]}]}]}
    row = {"edit":{"prompt":"Alice lives in new.","relation":"R"}, "Subject_Aliasing":[testcase], "Relation_Specificity":[testcase]}
    path.write_text(json.dumps([row]))
    case = load_ripple([path])[0]
    queries = case.queries
    assert case.prompt == "Alice lives in"
    assert case.target == "new"
    assert {q.role for q in queries} == {"should_propagate", "should_not_propagate"}
    assert all(q.condition == "AND" and q.condition_prompts == ("known?",) for q in queries)
    assert manifest_summary(load_ripple([path]))["conditioned_queries"] == 2


def test_wikibigedit_is_chronological_and_defines_curve_points(tmp_path):
    paths=[]
    for name, subject in (("202401_202402","A"),("202402_202403","B")):
        folder=tmp_path/name; folder.mkdir(); path=folder/"qa.json"
        path.write_text(json.dumps([{"update":f"Where is {subject}?","rephrase":f"Location of {subject}?","ans":"X","loc":"Other?","loc_ans":"Y"}]))
        paths.append(path)
    cases=load_wikibigedit(paths)
    assert [case.metadata["increment"] for case in cases] == ["202401_202402","202402_202403"]
    assert stream_checkpoints(50_000,(1_000,5_000,10_000,50_000)) == (1_000,5_000,10_000,50_000)
    assert stream_checkpoints(7_000,(1_000,5_000,10_000)) == (1_000,5_000,7_000)
