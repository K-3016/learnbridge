from learnbridge.constraint_llm import LABELS, build_balanced_examples, parse_label


def test_constraint_examples_are_balanced(resources):
    examples = build_balanced_examples(resources, 2, 42)
    assert len(examples) == 8
    assert {label: sum(row["label"] == label for row in examples) for label in LABELS} == {
        label: 2 for label in LABELS
    }


def test_constraint_examples_are_disjoint(resources):
    training = build_balanced_examples(resources, 1, 42)
    ids = {row["resource_id"] for row in training}
    evaluation = build_balanced_examples(resources, 1, 1042, ids)
    assert not ids & {row["resource_id"] for row in evaluation}


def test_parse_constraint_label():
    assert parse_label("REJECT_BUDGET\n") == "REJECT_BUDGET"
    assert parse_label("unknown") is None
