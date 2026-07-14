from learnbridge.metrics import categorical_diversity, ndcg_at_k, precision_at_k, recall_at_k


def test_metric_ranges():
    values = [precision_at_k([2, 0, 1, 0, 2]), recall_at_k([2, 0, 1, 0, 2], 8), categorical_diversity(["a", "b", "a"])]
    assert all(0 <= x <= 1 for x in values)


def test_ndcg_perfect_is_one():
    assert ndcg_at_k([2, 2, 1, 0, 0], [2, 2, 1, 0, 0]) == 1


def test_ndcg_penalizes_bad_order():
    assert ndcg_at_k([0, 1, 2], [2, 1, 0]) < 1
