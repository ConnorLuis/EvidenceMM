from evidencemm.retrieval_fusion import reciprocal_rank_fusion


def test_rrf_prefers_shared_rank():
    result = reciprocal_rank_fusion([
        ["a", "b"],
        ["a", "c"],
    ])

    assert result[0][0] == "a"
