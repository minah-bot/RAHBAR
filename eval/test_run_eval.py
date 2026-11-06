from run_eval import citation_validity, gate, recall_at_k, summarise


def test_recall_matches_source_and_section():
    assert recall_at_k([{"source": "a", "section": "1"}],
                       [{"source": "A", "section": "1"}], 5) == 1


def test_citation_validity_rejects_unretrieved_source():
    assert citation_validity({"citations": [{"source": "bad", "section": "1"}],
                              "retrieved": []}) == 0


def test_summary_counts_must_abstain_and_type_recall():
    row = {"type": "abstain", "must_abstain": True, "relevant": [],
           "response": {"answered_from_context": False, "citations": [], "retrieved": []}}
    result = summarise([row])
    assert result["abstention_correctness"] == 1
    assert result["recall_at_5_by_type"]["abstain"] is None


def test_gate_checks_abstention():
    assert gate({"citation_validity": 1, "must_abstain_questions": 1,
                 "abstention_correctness": 0, "recall_at_5": 1}, None)
