import review_server


def test_candidate_style_highlights_current_candidate_regardless_of_decision():
    color, thickness = review_server._candidate_style(is_highlighted=True, decision="reject")
    assert (color, thickness) == ((0, 0, 255), 8)


def test_candidate_style_dims_rejected_candidates():
    color, thickness = review_server._candidate_style(is_highlighted=False, decision="reject")
    assert (color, thickness) == ((150, 150, 150), 2)


def test_candidate_style_defaults_to_pending_color():
    color, thickness = review_server._candidate_style(is_highlighted=False, decision=None)
    assert (color, thickness) == ((255, 200, 0), 2)
