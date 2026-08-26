from windshield import WindshieldBlob
from candidate import point_in_polygon, find_slot_windshield


def test_point_in_polygon_true_and_false():
    polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]

    assert point_in_polygon((50.0, 50.0), polygon) is True
    assert point_in_polygon((150.0, 50.0), polygon) is False


def test_find_slot_windshield_picks_blob_inside_polygon_only():
    polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    inside_blob = WindshieldBlob(contour=None, bbox=(40, 40, 10, 10), centroid=(45.0, 45.0), area=100)
    outside_blob = WindshieldBlob(contour=None, bbox=(200, 200, 10, 10), centroid=(205.0, 205.0), area=100)

    assert find_slot_windshield(polygon, [outside_blob, inside_blob]) is inside_blob
    assert find_slot_windshield(polygon, [outside_blob]) is None
