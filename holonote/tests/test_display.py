import holoviews as hv

hv.extension("bokeh")


class TestPoint2D:
    def test_get_indices_by_position_exact(self, annotator_point2d):
        x, y = 0.5, 0.3
        description = "A test annotation!"
        annotator_point2d.set_regions(x=x, y=y)
        annotator_point2d.add_annotation(description=description)
        display = annotator_point2d.get_display("x", "y")
        indices = display.get_indices_by_position(x=x, y=y)
        assert len(indices) == 1

    def test_get_indices_by_position_nearest_2d_point_threshold(self, annotator_point2d):
        x, y = 0.5, 0.3
        annotator_point2d.get_display("x", "y").nearest_2d_point_threshold = 1
        description = "A test annotation!"
        annotator_point2d.set_regions(x=x, y=y)
        annotator_point2d.add_annotation(description=description)
        display = annotator_point2d.get_display("x", "y")
        indices = display.get_indices_by_position(x=x + 1.5, y=y + 1.5)
        assert len(indices) == 0

        display.nearest_2d_point_threshold = 5
        indices = display.get_indices_by_position(x=x + 0.5, y=y + 0.5)
        assert len(indices) == 1

    def test_get_indices_by_position_nearest(self, annotator_point2d):
        x, y = 0.5, 0.3
        description = "A test annotation!"
        annotator_point2d.set_regions(x=x, y=y)
        annotator_point2d.add_annotation(description=description)
        display = annotator_point2d.get_display("x", "y")
        indices = display.get_indices_by_position(x=x + 1.5, y=y + 1.5)
        assert len(indices) == 1

        display.nearest_2d_point_threshold = 5
        indices = display.get_indices_by_position(x=x + 0.5, y=y + 0.5)
        assert len(indices) == 1

    def test_get_indices_by_position_empty(self, annotator_point2d):
        display = annotator_point2d.get_display("x", "y")
        indices = display.get_indices_by_position(x=0.5, y=0.3)
        assert len(indices) == 0

    def test_get_indices_by_position_no_position(self, annotator_point2d):
        display = annotator_point2d.get_display("x", "y")
        indices = display.get_indices_by_position(x=None, y=None)
        assert len(indices) == 0

    def test_get_indices_by_position_multi_choice(self, annotator_point2d):
        x, y = 0.5, 0.3
        description = "A test annotation!"
        annotator_point2d.set_regions(x=x, y=y)
        annotator_point2d.add_annotation(description=description)

        x2, y2 = 0.51, 0.31
        description = "A test annotation!"
        annotator_point2d.set_regions(x=x2, y=y2)
        annotator_point2d.add_annotation(description=description)

        display = annotator_point2d.get_display("x", "y")
        display.nearest_2d_point_threshold = 1000

        indices = display.get_indices_by_position(x=x, y=y)
        assert len(indices) == 1
