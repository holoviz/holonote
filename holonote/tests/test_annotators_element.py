from __future__ import annotations

import holoviews as hv
import numpy as np

hv.extension("bokeh")


def get_data(
    annotation_element, annotations_type: str, element_type: hv.Element
)-> list[str | int | float] | None:
    hv.renderer("bokeh").get_plot(annotation_element)
    es = []
    for e in annotation_element.last.values():
        if not getattr(e, "_annotation_type", None) == annotations_type:
            continue
        if element_type is not None and not isinstance(e, element_type):
            continue
        es.append(e)

    if len(es) == 1:
        data = es[0].data
        if data.empty:
            return None
        else:
            return data.values.ravel().tolist()
    else:
        raise ValueError(
            f"Only as single type of {annotations_type!r} can be returned, not {len(es)}."
        )

def test_set_regions_rectangles(annotator_range2d, element_range2d) -> None:
    annotator = annotator_range2d
    element = element_range2d
    anno_el = annotator * element

    # No regions has been set.
    assert get_data(anno_el, "selector", hv.Rectangles) is None

    # Setting regions
    annotator.set_regions(x=(-0.25, 0.25), y=(-0.25, 0.25))
    output = get_data(anno_el, "selector", hv.Rectangles)
    expected = [-0.25, -0.25, 0.25, 0.25]
    np.testing.assert_allclose(output, expected)

    # Adding annotation and remove selection regions.
    annotator.add_annotation(description="Test")
    assert get_data(anno_el, "selector", hv.Rectangles) is None
    output = get_data(anno_el, "indicator", hv.Rectangles)
    expected = [-0.25, -0.25, 0.25, 0.25]
    np.testing.assert_allclose(output[:4], expected)
    assert output[4] == "Test"
