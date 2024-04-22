from __future__ import annotations

import weakref
from functools import cache, reduce
from typing import TYPE_CHECKING, Any

import colorcet as cc
import holoviews as hv
import numpy as np
import param
from bokeh.models.tools import BoxSelectTool, HoverTool, Tool

from .._warnings import warn

if TYPE_CHECKING:
    from .annotator import Annotator


class _StyleOpts(param.Dict):
    def _validate(self, val) -> None:
        super()._validate(val)
        for k in val:
            if k in ("color", "alpha"):
                warn("Color and alpha opts should be set directly on the style object.")


_default_opts = {"apply_ranges": False, "show_legend": False}

# Make red the first color
_default_color = cc.palette["glasbey_category10"].copy()
_default_color[:4] = [_default_color[3], *_default_color[:3]]


@cache
def _valid_element_opts():
    if not hv.extension._loaded:
        hv.extension("bokeh")
    return hv.opts._element_keywords("bokeh")


class Style(param.Parameterized):
    """
    Style class for controlling the appearance of the annotations
    indicator and editor.

    This can be accessed as an accessor on an annotator object,
    the following will set the annotation color to red:

    >>> from holonote.annotate import Annotator
    >>> annotator = Annotator(...)
    >>> annotator.style.color = "red"

    This will update existing annotation displays and any new
    displays with the new style.

    The style object can also be used to control the appearance
    of the editor and selected indicator:

    >>> annotator.style.edit_color = "blue"
    >>> annotator.style.edit_alpha = 0.5
    >>> annotator.style.selection_color = "green"
    >>> annotator.style.selection_alpha = 0.5

    See the [styling notebook](../../examples/styling.ipynb) for more examples
    of how to use the style object.
    """

    alpha = param.Number(
        default=0.2, bounds=(0, 1), allow_refs=True, doc="Alpha value for non-selected regions"
    )

    selection_alpha = param.Number(
        default=0.7, bounds=(0, 1), allow_refs=True, doc="Alpha value for selected regions"
    )

    edit_alpha = param.Number(
        default=0.4, bounds=(0, 1), allow_refs=True, doc="Alpha value for editing regions"
    )

    color = param.Parameter(
        default=hv.Cycle(_default_color), doc="Color of the indicator", allow_refs=True
    )
    edit_color = param.Parameter(default="blue", doc="Color of the editor", allow_refs=True)
    selection_color = param.Parameter(
        default=None, doc="Color of selection, by the default the same as color", allow_refs=True
    )

    # Indicator opts (default and selection)
    opts = _StyleOpts(default={})
    line_opts = _StyleOpts(default={})
    span_opts = _StyleOpts(default={})
    rectangle_opts = _StyleOpts(default={})

    # Editor opts
    edit_opts = _StyleOpts(default={"line_color": "black"})
    edit_line_opts = _StyleOpts(default={})
    edit_span_opts = _StyleOpts(default={})
    edit_rectangle_opts = _StyleOpts(default={})

    @property
    def _indicator_selection(self) -> dict[str, tuple]:
        select = {"alpha": (self.selection_alpha, self.alpha)}
        if self.selection_color is not None:
            if isinstance(self.color, hv.dim):
                msg = "'Style.color' cannot be a `hv.dim` when 'Style.selection_color' is not None"
                raise ValueError(msg)
            else:
                select["color"] = (self.selection_color, self.color)
        return select

    def indicator(self, **select_opts) -> tuple[hv.Options, ...]:
        opts = {**_default_opts, "color": self.color, **select_opts, **self.opts}
        return (
            hv.opts.Rectangles(**opts, **self.rectangle_opts),
            hv.opts.VSpans(**opts, **self.span_opts),
            hv.opts.HSpans(**opts, **self.span_opts),
            hv.opts.VLines(**opts, **self.line_opts),
            hv.opts.HLines(**opts, **self.line_opts),
        )

    def editor(self) -> tuple[hv.Options, ...]:
        opts = {
            **_default_opts,
            "alpha": self.edit_alpha,
            "color": self.edit_color,
            **self.edit_opts,
        }
        return (
            hv.opts.Rectangles(**opts, **self.edit_rectangle_opts),
            hv.opts.VSpan(**opts, **self.edit_span_opts),
            hv.opts.HSpan(**opts, **self.edit_span_opts),
            hv.opts.VLine(**opts, **self.edit_line_opts),
            hv.opts.HLine(**opts, **self.edit_line_opts),
        )

    def reset(self) -> None:
        params = self.param.objects().items()
        self.param.update(**{k: v.default for k, v in params if k != "name"})


class Indicator:
    """
    Collection of class methods that express annotation data as final
    displayed (vectorized) HoloViews object.
    """

    @classmethod
    def points_1d(
        cls, data, region_labels, fields_labels, invert_axes=False, groupby: str | None = None
    ):
        "Vectorizes point regions to VLines. Note does not support hover info"
        vdims = [*fields_labels, "__selected__"]
        element = hv.VLines(data, kdims=region_labels, vdims=vdims)
        hover = cls._build_hover_tool(data)
        return element.opts(tools=[hover])

    @classmethod
    def points_2d(
        cls, data, region_labels, fields_labels, invert_axes=False, groupby: str | None = None
    ):
        "Vectorizes point regions to VLines * HLines. Note does not support hover info"
        msg = "2D point regions not supported yet"
        raise NotImplementedError(msg)
        vdims = [*fields_labels, "__selected__"]
        element = hv.Points(data, kdims=region_labels, vdims=vdims)
        hover = cls._build_hover_tool(data)
        return element.opts(tools=[hover])

    @classmethod
    def ranges_2d(
        cls, data, region_labels, fields_labels, invert_axes=False, groupby: str | None = None
    ):
        "Vectorizes an nd-overlay of range_2d rectangles."
        kdims = [region_labels[i] for i in (0, 2, 1, 3)]  # LBRT format
        vdims = [*fields_labels, "__selected__"]
        ds = hv.Dataset(data, kdims=kdims, vdims=vdims)
        element = ds.to(hv.Rectangles, groupby=groupby)
        cds_map = dict(zip(region_labels, ("left", "right", "bottom", "top")))
        hover = cls._build_hover_tool(data, cds_map)
        return element.opts(tools=[hover])

    @classmethod
    def ranges_1d(
        cls, data, region_labels, fields_labels, invert_axes=False, groupby: str | None = None
    ):
        """
        Vectorizes an nd-overlay of range_1d rectangles.

        NOTE: Should use VSpans once available!
        """
        vdims = [*fields_labels, "__selected__"]
        ds = hv.Dataset(data, kdims=region_labels, vdims=vdims)
        element = ds.to(hv.VSpans, groupby=groupby)
        hover = cls._build_hover_tool(data)
        return element.opts(tools=[hover])

    @classmethod
    def _build_hover_tool(self, data, cds_map=None) -> HoverTool:
        if cds_map is None:
            cds_map = {}
        tooltips, formatters = [], {}
        for dim in data.columns:
            if dim == "__selected__":
                continue
            cds_name = cds_map.get(dim, dim)
            if data[dim].dtype.kind == "M":
                tooltips.append((dim, f"@{{{cds_name}}}{{%F}}"))
                formatters[f"@{{{cds_name}}}"] = "datetime"
            else:
                tooltips.append((dim, f"@{{{cds_name}}}"))
        return HoverTool(tooltips=tooltips, formatters=formatters)


class AnnotationDisplay(param.Parameterized):
    kdims = param.List(
        default=["x"], bounds=(1, 3), constant=True, doc="Dimensions of the element"
    )

    data = param.DataFrame(doc="Combined dataframe of annotation data", constant=True)

    _count = param.Integer(default=0, precedence=-1)

    def __init__(self, annotator: Annotator, **params) -> None:
        super().__init__(**params)

        self._annotation_count_stream = hv.streams.Params(
            parameterized=self,
            parameters=["_count"],
            transient=True,
        )

        self._selection_info = {}

        self._selection_enabled = True
        self._editable_enabled = True

        transient = False
        self._edit_streams = [
            hv.streams.BoundsXY(transient=transient),
            hv.streams.SingleTap(transient=transient),
            hv.streams.Lasso(transient=transient),
        ]

        self.annotator = weakref.proxy(annotator)
        self.style = weakref.proxy(annotator.style)
        self._update_data()
        self._set_region_format()
        self._element = self._make_empty_element()

    def _set_region_format(self) -> None:
        self.region_format = "-".join([self.annotator.spec[k]["region"] for k in self.kdims])

    def _update_data(self):
        with param.edit_constant(self):
            self.data = self.annotator.get_dataframe(dims=self.kdims)

    @property
    def element(self):
        return self.overlay()

    @property
    def edit_tools(self) -> list[Tool]:
        tools = []
        if self.region_format == "range":
            tools.append(BoxSelectTool(dimensions="width"))
        elif self.region_format == "range-range":
            tools.append(BoxSelectTool())
        elif self.region_format == "point":
            tools.append(BoxSelectTool(dimensions="width"))
        elif self.region_format == "point-point":
            tools.append("tap")
        return tools

    @classmethod
    def _infer_kdim_dtypes(cls, element):
        if not isinstance(element, hv.Element):
            msg = "Supplied object {element} is not a bare HoloViews Element"
            raise ValueError(msg)
        kdim_dtypes = {}
        for kdim in element.dimensions(selection="key"):
            kdim_dtypes[str(kdim)] = type(element.dimension_values(kdim)[0])
        return kdim_dtypes

    def clear_indicated_region(self):
        "Clear any region currently indicated on the plot by the editor"
        self._edit_streams[0].event(bounds=None)
        self._edit_streams[1].event(x=None, y=None)
        self._edit_streams[2].event(geometry=None)
        self.annotator.clear_regions()

    def _make_empty_element(self) -> hv.Curve | hv.Image:
        El = hv.Curve if len(self.kdims) == 1 else hv.Image
        return El([], kdims=self.kdims).opts(apply_ranges=False)

    @property
    def selection_element(self) -> hv.Element:
        if not hasattr(self, "_selection_element"):
            self._selection_element = self._make_empty_element()
        return self._selection_element

    @property
    def selection_enabled(self) -> bool:
        return self._selection_enabled

    @selection_enabled.setter
    def selection_enabled(self, enabled: bool) -> None:
        self._selection_enabled = enabled

    @property
    def editable_enabled(self) -> bool:
        return self._editable_enabled

    @editable_enabled.setter
    def editable_enabled(self, enabled: bool) -> None:
        self._editable_enabled = enabled
        if not enabled:
            self.clear_indicated_region()

    def _filter_stream_values(self, bounds, x, y, geometry):
        if not self._editable_enabled:
            return (None, None, None, None)
        if self.region_format == "point" and bounds:
            x = (bounds[0] + bounds[2]) / 2
            y = None
            bounds = (x, 0, x, 0)
        elif "range" not in self.region_format:
            bounds = None

        # If selection enabled, tap stream used for selection not for creating point regions
        # if ('point' in self.region_format and self.selection_enabled) or 'point' not in self.region_format:
        if "point" not in self.region_format:
            x, y = None, None

        return bounds, x, y, geometry

    def _make_selection_editor(self) -> hv.DynamicMap:
        def inner(bounds, x, y, geometry):
            bounds, x, y, geometry = self._filter_stream_values(bounds, x, y, geometry)

            info = self.selection_element._get_selection_expr_for_stream_value(
                bounds=bounds, x=x, y=y, geometry=geometry
            )
            (dim_expr, bbox, region_element) = info

            self._selection_info = {
                "dim_expr": dim_expr,
                "bbox": bbox,
                "x": x,
                "y": y,
                "geometry": geometry,
                "region_element": region_element,
            }

            if bbox is not None:
                # self.annotator.set_regions will give recursion error
                self.annotator._set_regions(**bbox)

            kdims = list(self.kdims)
            if self.region_format == "point" and x is not None:
                self.annotator._set_regions(**{kdims[0]: x})
            if None not in [x, y]:
                if len(kdims) == 1:
                    self.annotator._set_regions(**{kdims[0]: x})
                elif len(kdims) == 2:
                    self.annotator._set_regions(**{kdims[0]: x, kdims[1]: y})
                else:
                    msg = "Only 1d and 2d supported for Points"
                    raise ValueError(msg)

            return region_element.opts(*self.style.editor())

        return hv.DynamicMap(inner, streams=self._edit_streams)

    def region_editor(self) -> hv.DynamicMap:
        if not hasattr(self, "_region_editor"):
            self._region_editor = self._make_selection_editor()
        return self._region_editor

    def get_indices_by_position(self, **inputs) -> list[Any]:
        "Return primary key values matching given position in data space"
        if self.annotator.groupby:
            df = self.data[self.data[self.annotator.groupby].isin(self.annotator.visible)]
        else:
            df = self.data

        if df.empty:
            return []

        if "range" in self.region_format:
            iter_mask = (
                (df[f"start[{k}]"] <= v) & (v < df[f"end[{k}]"]) for k, v in inputs.items()
            )
        elif "point" in self.region_format:
            iter_mask = ((df[f"point[{k}]"] - v).abs().argmin() for k, v in inputs.items())
        else:
            msg = f"{self.region_format} not implemented"
            raise NotImplementedError(msg)

        return list(df[reduce(np.logical_and, iter_mask)].index)

    def register_tap_selector(self, element: hv.Element) -> hv.Element:
        def tap_selector(x, y) -> None:  # Tap tool must be enabled on the element
            # Only select the first
            inputs = {str(k): v for k, v in zip(self.kdims, (x, y))}
            indices = self.get_indices_by_position(**inputs)
            if indices:
                self.annotator.select_by_index(indices[0])
            else:
                self.annotator.select_by_index()

        tap_stream = hv.streams.Tap(source=element, transient=True)
        tap_stream.add_subscriber(tap_selector)
        return element

    def register_double_tap_clear(self, element: hv.Element) -> hv.Element:
        def double_tap_clear(x, y):
            self.clear_indicated_region()

        double_tap_stream = hv.streams.DoubleTap(source=element, transient=True)
        double_tap_stream.add_subscriber(double_tap_clear)
        return element

    def indicators(self) -> hv.DynamicMap:
        if not hasattr(self, "_indicators"):
            self.register_tap_selector(self._element)
            self.register_double_tap_clear(self._element)

            self._indicators = hv.DynamicMap(
                self.static_indicators,
                streams=[self._annotation_count_stream, self.annotator.param.selected_indices],
            )
        return self._indicators

    def overlay(self, indicators=True, editor=True) -> hv.Overlay:
        layers = []
        active_tools = []
        if "range" in self.region_format or self.region_format == "point":
            active_tools += ["box_select"]
        elif self.region_format == "point-point":
            active_tools += ["tap"]
        layers.append(self._element.opts(tools=self.edit_tools, active_tools=active_tools))

        if indicators:
            layers.append(self.indicators())
        if editor:
            layers.append(self.region_editor())
        return hv.Overlay(layers).collate()

    def static_indicators(self, **events):
        fields_labels = self.annotator.all_fields
        region_labels = [k for k in self.data.columns if k not in fields_labels]

        self.data["__selected__"] = self.data.index.isin(self.annotator.selected_indices)

        indicator_kwargs = {
            "data": self.data,
            "region_labels": region_labels,
            "fields_labels": fields_labels,
            "invert_axes": False,  # Not yet handled
            "groupby": self.annotator.groupby,
        }

        if self.region_format == "range":
            indicator = Indicator.ranges_1d(**indicator_kwargs)
        elif self.region_format == "range-range":
            indicator = Indicator.ranges_2d(**indicator_kwargs)
        elif self.region_format == "point":
            indicator = Indicator.points_1d(**indicator_kwargs)
        elif self.region_format == "point-point":
            indicator = Indicator.points_2d(**indicator_kwargs)
        else:
            msg = f"{self.region_format} not implemented"
            raise NotImplementedError(msg)

        if self.annotator.groupby and self.annotator.visible:
            indicator = indicator.get(self.annotator.visible)
            if indicator is None:
                vis = "', '".join(self.annotator.visible)
                msg = f"Visible dimensions {vis!r} not in spec"
                raise ValueError(msg)

        # Set styling on annotations indicator
        highlight = self.style._indicator_selection
        highlighters = {opt: self._selected_dim_expr(v[0], v[1]) for opt, v in highlight.items()}
        indicator = indicator.opts(*self.style.indicator(**highlighters))

        return indicator.overlay() if self.annotator.groupby else hv.NdOverlay({0: indicator})

    def _selected_dim_expr(self, selected_value, non_selected_value) -> hv.dim:
        if isinstance(non_selected_value, hv.Cycle):
            non_selected_value = non_selected_value.values[0]
        return hv.dim("__selected__").categorize(
            {True: selected_value}, default=non_selected_value
        )

    @property
    def dim_expr(self):
        return self._selection_info["dim_expr"]

    def show_region(self):
        kdims = list(self.kdims)
        region = {k: v for k, v in self.annotator._region.items() if k in self.kdims}

        if not region:
            return

        if self.region_format == "range":
            value = region[kdims[0]]
            bounds = (value[0], 0, value[1], 1)
        elif self.region_format == "range-range":
            bounds = (
                region[kdims[0]][0],
                region[kdims[1]][0],
                region[kdims[0]][1],
                region[kdims[1]][1],
            )
        elif self.region_format == "point":
            value = region[kdims[0]]
            bounds = (value, 0, value, 1)
        else:
            bounds = False

        if bounds:
            self._edit_streams[0].event(bounds=bounds)
