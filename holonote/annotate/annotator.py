from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import holoviews as hv
import pandas as pd
import param
from bokeh.models.tools import BoxSelectTool, HoverTool, Tool

from .._warnings import warn
from .connector import Connector, SQLiteDB
from .table import AnnotationTable

if TYPE_CHECKING:
    from .typing import SpecDict


class Style(param.Parameterized):
    alpha = param.Number(default=0.2, bounds=(0, 1), allow_refs=True)
    alpha.doc = "Alpha value for non-selected regions"

    highlight = param.Number(default=0.7, bounds=(0, 1), allow_refs=True)
    highlight.doc = "Alpha value for selected regions"

    groupby = param.Selector(default=None, doc="Groupby dimension", allow_refs=True)
    visible = param.ListSelector(
        default=[], doc="Visible dimensions, needs groupby enabled", allow_refs=True
    )

    color = param.Parameter(default="red", doc="Color of the indicator", allow_refs=True)

    range_style = {"apply_ranges": False, "show_legend": False}
    point_style = {"apply_ranges": False, "show_legend": False}

    edit_range_style = {"line_alpha": 1, "line_width": 1, "line_color": "black"}
    edit_point_style = {"line_alpha": 1, "line_color": "black"}

    @property
    def indicator_highlight(self) -> dict[str, tuple[float, float]]:
        return {"alpha": (self.highlight, self.alpha)}

    def indicator(self, range_style, point_style, highlighters):
        return (
            hv.opts.Rectangles(**dict(range_style, color=self.color, **highlighters)),
            hv.opts.VSpans(**dict(range_style, color=self.color, **highlighters)),
            hv.opts.HSpans(**dict(range_style, color=self.color, **highlighters)),
            hv.opts.VLines(**dict(range_style, color=self.color, **highlighters)),
            hv.opts.HLines(**dict(range_style, color=self.color, **highlighters)),
        )

    def region(self, edit_range_style, edit_point_style):
        return (
            hv.opts.Rectangles(**edit_range_style),
            hv.opts.VSpans(**edit_range_style),
            hv.opts.HSpans(**edit_range_style),
            hv.opts.VLines(**edit_range_style),
            hv.opts.HLines(**edit_range_style),
        )


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
        vdims = [*fields_labels, data.index.name]
        element = hv.VLines(data.reset_index(), kdims=region_labels, vdims=vdims)
        hover = cls._build_hover_tool(data)
        return element.opts(tools=[hover])

    @classmethod
    def points_2d(
        cls, data, region_labels, fields_labels, invert_axes=False, groupby: str | None = None
    ):
        "Vectorizes point regions to VLines * HLines. Note does not support hover info"
        msg = "2D point regions not supported yet"
        raise NotImplementedError(msg)
        vdims = [*fields_labels, data.index.name]
        element = hv.Points(data.reset_index(), kdims=region_labels, vdims=vdims)
        hover = cls._build_hover_tool(data)
        return element.opts(tools=[hover])

    @classmethod
    def ranges_2d(
        cls, data, region_labels, fields_labels, invert_axes=False, groupby: str | None = None
    ):
        "Vectorizes an nd-overlay of range_2d rectangles."
        kdims = [region_labels[i] for i in (0, 2, 1, 3)]  # LBRT format
        vdims = [*fields_labels, data.index.name]
        element = hv.Rectangles(data.reset_index(), kdims=kdims, vdims=vdims)
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
        vdims = [*fields_labels, data.index.name]
        ds = hv.Dataset(data.reset_index(), kdims=region_labels, vdims=vdims)
        element = ds.to(hv.VSpans, groupby=groupby)
        hover = cls._build_hover_tool(data)
        return element.opts(tools=[hover])

    @classmethod
    def _build_hover_tool(self, data, cds_map=None) -> HoverTool:
        if cds_map is None:
            cds_map = {}
        tooltips, formatters = [], {}
        for dim in data.columns:
            cds_name = cds_map.get(dim, dim)
            if data[dim].dtype.kind == "M":
                tooltips.append((dim, f"@{{{cds_name}}}{{%F}}"))
                formatters[f"@{{{cds_name}}}"] = "datetime"
            else:
                tooltips.append((dim, f"@{{{cds_name}}}"))
        return HoverTool(tooltips=tooltips, formatters=formatters)


class AnnotatorInterface(param.Parameterized):
    """
    Baseclass that expresses the Python interface of an Annotator
    without using any holoviews components and without requiring
    display. Most of this API centers around how the Annotator interacts
    with the AnnotationTable.
    """

    selected_indices = param.List(default=[], doc="Indices of selected annotations")

    spec = param.Dict(default={}, doc="Specification of annotation types")

    fields = param.List(default=["description"], doc="List of fields", constant=True)

    static_fields = param.Dict(
        default={},
        constant=True,
        doc="Dictionary with key and value which will be added to each commit",
    )

    connector = param.ClassSelector(class_=Connector, allow_None=False)

    connector_class = SQLiteDB

    def __init__(self, spec, **params):
        if "connector" not in params:
            params["connector"] = self.connector_class()

        spec = self.normalize_spec(spec)

        super().__init__(spec=spec, **params)
        if set(self.fields) & set(self.static_fields):
            msg = "The values of fields and static_fields must not overlap"
            raise ValueError(msg)
        if self.connector.fields is None:
            self.connector.fields = self.all_fields
        self._region = {}
        self._last_region = None

        self.annotation_table = AnnotationTable()
        self.connector._create_column_schema(self.spec, self.all_fields)
        self.connector._initialize(self.connector.column_schema)
        self.annotation_table.load(self.connector, fields=self.connector.fields, spec=self.spec)

    @property
    def all_fields(self) -> list:
        """Return a list of all fields including static fields"""
        return [*self.fields, *self.static_fields]

    @classmethod
    def normalize_spec(self, input_spec: dict[str, Any]) -> SpecDict:
        """Normalize the spec to conform to SpecDict format

        Accepted input spec formats:
        spec = {
            # Range (two values)
            "A1": (np.float64, "range"),
            "A2": {"type": np.float64, "region": "range"},
            "A3": np.float64,  # Special case
            # Point
            "B1": (np.float64, "point"),
            "B2": {"type": np.float64, "region": "point"},
            # Geometry
            ("C1", "D1"): {"type": np.float64, "region": "geometry"},
            ("C2", "D2"): (np.float64, "geometry"),
        }
        """

        new_spec: SpecDict = {}
        for k, v in input_spec.items():
            if isinstance(v, dict):
                pass
            elif isinstance(v, tuple):
                v = {"type": v[0], "region": v[1]}
            else:
                v = {"type": v, "region": "range"}

            if v["region"] not in ["range", "point", "geometry"]:
                msg = "Region type must be range, point, or geometry."
                raise ValueError(msg)
            if v["region"] == "geometry" and not isinstance(k, tuple):
                msg = "Geometry region dimension must be a tuple."
                raise ValueError(msg)
            new_spec[k] = v

        return new_spec

    @property
    def df(self) -> pd.DataFrame:
        return self.annotation_table.get_dataframe(spec=self.spec)

    def get_dataframe(self, dims) -> pd.DataFrame:
        return self.annotation_table.get_dataframe(spec=self.spec, dims=dims)

    def refresh(self, clear=False):
        "Method to update display state of the annotator and optionally clear stale visual state"

    # Selecting annotations

    def select_by_index(self, *inds):
        "Set the selection state by the indices i.e. primary key values"
        self.selected_indices = list(inds)

    @property
    def selected_index(self):
        "Convenience property returning a single selected index (the first one) or None"
        return self.selected_indices[0] if len(self.selected_indices) > 0 else None

    @property
    def region(self):
        return self._region

    def set_regions(self, **items):
        self._set_regions(**items)

    def _set_regions(self, **items):
        """Updating regions"""
        # TODO: Validate values based on spec
        for dim, values in items.items():
            if dim not in self.spec:
                msg = f"Dimension {dim} not in spec"
                raise ValueError(msg)
            self._region[dim] = values

    def clear_regions(self):
        self._region = {}
        self._last_region = {}

    def _add_annotation(self, **fields):
        # Primary key specification is optional
        if self.connector.primary_key.field_name not in fields:
            index_val = self.connector.primary_key(
                self.connector, list(self.annotation_table._field_df.index)
            )
            fields[self.connector.primary_key.field_name] = index_val

        # Don't do anything if self.region is an empty dict
        if self.region and self.region != self._last_region:
            self.annotation_table.add_annotation(
                self._region, spec=self.spec, **fields, **self.static_fields
            )
            self._last_region = self._region.copy()

    def add_annotation(self, **fields):
        self._add_annotation(**fields)

    def update_annotation_region(self, index):
        self.annotation_table.update_annotation_region(self._region, index)

    def update_annotation_fields(self, index, **fields):
        self.annotation_table.update_annotation_fields(index, **fields, **self.static_fields)

    def delete_annotation(self, index):
        try:
            self.annotation_table.delete_annotation(index)
        except KeyError:
            msg = f"Annotation with index {index!r} does not exist."
            raise ValueError(msg) from None

        if index in self.selected_indices:
            self.selected_indices.remove(index)

    # Defining initial annotations

    def define_annotations(self, data: pd.DataFrame, **kwargs) -> None:
        # Will both set regions and add annotations. Can accept multiple inputs
        # if index is none a new index will be set.
        # if nothing is given it infer the regions and fields from the column header.
        # annotator.define_annotations(
        #     df, TIME=("start", "end"), description="description", index=None
        # )

        # kwargs = dict(A=("start", "end"), description="description")
        # kwargs = dict(A=("start", "end"))
        # kwargs = dict(A=("start", "end"), index=True)

        index = kwargs.pop("index", False)
        field_data = set(self.fields) & set(data.columns)
        f_keys = (set(self.fields) & kwargs.keys()) | field_data
        r_keys = (kwargs.keys() - f_keys) | (set(self.spec) & set(data.columns))
        pk = self.connector.primary_key

        for k, v in kwargs.items():
            if k == v:
                continue
            if v in field_data:
                msg = (
                    f"Input {v!r} has overlapping name with a field or spec. "
                    "This can give weird behavior. Consider renaming the input."
                )
                warn(msg)
        # assert len(set(data.columns) - f_keys - r_keys) == 0

        # Vectorize the conversion?
        for r in data.itertuples(index=index):
            regions = {
                k: tuple([getattr(r, a) for a in kwargs[k]])
                if isinstance(kwargs.get(k, k), tuple)
                else getattr(r, kwargs.get(k, k))
                for k in r_keys
            }
            fields = {k: getattr(r, kwargs.get(k, k)) for k in f_keys}
            if index:
                fields[pk.field_name] = pk.cast(r.Index)

            self._set_regions(**regions)
            self._add_annotation(**fields)

    # Snapshotting and reverting
    @property
    def has_snapshot(self) -> bool:
        return self.annotation_table.has_snapshot

    def revert_to_snapshot(self) -> None:
        self.annotation_table.revert_to_snapshot()

    def snapshot(self) -> None:
        self.annotation_table.snapshot()

    def commit(self, return_commits=False):
        # self.annotation_table.initialize_table(self.connector)  # Only if not in params
        commits = self.annotation_table.commits(self.connector)
        if return_commits:
            return commits


class AnnotationDisplay(param.Parameterized):
    kdims = param.List(
        default=["x"], bounds=(1, 3), constant=True, doc="Dimensions of the element"
    )

    indicator = Indicator

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
        self._selected_values = []
        self._selected_options = []

        transient = False
        self._edit_streams = [
            hv.streams.BoundsXY(transient=transient),
            hv.streams.SingleTap(transient=transient),
            hv.streams.Lasso(transient=transient),
        ]

        self.annotator = weakref.proxy(annotator)
        self.style = weakref.proxy(annotator.style)
        self._set_region_types()
        self._element = self._make_empty_element()

    def _set_region_types(self) -> None:
        self.region_types = "-".join([self.annotator.spec[k]["region"] for k in self.kdims])

    @property
    def element(self):
        return self.overlay()

    @property
    def edit_tools(self) -> list[Tool]:
        tools = []
        if self.region_types == "range":
            tools.append(BoxSelectTool(dimensions="width"))
        elif self.region_types == "range-range":
            tools.append(BoxSelectTool())
        elif self.region_types == "point":
            tools.append(BoxSelectTool(dimensions="width"))
        elif self.region_types == "point-point":
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
        if self.region_types == "point" and bounds:
            x = (bounds[0] + bounds[2]) / 2
            y = None
            bounds = (x, 0, x, 0)
        elif "range" not in self.region_types:
            bounds = None

        # If selection enabled, tap stream used for selection not for creating point regions
        # if ('point' in self.region_types and self.selection_enabled) or 'point' not in self.region_types:
        if "point" not in self.region_types:
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
            if self.region_types == "point" and x is not None:
                self.annotator._set_regions(**{kdims[0]: x})
            if None not in [x, y]:
                if len(kdims) == 1:
                    self.annotator._set_regions(**{kdims[0]: x})
                elif len(kdims) == 2:
                    self.annotator._set_regions(**{kdims[0]: x, kdims[1]: y})
                else:
                    msg = "Only 1d and 2d supported for Points"
                    raise ValueError(msg)

            return region_element

        return hv.DynamicMap(inner, streams=self._edit_streams)

    def region_editor(self) -> hv.DynamicMap:
        if not hasattr(self, "_region_editor"):
            self._region_editor = self._make_selection_editor()
        return self._region_editor

    def _get_range_indices_by_position(self, **inputs) -> list[Any]:
        if isinstance(self.static_indicators, hv.NdOverlay):
            df = pd.concat([el.data for el in self.static_indicators.values()])
        else:
            df = self.static_indicators.data

        if df.empty:
            return []

        # Because we reset_index in Indicators
        id_col = df.columns[0]

        for i, (k, v) in enumerate(inputs.items()):
            mask = (df[f"start[{k}]"] <= v) & (v < df[f"end[{k}]"])
            if i == 0:
                ids = set(df[mask][id_col])
            else:
                ids &= set(df[mask][id_col])
        return list(ids)

    def _get_point_indices_by_position(self, **inputs) -> list[Any]:
        """
        Simple algorithm for finding the closest point
        annotation to the given position.
        """

        df = self.static_indicators.data
        if df.empty:
            return []

        # Because we reset_index in Indicators
        id_col = df.columns[0]

        for i, (k, v) in enumerate(inputs.items()):
            nearest = (df[f"point[{k}]"] - v).abs().argmin()
            if i == 0:
                ids = {df.iloc[nearest][id_col]}
            else:
                ids &= {df.iloc[nearest][id_col]}
        return list(ids)

    def get_indices_by_position(self, **inputs) -> list[Any]:
        "Return primary key values matching given position in data space"
        if "range" in self.region_types:
            return self._get_range_indices_by_position(**inputs)
        elif "point" in self.region_types:
            return self._get_point_indices_by_position(**inputs)
        else:
            msg = f"{self.region_types} not implemented"
            raise NotImplementedError(msg)

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
        self.register_tap_selector(self._element)
        self.register_double_tap_clear(self._element)

        def inner(_count):
            return self.static_indicators

        return hv.DynamicMap(inner, streams=[self._annotation_count_stream])

    def overlay(self, indicators=True, editor=True) -> hv.Overlay:
        edit_range_style = self.style.edit_range_style
        edit_point_style = self.style.edit_point_style
        region = self.style.region(edit_range_style, edit_point_style)

        layers = []
        active_tools = []
        if "range" in self.region_types or self.region_types == "point":
            active_tools += ["box_select"]
        elif self.region_types == "point-point":
            active_tools += ["tap"]
        layers.append(self._element.opts(tools=self.edit_tools, active_tools=active_tools))

        if indicators:
            layers.append(self.indicators())
        if editor:
            layers.append(self.region_editor().opts(*region))
        return hv.Overlay(layers).collate()

    @property
    def static_indicators(self):
        data = self.annotator.get_dataframe(dims=self.kdims)
        fields_labels = self.annotator.all_fields
        region_labels = [k for k in data.columns if k not in fields_labels]

        indicator_kwargs = {
            "data": data,
            "region_labels": region_labels,
            "fields_labels": fields_labels,
            "invert_axes": False,  # Not yet handled
            "groupby": self.style.groupby,
        }

        if self.region_types == "range":
            indicator = Indicator.ranges_1d(**indicator_kwargs)
        elif self.region_types == "range-range":
            indicator = Indicator.ranges_2d(**indicator_kwargs)
        elif self.region_types == "point":
            indicator = Indicator.points_1d(**indicator_kwargs)
        elif self.region_types == "point-point":
            indicator = Indicator.points_2d(**indicator_kwargs)
        else:
            msg = f"{self.region_types} not implemented"
            raise NotImplementedError(msg)

        if self.style.groupby and self.style.visible:
            indicator = indicator.get(self.style.visible)
            # TODO: Handle when indicator is empty

        # Set styling on annotations indicator
        range_style = self.style.range_style
        point_style = self.style.point_style
        highlight = self.style.indicator_highlight

        highlighters = {opt: self.selected_dim_expr(v[0], v[1]) for opt, v in highlight.items()}
        indicator = indicator.opts(*self.style.indicator(range_style, point_style, highlighters))

        return indicator.overlay() if self.style.groupby else hv.NdOverlay({0: indicator})

    def selected_dim_expr(self, selected_value, non_selected_value):
        self._selected_values.append(selected_value)
        self._selected_options.append({i: selected_value for i in self.annotator.selected_indices})
        index_name = (
            "id"
            if self.annotator.annotation_table._field_df.index.name is None
            else self.annotator.annotation_table._field_df.index.name
        )
        return hv.dim(index_name).categorize(
            self._selected_options[-1], default=non_selected_value
        )

    @property
    def dim_expr(self):
        return self._selection_info["dim_expr"]

    def show_region(self):
        kdims = list(self.kdims)
        region = {k: v for k, v in self.annotator._region.items() if k in self.kdims}

        if not region:
            return

        if self.region_types == "range":
            value = region[kdims[0]]
            bounds = (value[0], 0, value[1], 1)
        elif self.region_types == "range-range":
            bounds = (
                region[kdims[0]][0],
                region[kdims[1]][0],
                region[kdims[0]][1],
                region[kdims[1]][1],
            )
        elif self.region_types == "point":
            value = region[kdims[0]]
            bounds = (value, 0, value, 1)
        else:
            bounds = False

        if bounds:
            self._edit_streams[0].event(bounds=bounds)


class Annotator(AnnotatorInterface):
    """
    An annotator displays the contents of an AnnotationTable and
    provides the means to manipulate view the corresponding contents,
    add new annotations and update existing annotations.
    """

    style = param.ClassSelector(default=Style(), class_=Style, doc="Style parameters")

    def __init__(self, spec: dict, **params):
        """
        The spec argument must be an element or a dictionary of kdim dtypes
        """

        self._displays = {}

        super().__init__(
            spec if isinstance(spec, dict) else self._infer_kdim_dtypes(spec),
            **params,
        )

        self._selection_enabled = True
        self._editable_enabled = True

        self.style.param.groupby.objects = [*self.fields, None]  # TODO: Smarter way?

    @classmethod
    def _infer_kdim_dtypes(self, element: hv.Element) -> dict:
        # Remove?
        return AnnotationDisplay._infer_kdim_dtypes(element)

    def _create_annotation_element(self, element_key: tuple[str, ...]) -> AnnotationDisplay:
        for key in element_key:
            if key not in self.spec:
                msg = f"Dimension {key!r} not in spec"
                raise ValueError(msg)
        return AnnotationDisplay(self, kdims=list(element_key))

    def get_element(self, *kdims: str | hv.Dimension) -> hv.DynamicMap:
        return self.get_display(*kdims).element

    def get_display(self, *kdims: str | hv.Dimension) -> AnnotationDisplay:
        element_key = tuple(map(str, kdims))
        if element_key not in self._displays:
            self._displays[element_key] = self._create_annotation_element(element_key)
        return self._displays[element_key]

    def __mul__(self, other: hv.Element) -> hv.Overlay:
        return other * self.get_element(*other.kdims)

    def __rmul__(self, other: hv.Element) -> hv.Overlay:
        return self.__mul__(other)

    def refresh(self, clear=False) -> None:
        for v in self._displays.values():
            hv.streams.Stream.trigger([v._annotation_count_stream])
            if clear:
                v.clear_indicated_region()
            v.show_region()

    # all the style parameters?
    @param.depends(
        "style.groupby",
        "style.visible",
        "style.color",
        "style.alpha",
        "style.highlight",
        watch=True,
    )
    def _refresh_style(self):
        self.refresh()

    def set_annotation_table(self, annotation_table):
        self.select_by_index()
        for v in self._displays.values():
            v.select_by_index()
        self.clear_regions()
        super().set_annotation_table(annotation_table)
        self.refresh(clear=True)

    def clear_edits(self):
        super().clear_edits()
        self.clear_indicated_region()

    def add_annotation(self, **fields):
        super().add_annotation(**fields)
        self.refresh(clear=True)

    def update_annotation_region(self, index):
        super().update_annotation_region(index)
        self.refresh()

    def update_annotation_fields(self, index, **fields):
        super().update_annotation_fields(index, **fields)
        self.refresh()

    def delete_annotation(self, index):
        super().delete_annotation(index)
        self.refresh()

    def delete_annotations(self, *indices):
        if not indices:
            msg = "At least one index must be specified to delete annotations"
            raise ValueError(msg)
        for index in indices:
            super().delete_annotation(index)
        self.refresh()

    def select_by_index(self, *inds):
        "Set the selection state by the indices i.e. primary key values"

        for v in self._displays.values():
            if not v.selection_enabled:
                inds = ()

            for d, val in zip(v._selected_options, v._selected_values):
                d.clear()
                for ind in inds:
                    d[ind] = val
        super().select_by_index(*inds)
        self.refresh()

    def define_annotations(self, data: pd.DataFrame, **kwargs) -> None:
        super().define_annotations(data, **kwargs)
        self.refresh(clear=True)

    def revert_to_snapshot(self):
        super().revert_to_snapshot()
        self.refresh()

    def set_regions(self, **items):
        super().set_regions(**items)
        self.refresh()

    @property
    def selection_enabled(self) -> bool:
        return self._selection_enabled

    @selection_enabled.setter
    def selection_enabled(self, enabled: bool) -> None:
        for v in self._displays.values():
            v.selection_enabled = enabled

        if not enabled:
            self.select_by_index()

    @property
    def editable_enabled(self) -> bool:
        return self._editable_enabled

    @editable_enabled.setter
    def editable_enabled(self, enabled: bool) -> None:
        for v in self._displays.values():
            v.editable_enabled = enabled
