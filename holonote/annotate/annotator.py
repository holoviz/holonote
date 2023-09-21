from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import holoviews as hv
import pandas as pd
import param
from bokeh.models.tools import BoxSelectTool, HoverTool, Tool
from holoviews.core import datetime_types

from .connector import Connector, SQLiteDB
from .table import AnnotationTable

if TYPE_CHECKING:
    from .typing import SpecDict


class Indicator:
    """
    Collection of class methods that express annotation data as final
    displayed (vectorized) HoloViews object.
    """

    range_style = {'color': 'red', 'alpha': 0.4, 'apply_ranges': False}
    point_style = {}
    indicator_highlight = {'alpha':(0.7,0.2)}

    edit_range_style = {'alpha': 0.4, 'line_alpha': 1, 'line_width': 1, 'line_color': 'black'}
    edit_point_style = {}

    @classmethod
    def indicator_style(cls, range_style, point_style, highlighters):
        return (hv.opts.Rectangles(**dict(range_style, **highlighters)),
                hv.opts.VSpan(**dict(range_style, **highlighters)),
                hv.opts.HSpan(**dict(range_style, **highlighters)))


    @classmethod
    def region_style(cls, edit_range_style, edit_point_style):
        return (hv.opts.Rectangles(**edit_range_style),
                hv.opts.VSpan(**edit_range_style),
                hv.opts.HSpan(**edit_range_style))

    @classmethod
    def points_1d(cls, region_df, field_df, invert_axes=False):
        "Vectorizes point regions to VLines. Note does not support hover info"
        return hv.VLines(list(region_df["value"]))

    @classmethod
    def points_2d(cls, region_df, field_df, invert_axes=False):
        "Vectorizes point regions to VLines * HLines. Note does not support hover info"
        return (hv.VLines([el[0] for el in region_df["value"]])
              * hv.HLines([el[1] for el in region_df["value"]]))

    @classmethod
    def ranges_2d(cls, region_df, field_df, invert_axes=False):
        "Vectorizes an nd-overlay of range_2d rectangles."
        return cls._range_indicators(region_df, field_df, "2d", invert_axes, {})

    @classmethod
    def ranges_1d(cls, region_df, field_df, invert_axes=False, extra_params=None):
        """
        Vectorizes an nd-overlay of range_1d rectangles.

        NOTE: Should use VSpans once available!
        """
        if extra_params is None:
            msg = 'Extra parameters required until vectorized HSpans/VSpans supported'
            raise Exception(msg)
        return cls._range_indicators(region_df, field_df, "1d", invert_axes, extra_params)

    @classmethod
    def _range_indicators(cls, region_df, field_df, dimensionality, invert_axes=False, extra_params=None):
        # TODO: Clean this up VSpans/HSpans/VLines/HLines
        index_col_name = 'id' if field_df.index.name is None else field_df.index.name

        if region_df.empty:
            return hv.Rectangles([], vdims=[*field_df.columns, index_col_name])

        data = region_df.merge(field_df, left_on="_id", right_index=True)
        values = pd.DataFrame.from_records(data["value"])
        id_vals = data["_id"].rename({"_id": index_col_name})
        mdata_vals = data[field_df.columns]

        # TODO: Add check for None, (None, None), or (None, None, None, None) in values?

        if dimensionality=='1d':
            coords = values[[0, 0, 1, 1]].copy()
            coords.iloc[:, 1] = extra_params["rect_min"]
            coords.iloc[:, 3] = extra_params["rect_max"]
        else:
            coords = values[[0, 2, 1, 3]] # LBRT format

        rect_data = list(pd.concat([coords, mdata_vals, id_vals], axis=1).itertuples(index=False))
        return hv.Rectangles(rect_data, vdims=[*field_df.columns, index_col_name]) # kdims?


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

    connector = param.ClassSelector(class_=Connector, allow_None=False)

    connector_class = SQLiteDB

    def __init__(self, spec, **params):
        connector_kws = {'fields': params.get('fields')} if 'fields' in params else {}
        connector = params.pop('connector') if 'connector' in params else self.connector_class(**connector_kws)

        spec = self.normalize_spec(spec)

        super().__init__(spec=spec, connector=connector, **params)
        if connector.fields is None:
            connector.fields = self.fields
        self._region = {}
        self._last_region = None

        self.annotation_table = AnnotationTable()
        self.connector._create_column_schema(self.spec, self.fields)
        self.connector._initialize(self.connector.column_schema)
        self.annotation_table.load(self.connector, fields=self.connector.fields, spec=self.spec)

    @classmethod
    def normalize_spec(self, input_spec: dict[str, Any]) -> SpecDict:
        """ Normalize the spec to conform to SpecDict format

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
                msg = 'Region type must be range, point, or geometry.'
                raise ValueError(msg)
            if v["region"] == "geometry" and not isinstance(k, tuple):
                msg = 'Geometry region dimension must be a tuple.'
                raise ValueError(msg)
            new_spec[k] = v

        return new_spec

    @property
    def df(self):
        return self.annotation_table.dataframe

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

    def _get_range_indices_by_position(self, **inputs) -> list[Any]:
        df = self.annotation_table._region_df
        ranges = df[df['region']=='range']
        if ranges.empty:
            return []

        for i, (k, v) in enumerate(inputs.items()):
            dim = ranges[ranges["dim"] == k]
            mask = dim["value"].apply(lambda d: d[0] <= v < d[1])  # noqa: B023
            if i == 0:
                ids = set(dim[mask]._id)
            else:
                ids &= set(dim[mask]._id)
        return list(ids)

    def get_indices_by_position(self, **inputs) -> list[Any]:
        "Return primary key values matching given position in data space"
        # Lots TODO! 2 Dimensions, different annotation types etc.
        range_matches = self._get_range_indices_by_position(**inputs)
        event_matches = [] # TODO: Needs hit testing or similar for point events
        return range_matches + event_matches

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
                msg = f'Dimension {dim} not in spec'
                raise ValueError(msg)
            self._region[dim] = values

    def clear_regions(self):
        self._region = {}
        self._last_region = {}

    def set_range(self, startx, endx, starty=None, endy=None):
        print("set_range is legacy use set_regions instead")
        if None in [starty, endy] and ([starty, endy] != [None, None]):
            msg = 'Both starty and endy need to be non-None'
            raise ValueError(msg)

        value = (startx, endx) if starty is None else (startx, endx, starty, endy)
        kdims = list(self.spec)
        if len(value) == 2:
            if len(kdims) != 1:
                msg = 'Only one key dimension is allowed in spec.'
                raise ValueError(msg)
            if (r := self.spec[kdims[0]]['region']) != 'range':
                msg = f"Only 'range' region allowed for 'set_range', {kdims[0]!r} is {r!r}."
                raise ValueError(msg)
            regions = {kdims[0]: value}
        elif len(value) == 4:
            if len(kdims) != 2:
                msg = 'Only two key dimensions is allowed in spec.'
                raise ValueError(msg)
            if (r := self.spec[kdims[0]]['region']) != 'range':
                msg = f"Only 'range' region allowed for 'set_range', {kdims[0]!r} is {r!r}."
                raise ValueError(msg)
            if (r := self.spec[kdims[1]]['region']) != 'range':
                msg = f"Only 'range' region allowed for 'set_range', {kdims[1]!r} is {r!r}."
                raise ValueError(msg)
            regions = {kdims[0]: (value[0], value[1]), kdims[1]: (value[2], value[3])}

        self.set_regions(**regions)

    def set_point(self, posx, posy=None):
        print("set_point is legacy use set_regions instead")

        kdims = list(self.spec)
        if posy is None:
            if len(kdims) != 1:
                msg = 'Only one key dimension is allowed in spec.'
                raise ValueError(msg)
            if (r := self.spec[kdims[0]]['region']) != 'point':
                msg = f"Only 'point' region allowed for 'set_point', {kdims[0]!r} is {r!r}."
                raise ValueError(msg)
            regions = {kdims[0]: posx}
        else:
            if len(kdims) != 2:
                msg = 'Only two key dimensions is allowed in spec.'
                raise ValueError(msg)
            if (r := self.spec[kdims[0]]['region']) != 'point':
                msg = f"Only 'point' region allowed for 'set_point', {kdims[0]!r} is {r!r}."
                raise ValueError(msg)
            if (r := self.spec[kdims[1]]['region']) != 'point':
                msg = f"Only 'point' region allowed for 'set_point', {kdims[1]!r} is {r!r}."
                raise ValueError(msg)
            regions = {kdims[0]: posx, kdims[1]: posy}

        self.set_regions(**regions)

    def _add_annotation(self, **fields):
        # Primary key specification is optional
        if self.connector.primary_key.field_name not in fields:
            index_val = self.connector.primary_key(self.connector,
                                                   list(self.annotation_table._field_df.index))
            fields[self.connector.primary_key.field_name] = index_val

        # Don't do anything if self.region is an empty dict
        if self.region and self.region != self._last_region:
            self.annotation_table.add_annotation(self._region, spec=self.spec, **fields)
            self._last_region = self._region.copy()

    def add_annotation(self, **fields):
        self._add_annotation(**fields)

    def update_annotation_region(self, index):
        self.annotation_table.update_annotation_region(self._region, index)

    def update_annotation_fields(self, index, **fields):
        self.annotation_table.update_annotation_fields(index, **fields)


    def delete_annotation(self, index):
        try:
            self.annotation_table.delete_annotation(index)
        except KeyError:
            msg = f'Annotation with index {index!r} does not exist.'
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
        f_keys = set(self.fields) & set(data.columns)
        r_keys = (kwargs.keys() - f_keys) | (set(self.spec) & set(data.columns))
        pk = self.connector.primary_key

        # assert len(set(data.columns) - f_keys - r_keys) == 0

        # Vectorize the conversion?
        for r in data.itertuples(index=index):
            regions = {
                k: tuple([getattr(r, a) for a in kwargs[k]])
                if isinstance(kwargs.get(k, k), tuple)
                else getattr(r, kwargs.get(k, k))
                for k in r_keys
            }
            fields = {k: getattr(r, k) for k in f_keys}
            if index:
                fields[pk.field_name] = pk.cast(r.Index)

            self._set_regions(**regions)
            self._add_annotation(**fields)

    def define_fields(self, fields_df, preserve_index=False):
        print("define_fields is legacy use define_annotations instead")
        if not preserve_index:
            indices = [self.connector.primary_key(self.connector) for el in range(len(fields_df))]
            index_mapping = dict(zip(fields_df.index, indices))
            fields_df = fields_df.set_index(pd.Series(indices,
                                                      name=self.connector.primary_key.field_name))
        else:
            index_mapping = {ind:ind for ind in fields_df.index}
        self.annotation_table.define_fields(fields_df, index_mapping)

    def define_ranges(self, startx, endx, starty=None, endy=None, dims=None):
        print("define_ranges is legacy use define_annotations instead")
        if dims is None:
            msg = 'Please specify dimension annotated by defined ranges'
            raise ValueError(msg)

        self.annotation_table.define_ranges(dims, startx, endx, starty, endy)

    def define_points(self, posx, posy=None, dims=None):
        print("define_points is legacy use define_annotations instead")
        if dims is None:
            msg = 'Please specify dimension annotated by defined ranges'
            raise ValueError(msg)
        self.annotation_table.define_points(dims, posx, posy=posy)


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
    rect_min = param.Number(default=-1000, doc="Temporary parameter until vectorized element fully supported")

    rect_max = param.Number(default=1050, doc="Temporary parameter until vectorized element fully supported")

    kdims = param.List(default=["x"], bounds=(1,3), constant=True, doc="Dimensions of the element")

    indicator = Indicator

    _count = param.Integer(default=0, precedence=-1)

    def __init__(self, annotator: Annotator, **params) -> None:
        super().__init__(**params)

        self._annotation_count_stream = hv.streams.Params(
            parameterized=self,
            parameters=['_count'],
            transient=True,
        )

        self._selection_info = {}

        self._selection_enabled = True
        self._editable_enabled = True
        self._selected_values = []
        self._selected_options = []

        transient=False
        self._edit_streams = [
                hv.streams.BoundsXY(transient=transient),
                hv.streams.SingleTap(transient=transient),
                hv.streams.Lasso(transient=transient),
        ]

        self.annotator = weakref.proxy(annotator)
        self._set_region_types()
        self._element = self._make_empty_element()

    def _set_region_types(self) -> None:
        self.region_types = "-".join([self.annotator.spec[k]['region'] for k in self.kdims])

    @property
    def element(self):
        return self.overlay()

    @property
    def edit_tools(self)-> list[Tool]:
        tools = []
        if self.region_types == "range":
            tools.append(BoxSelectTool(dimensions="width"))
        elif self.region_types == "range-range":
            tools.append(BoxSelectTool())
        elif self.region_types == 'point':
            tools.append('tap')
        return tools

    @classmethod
    def _infer_kdim_dtypes(cls, element):
        if not isinstance(element, hv.Element):
            msg = 'Supplied object {element} is not a bare HoloViews Element'
            raise ValueError(msg)
        kdim_dtypes = {}
        for kdim in element.dimensions(selection='key'):
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
        if "range" not in self.region_types:
            bounds = None

        # If selection enabled, tap stream used for selection not for creating point regions
        if ('point' in self.region_types and self.selection_enabled) or 'point' not in self.region_types:
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

            if None not in [x,y]:
                kdims = list(self.kdims)
                if len(kdims) == 1:
                    self.annotator._set_regions(**{kdims[0]: x})
                elif len(kdims) == 2:
                    self.annotator._set_regions(**{kdims[0]: x, kdims[1]: y})
                else:
                    msg = 'Only 1d and 2d supported for Points'
                    raise ValueError(msg)

            return region_element

        return hv.DynamicMap(inner, streams=self._edit_streams)

    def region_editor(self) -> hv.DynamicMap:
        if not hasattr(self, "_region_editor"):
            self._region_editor = self._make_selection_editor()
        return self._region_editor

    def register_tap_selector(self, element: hv.Element) -> hv.Element:
        def tap_selector(x,y): # Tap tool must be enabled on the element
            # Only select the first
            inputs = {str(k): v for k, v in zip(self.kdims, (x, y))}
            indices = self.annotator.get_indices_by_position(**inputs)
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

    def overlay(self,
        indicators=True,
        editor=True,
        range_style=None,
        point_style=None,
        edit_range_style=None,
        edit_point_style=None,
        highlight=None
    ) -> hv.Overlay:

        if range_style is None: range_style = Indicator.range_style
        if point_style is None: point_style = Indicator.point_style
        if edit_range_style is None: edit_range_style = Indicator.edit_range_style
        if edit_point_style is None: edit_point_style = Indicator.edit_point_style
        if highlight is None: highlight=Indicator.indicator_highlight

        highlighters = {opt:self.selected_dim_expr(v[0], v[1]) for opt,v in highlight.items()}
        indicator_style = Indicator.indicator_style(range_style, point_style, highlighters)
        region_style = Indicator.region_style(edit_range_style, edit_point_style)

        layers = []
        active_tools = []
        if "range" in self.region_types:
             active_tools += ["box_select"]
        elif "point" in self.region_types:
            active_tools += ["tap"]
        layers.append(self._element.opts(tools=self.edit_tools, active_tools=active_tools))

        if indicators:
            layers.append(self.indicators().opts(*indicator_style))
        if editor:
            layers.append(self.region_editor().opts(*region_style))
        return hv.Overlay(layers).collate()

    def _build_hover_tool(self):
        # FIXME: Not generalized yet - assuming range
        extra_cols = [(col, '@{%s}' % col.replace(' ','_')) for col in self.annotator.fields]
        region_tooltips = []
        region_formatters = {}
        for direction, kdim in zip(['x','y'], self.kdims):
            if issubclass(self.annotator.spec[kdim]["type"], datetime_types):
                region_tooltips.append((f'start {kdim}', f'@{direction}0{{%F}}'))
                region_tooltips.append((f'end {kdim}', f'@{direction}1{{%F}}'))
                region_formatters[f'@{direction}0'] = 'datetime'
                region_formatters[f'@{direction}1'] = 'datetime'
            else:
                region_tooltips.append((f'start {kdim}', f'@{direction}0'))
                region_tooltips.append((f'end {kdim}', f'@{direction}1'))

        return HoverTool(tooltips=region_tooltips+extra_cols,
                         formatters = region_formatters)


    def _point_indicators(self, filtered_df, dimensionality, invert_axes=False):
        if dimensionality == '1d':
            return Indicator.points_1d(filtered_df, None, invert_axes=invert_axes)
        else:
            return Indicator.points_2d(filtered_df, None, invert_axes=invert_axes)

    def _range_indicators(self, filtered_df, dimensionality, invert_axes=False):
        fields_mask = self.annotator.annotation_table._field_df.index.isin(filtered_df['_id'])
        field_df = self.annotator.annotation_table._field_df[fields_mask]  # Currently assuming 1-to-1
        if dimensionality == '1d':
            extra_params = {'rect_min':self.rect_min, 'rect_max':self.rect_max} # TODO: Remove extra_params!
            vectorized = Indicator.ranges_1d(filtered_df, field_df, invert_axes=invert_axes,
                                             extra_params=extra_params)
        else:
            vectorized = Indicator.ranges_2d(filtered_df, field_df, invert_axes=invert_axes)
        return vectorized.opts(tools=[self._build_hover_tool()])


    @property
    def static_indicators(self):
        invert_axes = False  # Not yet handled
        if len(self.kdims) == 1:
            dim_mask = self.annotator.annotation_table._mask1D(self.kdims)
            points_df = self.annotator.annotation_table._filter(dim_mask, "point")
            ranges_df = self.annotator.annotation_table._filter(dim_mask, "range")
            if len(points_df) == 0:
                return self._range_indicators(ranges_df, '1d', invert_axes=invert_axes)
            elif len(ranges_df) == 0:
                return self._point_indicators(points_df, '1d', invert_axes=invert_axes)
            else:
                raise NotImplementedError  # FIXME: Both in overlay

        if len(self.kdims) > 1:

            # FIXME: SHH, Converting new region_df format into old format
            df_dim = self.annotator.annotation_table._collapse_region_df(columns=self.kdims)
            if df_dim.empty:
                ranges_df = pd.DataFrame({"_id": [], "value": []})
            else:
                order = [
                    f"start[{self.kdims[0]}]",
                    f"end[{self.kdims[0]}]",
                    f"start[{self.kdims[1]}]",
                    f"end[{self.kdims[1]}]"
                ]
                df2 = df_dim.dropna(axis=0)
                value = tuple(df2[order].values)

                # Convert to accepted format for further processing
                ranges_df = pd.DataFrame({"_id": df2.index, "value": value})
            points_df = [] # self.annotator.annotation_table._filter(dim_mask, "Point")
            if len(points_df) == 0:
                return self._range_indicators(ranges_df, '2d', invert_axes=invert_axes)
            elif len(ranges_df) == 0:
                return self._point_indicators(points_df, '2d', invert_axes=invert_axes)
            else:
                raise NotImplementedError  # FIXME: Both in overlay

    def selected_dim_expr(self, selected_value, non_selected_value):
        self._selected_values.append(selected_value)
        self._selected_options.append({i:selected_value for i in self.annotator.selected_indices})
        index_name = ('id' if (self.annotator.annotation_table._field_df.index.name is None)
                      else self.annotator.annotation_table._field_df.index.name)
        return hv.dim(index_name).categorize(
            self._selected_options[-1], default=non_selected_value)

    @property
    def dim_expr(self):
        return self._selection_info["dim_expr"]

    def show_region(self):
        kdims = list(self.kdims)
        region = {k: v for k, v in self.annotator._region.items() if k in self.kdims}

        if not region:
            return

        if len(kdims) == 1:
            value = region[kdims[0]]
            bounds = (value[0], 0, value[1], 1)
        elif len(kdims) == 2:
            bounds = (region[kdims[0]][0], region[kdims[1]][0],
                      region[kdims[0]][1], region[kdims[1]][1])
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

    def __init__(self, spec: dict, **params):
        """
        The spec argument must be an element or a dictionary of kdim dtypes
        """

        self._elements = {}

        super().__init__(
            spec if isinstance(spec, dict) else self._infer_kdim_dtypes(spec),
            **params,
        )

        self._selection_enabled = True
        self._editable_enabled = True

    @classmethod
    def _infer_kdim_dtypes(self, element: hv.Element) -> dict:
        # Remove?
        return AnnotationDisplay._infer_kdim_dtypes(element)

    def _create_annotation_element(self, element_key: tuple[str, ...]) -> AnnotationDisplay:
        for key in element_key:
            if key not in self.spec:
                msg = f'Dimension {key!r} not in spec'
                raise ValueError(msg)
        return AnnotationDisplay(self, kdims=list(element_key))

    def get_element(self, kdims: tuple[str, ...] | str) -> AnnotationDisplay:
        element_key = (kdims,) if isinstance(kdims, str) else tuple(map(str, kdims))
        if element_key not in self._elements:
            self._elements[element_key] = self._create_annotation_element(element_key)
        return self._elements[element_key]

    def __mul__(self, other: hv.Element) -> hv.Overlay:
        return other * self.get_element(other.kdims).element

    def __rmul__(self, other: hv.Element) -> hv.Overlay:
        return self.__mul__(other)

    def refresh(self, clear=False) -> None:
        for v in self._elements.values():
            hv.streams.Stream.trigger([v._annotation_count_stream])
            if clear:
                v.clear_indicated_region()
            v.show_region()

    def set_annotation_table(self, annotation_table):
        self.select_by_index()
        for v in self._elements.values():
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
            msg = 'At least one index must be specified to delete annotations'
            raise ValueError(msg)
        for index in indices:
            super().delete_annotation(index)
        self.refresh()

    def select_by_index(self, *inds):
        "Set the selection state by the indices i.e. primary key values"

        for v in self._elements.values():
            if not v.selection_enabled:
                inds = ()

            for d, val in zip(v._selected_options, v._selected_values):
                d.clear()
                for ind in inds:
                    d[ind] = val
        super().select_by_index(*inds)
        self.refresh()


    def define_fields(self, fields_df, preserve_index=False):
        """
        If insert_index is True, the index values are inserted as primary key values
        """
        super().define_fields(fields_df, preserve_index=preserve_index)

    def define_ranges(self, startx, endx, starty=None, endy=None, dims=None):
        "Define ranges using element kdims as default dimensions."
        if dims is None:
            dims = list(self.spec)
        super().define_ranges(startx, endx, starty=starty, endy=endy, dims=dims)
        self.refresh()


    def define_points(self, posx, posy=None, dims=None):
        "Define points using element kdims as default dimensions."
        if dims is None:
            dims = list(self.spec)
        super().define_points(posx, posy=posy, dims=dims)
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
        for v in self._elements.values():
            v.selection_enabled = enabled

        if not enabled:
            self.select_by_index()

    @property
    def editable_enabled(self) -> bool:
        return self._editable_enabled

    @editable_enabled.setter
    def editable_enabled(self, enabled: bool) -> None:
        for v in self._elements.values():
            v.editable_enabled = enabled
