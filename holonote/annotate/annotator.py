import sys
import weakref
import holoviews as hv
import numpy as np
import pandas as pd
import param
from bokeh.models.tools import BoxSelectTool, HoverTool
from holoviews.element.selection import Selection1DExpr
from .connector import Connector, SQLiteDB, AnnotationTable



class Indicator:
    """
    Collection of class methods that express annotation data as final
    displayed (vectorized) HoloViews object.
    """

    range_style = dict(color='red', alpha=0.4, apply_ranges=False)
    point_style = dict()
    indicator_highlight = {'alpha':(0.7,0.2)}

    edit_range_style = dict(alpha=0.4, line_alpha=1, line_width=1, line_color='black')
    edit_point_style = dict()

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
            raise Exception('Extra parameters required until vectorized HSpans/VSpans supported')
        return cls._range_indicators(region_df, field_df, "1d", invert_axes, extra_params)

    @classmethod
    def _range_indicators(cls, region_df, field_df, dimensionality, invert_axes=False, extra_params=None):
        rect_data = []

        mdata_vals = ([None] * len(region_df['_id'])
                      if len(field_df.columns)==0 else field_df.to_dict('records'))
        for id_val, value, mdata in zip(region_df['_id'], region_df["value"], mdata_vals):
            if dimensionality=='1d':
                coords = (value[0], extra_params['rect_min'], value[1], extra_params['rect_max'])
            else:
                coords = (value[0], value[2], value[1], value[3]) # LBRT format

            if None in coords: continue

            mdata_tuple =  () if len(field_df.columns)==0 else tuple(mdata.values())
            rect_data.append(coords + mdata_tuple + (id_val,))

        index_col_name = ['id'] if field_df.index.name is None else [field_df.index.name]
        return hv.Rectangles(rect_data, vdims=list(field_df.columns)+index_col_name) # kdims?




class AnnotatorInterface(param.Parameterized):
    """
    Baseclass that expresses the Python interface of an Annotator
    without using any holoviews components and without requiring
    display. Most of this API centers around how the Annotator interacts
    with the AnnotationTable.
    """

    selected_indices = param.List(default=[], doc="Indices of selected annotations")

    kdim_dtypes = param.Dict(default=None, allow_None=True, doc="""
       Dictionary of one or two key dimension names to dtypes (e.g. int, float, datetime).""")

    connector = param.ClassSelector(class_=Connector, allow_None=False)

    annotation_table = param.ClassSelector(class_=AnnotationTable, allow_None=False)

    region_types = param.ListSelector(default=['Range'], objects=['Range', 'Point'], doc="""
       Enabled region types for the current Annotator.""")

    connector_class = SQLiteDB

    def __init__(self, init=True, **params):
        if "annotation_table" not in params:
            params["annotation_table"] = AnnotationTable() #connector=params.get("connector"))

        super().__init__(**params)
        self._region = {}
        self._last_region = None

        self.annotation_table.register_annotator(self)
        self.annotation_table.add_schema_to_conn(self.connector)

        if init:
            self.load()

    def load(self):
        self.connector._initialize(self.connector.column_schema)
        self.annotation_table.load(self.connector, fields=self.connector.fields)

    @property
    def df(self):
        # NOTE: This code is not general yet and makes assumptions
        # 1) That there is a 1-to-1 mapping after filtering
        field_name = self.connector.primary_key.field_name
        field_df = self.annotation_table._field_df
        fields_columns = list(field_df.columns)
        kdim_names = list(self.kdim_dtypes.keys()) # Needed?? If not, moved to AnnotationTable
        if len(self.kdim_dtypes) == 2:
            dim_mask = self.annotation_table._mask2D(kdim_names)
        elif len(self.kdim_dtypes) == 1:
            dim_mask = self.annotation_table._mask1D(kdim_names)

        region_column_names = []
        views = []
        if 'Range' in self.region_types:
            ranges_df = self.annotation_table._filter(dim_mask, 'Range')
            range_view = field_df.merge(ranges_df, left_index=True, right_on='_id')
            views.append(range_view)
            kdim_names = list(self.kdim_dtypes.keys())
            range_column_names = [f'start[{kdim_names[0]}]', f'end[{kdim_names[0]}]']
            range_view[range_column_names[0]] = range_view['value'].apply(lambda x: x[0])
            range_view[range_column_names[1]] = range_view['value'].apply(lambda x: x[1])
            if len(self.kdim_dtypes) == 2:
                range_column_names.extend([f'start[{kdim_names[1]}]', f'end[{kdim_names[1]}]'])
                range_view[range_column_names[2]] = range_view['value'].apply(lambda x: x[2])
                range_view[range_column_names[3]] = range_view['value'].apply(lambda x: x[3])
            region_column_names.extend(range_column_names)
        if 'Point' in self.region_types:
            points_df = self.annotation_table._filter(dim_mask, 'Point')
            point_view = field_df.merge(points_df, left_index=True, right_on="_id")
            views.append(point_view)
            kdim_names = list(self.kdim_dtypes.keys())
            point_column_names = [f'point[{kdim_names[0]}]']
            point_view[point_column_names[0]] = point_view['value'].apply(lambda x: x[0])
            if len(self.kdim_dtypes) == 2:
                point_column_names.extend([f'point[{kdim_names[1]}]'])
                point_view[point_column_names[1]] = point_view['value'].apply(lambda x: x[1])
            region_column_names.extend(point_column_names)
        view = pd.concat(views)
        view = view.rename(columns={'_id':field_name})
        column_ordering = [field_name] + region_column_names + fields_columns
        return view[column_ordering].set_index(field_name)


    def refresh(self, clear=False):
        "Method to update display state of the annotator and optionally clear stale visual state"
        pass

    # def set_annotation_table(self, annotation_table): # FIXME! Won't work anymore, set_connector??
    #     self._region = {}
    #     self.annotation_table = annotation_table
    #     self.annotation_table.register_annotator(self)
    #     self.annotation_table._update_index()
    #     self.snapshot()

    # Selecting annotations

    def select_by_index(self, *inds):
        "Set the selection state by the indices i.e. primary key values"
        self.selected_indices = list(inds)

    @property
    def selected_index(self):
        "Convenience property returning a single selected index (the first one) or None"
        return self.selected_indices[0] if len(self.selected_indices) > 0 else None


    def _get_range_indices_by_position(self, dim1_pos, dim2_pos=None):
        ranges = self.annotation_table._region_df[self.annotation_table._region_df['region_type']=='Range']
        mask = ranges['value'].apply(lambda tpl: tpl[0] <= dim1_pos < tpl[1])
        if dim2_pos is not None:
            mask_dim2 = ranges['value'].apply(lambda tpl: tpl[2] <= dim2_pos < tpl[3])
            mask = mask & mask_dim2
        return list(ranges[mask]['_id'])


    def get_indices_by_position(self, dim1_pos, dim2_pos=None):
        "Return primary key values matching given position in data space"
        # Lots TODO! 2 Dimensions, different annotation types etc.
        range_matches = self._get_range_indices_by_position(dim1_pos, dim2_pos)
        event_matches = [] # TODO: Needs hit testing or similar for point events
        return range_matches + event_matches

    @property
    def region(self):
        return self._region

    def set_range(self, startx, endx, starty=None, endy=None):
        if len(self.kdim_dtypes) == 2 and None in [starty, endy ]:
            raise ValueError('Two key dimensions specified: both starty and endy arguments required')
        if 'Range' not in self.region_types:
            raise ValueError(f'Range region types not enabled as region_types={self.region_types}')
        if None in [starty, endy] and ([starty, endy] != [None, None]):
            raise ValueError('Both starty and endy need to be non-None')

        value = (startx, endx) if(None in [starty, endy]) else (startx, endx, starty, endy)
        kdims = list(self.kdim_dtypes.keys())
        self._set_region('Range', value, *kdims)

    def set_point(self, posx, posy=None):
        if 'Point' not in self.region_types:
            raise ValueError(f'Point region types not enabled as region_types={self.region_types}')

        value = (posx,None) if posy is None else (posx, posy)
        kdims = list(self.kdim_dtypes.keys())
        self._set_region('Point', value,  *kdims)


    def _set_region(self, region_type, value=None, dim1=None, dim2=None):
        "Use _set_region(None) to clear currently defined region"
        if (region_type, value, dim1, dim2) == (None, None, None, None):
            self._region = {}
        elif None not in [value, dim1]:
            self._region = {'region_type':region_type, 'value':value, 'dim1':dim1, 'dim2':dim2}
        else:
            raise Exception('Both value and dim1 required for non-None region type')

    def add_annotation(self, **fields):  #   Rename box to range.
        "Primary key specification is optional"
        if self.connector.primary_key.field_name not in fields:
            index_val = self.connector.primary_key(self.connector,
                                                   list(self.annotation_table._field_df.index))
            fields[self.connector.primary_key.field_name] = index_val

        if self.region != self._last_region:
            if len(self.annotation_table._annotators)>1:
                raise AssertionError('Multiple annotation instances attached to the connector: '
                                     'Call add_annotation directly from the associated connector.')
            self.annotation_table.add_annotation('annotator-regions', **fields)
        self._last_region = self._region

    def update_annotation_region(self, index):
        self.annotation_table.update_annotation_region(index)


    def update_annotation_fields(self, index, **fields):
        self.annotation_table.update_annotation_fields(index, **fields)


    def delete_annotation(self, index):
        self.annotation_table.delete_annotation(index)

    # Defining initial annotations

    def define_fields(self, fields_df, preserve_index=False):
        if not preserve_index:
            indices = [self.connector.primary_key(self.connector) for el in range(len(fields_df))]
            index_mapping = {old:new for old, new in zip(fields_df.index, indices)}
            fields_df = fields_df.set_index(pd.Series(indices,
                                                      name=self.connector.primary_key.field_name))
        else:
            index_mapping = {ind:ind for ind in fields_df.index}
        self.annotation_table.define_fields(fields_df, index_mapping)

    def define_ranges(self, startx, endx, starty=None, endy=None, dims=None):
        if 'Range' not in self.region_types:
            raise ValueError('Range region types not enabled')
        if dims is None:
            raise ValueError('Please specify dimension annotated by defined ranges')

        self.annotation_table.define_ranges(dims, startx, endx, starty, endy)

    def define_points(self, posx, posy=None, dims=None):
        if 'Point' not in self.region_types:
            raise ValueError('Point region types not enabled')

        if dims is None:
            raise ValueError('Please specify dimension annotated by defined ranges')
        self.annotation_table.define_points(dims, posx, posy=posy)


    # Snapshotting and reverting

    def revert_to_snapshot(self):
        self.annotation_table.revert_to_snapshot()

    def snapshot(self):
        self.annotation_table.snapshot()

    def commit(self):
        # self.annotation_table.initialize_table(self.connector)  # Only if not in params
        return self.annotation_table.commits(self.connector)


class Annotator(AnnotatorInterface):
    """
    An annotator displays the contents of an AnnotationTable and
    provides the means to manipulate view the corresponding contents,
    add new annotations and update existing annotations.
    """

    rect_min = param.Number(default=-1000, doc="Temporary parameter until vectorized element fully supported")

    rect_max = param.Number(default=1050, doc="Temporary parameter until vectorized element fully supported")

    _count = param.Integer(default=None, precedence=-1)

    indicator = Indicator

    def __init__(self, spec, **params):
        """
        The spec argument must be an element or a dictionary of kdim dtypes
        """
        params['_count'] = 0
        self.element = None
        if not isinstance(spec, dict):
            self.element = spec.clone() # Clone so we can add a tap tool and trust our copy.

        connector_kws = {'fields':params.pop('fields')} if 'fields' in params else {}
        connector = params.pop('connector') if 'connector' in params else self.connector_class(**connector_kws)
        super().__init__(connector = connector,
                         **dict(kdim_dtypes=(spec if self.element is None
                                             else self._infer_kdim_dtypes(spec)), **params))

        self._selection_info = {}

        self._annotation_count_stream = hv.streams.Params(parameterized=self,
                                                          parameters=['_count'], transient=True)
        self._selected_values = []
        self._selected_options = []

        self._selection_enabled = True
        self._editable_enabled = True

        transient=False
        self._edit_streams = [
                hv.streams.BoundsXY(transient=transient),
                hv.streams.SingleTap(transient=transient),
                hv.streams.Lasso(transient=transient),
        ]

    @property
    def edit_tools(self):
        tools = []
        if 'Range' in self.region_types and len(self.kdim_dtypes)==1:
            tools.append(BoxSelectTool(dimensions="width"))
        elif 'Range' in self.region_types and len(self.kdim_dtypes)==2:
            tools.append(BoxSelectTool())

        if 'Point' in self.region_types:
            tools.append('tap')

        return tools

    @classmethod
    def _infer_kdim_dtypes(cls, element):
        if not isinstance(element, hv.Element):
            raise ValueError('Supplied object {element} is not a bare HoloViews Element')
        kdim_dtypes = {}
        for kdim in element.dimensions(selection='key'):
            kdim_dtypes[str(kdim)] = type(element.dimension_values(kdim)[0])
        return kdim_dtypes

    def refresh(self, clear=False):
        hv.streams.Stream.trigger([self._annotation_count_stream])
        if clear:
            self.clear_indicated_region()

    def set_annotation_table(self, annotation_table):
        self.select_by_index()
        self._set_region(None)
        super().set_annotation_table(annotation_table)
        self.refresh(clear=True)

    @property
    def selection_enabled(self):
        return self._selection_enabled

    @selection_enabled.setter
    def selection_enabled(self, enabled):
        self._selection_enabled = enabled
        if not enabled:
            self.select_by_index()

    @property
    def editable_enabled(self):
        return self._editable_enabled

    @editable_enabled.setter
    def editable_enabled(self, enabled):
        self._editable_enabled = enabled
        if not enabled:
            self.clear_indicated_region()

    def enable_editable(self, enabled=True):
        print('enable_editable method deprecated - set editable_enabled property directly')
        self.editable_enabled = enabled

    def clear_indicated_region(self):
        "Clear any region currently indicated on the plot by the editor"
        self._edit_streams[0].event(bounds=None)
        self._edit_streams[1].event(x=None, y=None)
        self._edit_streams[2].event(geometry=None)
        self._set_region(None)

    @property
    def selection_element(self):
        if self.element is None:
            kdims = list(self.kdim_dtypes.keys())
            kdim_dtype = list(self.kdim_dtypes.values())[0]
            return hv.Curve(([kdim_dtype(), kdim_dtype()],
                             [0,1]), kdims=kdims) # Note: Any concrete Selection1dExpr will do...

        if isinstance(self.element, (hv.NdOverlay, hv.Overlay)):
            expr_1d_set = set(isinstance(el, Selection1DExpr) for el in self.element)
            if len(expr_1d_set) == 1:
                return next(iter(self.element))
            else:
                raise ValueError("Inconsistent selection types in overlay")
        else:
            return self.element


    def selected_dim_expr(self, selected_value, non_selected_value):
        self._selected_values.append(selected_value)
        self._selected_options.append({i:selected_value for i in self.selected_indices})
        index_name = ('id' if (self.annotation_table._field_df.index.name is None)
                      else self.annotation_table._field_df.index.name)
        return hv.dim(index_name).categorize(
            self._selected_options[-1], default=non_selected_value)

    def clear_edits(self):
        super().clear_edits()
        self.clear_indicated_region()

    @property
    def dim_expr(self):
        return self._selection_info["dim_expr"]


    def show_region(self):
        if self.region != {}:
            if len(self.region['value']) == 2:
                bounds = (self.region['value'][0], 0,
                      self.region['value'][1], 1)
            else:
                bounds = (self.region['value'][0], self.region['value'][2],
                          self.region['value'][1], self.region['value'][3])
            self._edit_streams[0].event(bounds = bounds)

    def _filter_stream_values(self, bounds, x, y, geometry):
        if not self._editable_enabled:
            return (None, None, None, None)
        if 'Range' not in self.region_types:
            bounds = None

        # If selection enabled, tap stream used for selection not for creating point regions
        if 'Point' in self.region_types and self.selection_enabled:
            x, y = None, None
        elif 'Point' not in self.region_types:
            x, y = None, None

        return bounds, x, y, geometry

    def _make_selection_dmap(self):
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
                dim1, dim2 = None, None
                dim_names = list(bbox.keys())
                dim1 = dim_names[0]
                value = bbox[dim1]

                if len(bbox)== 2:
                    dim2 = dim_names[1]
                    start1,end1 = value
                    start2,end2 = bbox[dim2]
                    value = (start1, end1, start2,end2)

                self._set_region('Range', value=value, dim1=dim1, dim2=dim2)

            if None not in [x,y]:
                kdims = list(self.kdim_dtypes.keys())
                value = (x,None) if len(self.kdim_dtypes) == 1 else (x,y)
                self._set_region('Point', value,  *kdims)

            return region_element

        return hv.DynamicMap(inner, streams=self._edit_streams)

    def register_tap_selector(self, element):
        def tap_selector(x,y): # Tap tool must be enabled on the element
            # Only select the first

            if len(self.kdim_dtypes) == 1:
                indices = self.get_indices_by_position(x)
            elif len(self.kdim_dtypes) == 2:
                indices = self.get_indices_by_position(x, y)
            if indices:
                self.select_by_index(indices[0])
            else:
                self.select_by_index()
        tap_stream = hv.streams.Tap(source=element, transient=True)
        tap_stream.add_subscriber(tap_selector)
        return element


    def indicators(self):

        if self.element is not None:
            self.register_tap_selector(self.element)

        def inner(_count):
            return self.static_indicators

        return hv.DynamicMap(inner, streams=[self._annotation_count_stream])

    def overlay(self,
                element = True,
                indicators=True,
                editor=True,
                range_style=None,
                point_style=None,
                edit_range_style=None,
                edit_point_style=None,
                highlight=None):

        if range_style is None: range_style = Indicator.range_style
        if point_style is None: point_style = Indicator.point_style
        if edit_range_style is None: edit_range_style = Indicator.edit_range_style
        if edit_point_style is None: edit_point_style = Indicator.edit_point_style
        if highlight is None: highlight=Indicator.indicator_highlight

        highlighters = {opt:self.selected_dim_expr(v[0], v[1]) for opt,v in highlight.items()}
        indicator_style = Indicator.indicator_style(range_style, point_style, highlighters)
        region_style = Indicator.region_style(edit_range_style, edit_point_style)

        layers = []
        if element is True and self.element is None:
            raise ValueError('Annotator was not supplied an element to display.')
        elif element not in [True, False]:
            element = self.register_tap_selector(element)
            layers.append(element.opts(tools=self.edit_tools, active_tools=["box_select"]))
        else:
            layers.append(self.element.opts(tools=self.edit_tools))

        if indicators:
            layers.append(self.indicators().opts(*indicator_style))
        if editor:
            layers.append(self.region_editor().opts(*region_style))
        return hv.Overlay(layers).collate()

    def region_editor(self):
        return self._make_selection_dmap()

    def _build_hover_tool(self):
        # FIXME: Not generalized yet - assuming range
        extra_cols = [(col, '@{%s}' % col.replace(' ','_')) for col in self.annotation_table._field_df.columns]
        region_tooltips = []
        region_formatters = {}
        for direction, kdim, dtype in zip(['x','y'], self.kdim_dtypes.keys(), self.kdim_dtypes.values()):
            if self.kdim_dtypes[kdim] is np.datetime64:
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
        fields_mask = self.annotation_table._field_df.index.isin(filtered_df['_id'])
        field_df = self.annotation_table._field_df[fields_mask]  # Currently assuming 1-to-1
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
        kdims = list(self.kdim_dtypes.keys())
        if len(kdims) == 1:
            dim_mask = self.annotation_table._mask1D(kdims)
            points_df = self.annotation_table._filter(dim_mask, "Point")
            ranges_df = self.annotation_table._filter(dim_mask, "Range")
            if len(points_df) == 0:
                return self._range_indicators(ranges_df, '1d', invert_axes=invert_axes)
            elif len(ranges_df) == 0:
                return self._point_indicators(points_df, '1d', invert_axes=invert_axes)
            else:
                raise NotImplementedError  # FIXME: Both in overlay

        if len(kdims) > 1:
            dim_mask = self.annotation_table._mask2D(kdims)
            points_df = self.annotation_table._filter(dim_mask, "Point")
            ranges_df = self.annotation_table._filter(dim_mask, "Range")
            if len(points_df) == 0:
                return self._range_indicators(ranges_df, '2d', invert_axes=invert_axes)
            elif len(ranges_df) == 0:
                return self._point_indicators(points_df, '2d', invert_axes=invert_axes)
            else:
                raise NotImplementedError  # FIXME: Both in overlay


    # Methods overriding the baseclass to add extra interactivity

    def select_by_index(self, *inds):
        "Set the selection state by the indices i.e. primary key values"

        if not self.selection_enabled:
            inds = ()
        super().select_by_index(*inds)

        for d, val in zip(self._selected_options, self._selected_values):
            d.clear()
            for ind in inds:
                d[ind] = val

        self.refresh()

    def add_annotation(self, **fields):
        super().add_annotation(**fields)
        self.refresh(clear=True)

    def update_annotation_region(self, index):
        super().update_annotation_region(index)
        self.refresh()


    def update_annotation_fields(self, index, **fields):
        super().update_annotation_fields(index, **fields)
        self.refresh()


    def delete_annotation(self, *indices):
        if len(indices)==0:
            raise ValueError('At least one index must be specified to delete annotations')
        for index in indices:
            super().delete_annotation(index)
        self.refresh()

    def define_fields(self, fields_df, preserve_index=False):
        """
        If insert_index is True, the index values are inserted as primary key values
        """
        super().define_fields(fields_df, preserve_index=preserve_index)

    # Pass-through methods that get dimensionality from self.element
    def define_ranges(self, startx, endx, starty=None, endy=None, dims=None):
        "Define ranges using element kdims as default dimensions."
        if dims is None:
            dims = list(self.kdim_dtypes.keys())
        super().define_ranges(startx, endx, starty=starty, endy=endy, dims=dims)
        self.refresh()


    def define_points(self, posx, posy=None, dims=None):
        "Define points using element kdims as default dimensions."
        if dims is None:
            dims = list(self.kdim_dtypes.keys())
        super().define_points(posx, posy=posy, dims=dims)
        self.refresh()

    @property
    def has_snapshot(self):
        return self.annotation_table.has_snapshot

    def revert_to_snapshot(self):
        super().revert_to_snapshot()
        self.refresh()

    def set_range(self, startx, endx, starty=None, endy=None):
        super().set_range(startx, endx, starty, endy)
        self.show_region()

    def set_point(self, posx, posy=None):
        super().set_point(posx, posy)
        self.show_region()
