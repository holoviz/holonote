from __future__ import annotations

from typing import TYPE_CHECKING, Any

import holoviews as hv
import pandas as pd
import param

from .._warnings import warn
from .connector import Connector, SQLiteDB
from .plotting import AnnotationDisplay, Indicator, Style  # noqa: F401
from .table import AnnotationTable

if TYPE_CHECKING:
    from .typing import SpecDict


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

    default_region = param.Selector(
        default="range", objects=["range", "point"], doc="Default region, if nothing is provided"
    )

    connector = param.ClassSelector(class_=Connector, allow_None=False)

    connector_class = SQLiteDB

    def __init__(self, spec, **params):
        if "connector" not in params:
            params["connector"] = self.connector_class()

        spec = self.normalize_spec(spec, default_region=params.get("default_region"))
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
    def normalize_spec(self, input_spec: dict[str, Any], default_region=None) -> SpecDict:
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
                v = {"type": v, "region": default_region or self.default_region}

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

        # See: https://github.com/holoviz/holonote/pull/50
        self.clear_regions()

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


class Annotator(AnnotatorInterface):
    """
    An annotator displays the contents of an AnnotationTable and
    provides the means to manipulate view the corresponding contents,
    add new annotations and update existing annotations.
    """

    groupby = param.Selector(default=None, doc="Groupby dimension", allow_refs=True)
    visible = param.ListSelector(
        default=[], doc="Visible dimensions, needs groupby enabled", allow_refs=True
    )
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

    @param.depends("style.param", "groupby", "visible", watch=True)
    def _refresh_style(self) -> None:
        self.refresh()

    @param.depends("fields", watch=True, on_init=True)
    def _set_groupby_objects(self) -> None:
        self.param.groupby.objects = [*self.fields, None]
