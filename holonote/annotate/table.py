from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .connector import Connector
    from .typing import SpecDict


class AnnotationTable:
    """
    Class that stores and manipulates annotation data, including methods
    to declare annotations and commit edits back to the original data
    source such as a database.
    """

    columns = ("region", "dim", "value", "_id")

    def __init__(self):
        """
        Either specify annotation fields with filled field columns
        (via connector or dataframe) or declare the expected
        field columns if starting with no annotation data.
        """
        self.__region_df = pd.DataFrame(columns=self.columns)
        self._new_regions = []
        self.__field_df = None
        self._new_fields = []

        self._edits = []
        self._index_mapping = {}

        self._field_df_snapshot, self._region_df_snapshot = None, None

    def load(self, connector=None, fields_df=None, primary_key_name=None, fields=None, spec=None):
        """
        Load the AnnotationTable from a connector or a fields DataFrame.
        """
        if fields is None:
            fields = []

        if [connector, primary_key_name] == [None, None]:
            msg = "Either a connector instance must be supplied or the primary key name supplied"
            raise ValueError(msg)
        if len(fields) < 1:
            msg = "More than one field column is required"
            raise ValueError(msg)
        primary_key_name = (
            primary_key_name if primary_key_name else connector.primary_key.field_name
        )

        if fields_df:
            fields_df = fields_df[fields].copy()  # Primary key/index for annotations
            self.__field_df = fields_df
        elif connector:
            self.load_annotation_table(connector, fields, spec)
        elif fields_df is None:
            fields_df = pd.DataFrame(columns=[primary_key_name, *fields])
            fields_df = fields_df.set_index(primary_key_name)
            self.__field_df = fields_df

        self.clear_edits()

    @property
    def has_snapshot(self) -> bool:
        return self._field_df_snapshot is not None

    def revert_to_snapshot(self):
        "Clears outstanding changes and used to implement an basic undo system."
        if self._field_df_snapshot is None:
            msg = "Call snapshot method before calling revert_to_snapshot"
            raise Exception(msg)
        self._new_regions = []
        self._new_fields = []
        self.__field_df = self._field_df_snapshot
        self.__region_df = self._region_df_snapshot
        self.clear_edits()

    def snapshot(self):
        "Saves a snapshot. Expected to only be used after a syncing commit"
        self._field_df_snapshot, self._region_df_snapshot = self._snapshot()

    def _snapshot(self):
        return self._field_df.copy(), self._region_df.copy()

    @property
    def index(self):
        return list(self._field_df.index)

    @property
    def index_name(self):
        return self.__field_df.index.name

    def _expand_commit_by_id(self, id_val, fields=None, region_fields=None):
        kwargs = self._field_df.loc[[id_val]].to_dict("records")[0]
        if fields:
            kwargs = {k: v for k, v in kwargs.items() if k in fields}
        kwargs[self.index_name] = id_val
        if region_fields == []:
            return kwargs
        items = self._region_df[self._region_df["_id"] == id_val]
        for i in items.itertuples(index=False):
            if i.region == "range":
                kwargs[f"start_{i.dim}"] = i.value[0]
                kwargs[f"end_{i.dim}"] = i.value[1]
            else:
                kwargs[f"{i.region}_{i.dim}"] = i.value
        return kwargs

    def _expand_save_commits(self, ids):
        return {"field_list": [self._expand_commit_by_id(id_val) for id_val in ids]}

    def _create_commits(self):
        "Expands out the commit history into commit operations"
        fields_inds = set(self._field_df.index)
        region_inds = set(self._region_df["_id"].unique())
        unassigned_inds = fields_inds - region_inds
        if unassigned_inds:
            msg = f"Following annotations have no associated region: {unassigned_inds}"
            raise ValueError(msg)

        commits = []
        for edit in self._edits:
            operation = edit["operation"]
            if operation == "insert":
                # May be missing due to earlier deletion operation - nothing to do
                if edit["id"] not in self._field_df.index:
                    continue
                kwargs = self._expand_commit_by_id(edit["id"])

            elif operation == "delete":
                kwargs = {"id_val": edit["id"]}
            elif operation == "update":
                if edit["id"] not in self._field_df.index:
                    continue
                kwargs = self._expand_commit_by_id(
                    edit["id"], fields=edit["fields"], region_fields=edit["region_fields"]
                )
            elif operation == "save":
                kwargs = self._expand_save_commits(edit["ids"])
            commits.append({"operation": operation, "kwargs": kwargs})

        return commits

    def commits(self, connector):
        commits = self._create_commits()
        for commit in commits:
            operation = commit["operation"]
            kwargs = connector.transforms[operation](commit["kwargs"])
            getattr(connector, connector.operation_mapping[operation])(**kwargs)

        self.clear_edits()
        return commits

    def clear_edits(self, edit_type=None):
        "Clear edit state and index mapping"
        self._edits = []
        self._index_mapping = {}

    def add_annotation(self, regions: dict[str, Any], spec: SpecDict, **fields):
        "Takes a list of regions or the special value 'annotation-regions' to use attached annotators"
        index_value = fields.pop(self.index_name)
        self._add_annotation_fields(index_value, fields=fields)

        for kdim, value in regions.items():
            if not value:
                continue

            d = {"region": spec[kdim]["region"], "dim": kdim, "value": value, "_id": index_value}
            self._new_regions.append(d)

        self._edits.append({"operation": "insert", "id": index_value})

    def _add_annotation_fields(self, index_value, fields=None):
        index_name_set = set() if self.index_name is None else {self.index_name}
        unknown_kwargs = set(fields.keys()) - set(self.__field_df.columns)
        if unknown_kwargs - index_name_set:
            unknown_str = ", ".join([f"{k!r}" for k in sorted(unknown_kwargs)])
            msg = f"Unknown fields columns: {unknown_str}"
            raise KeyError(msg)

        self._new_fields.append(dict(fields, **{self.index_name: index_value}))

    def delete_annotation(self, index):
        if index is None:
            msg = "Deletion index cannot be None"
            raise ValueError(msg)
        self.__region_df = self._region_df[
            self._region_df["_id"] != index
        ]  # Could match multiple rows
        self.__field_df = self._field_df.drop(index, axis=0)

        self._edits.append({"operation": "delete", "id": index})

    def update_annotation_fields(self, index, **fields):
        for column, value in fields.items():
            self._field_df.loc[index, column] = value

        self._edits.append(
            {"operation": "update", "id": index, "fields": list(fields), "region_fields": []}
        )

    def update_annotation_region(self, region, index):
        index_mask = self._region_df._id == index
        for kdim, value in region.items():
            mask = self._region_df[index_mask & (self._region_df.dim == kdim)]
            if mask.shape[0] != 1:
                msg = (
                    f"Expected one region for {kdim} with index {index} but found {mask.shape[0]}"
                )
                raise ValueError(msg)
            self._region_df.loc[mask.index.values[0], "value"] = value
            self._edits.append(
                {"operation": "update", "id": index, "fields": None, "region_fields": kdim}
            )

    def define_fields(self, fields_df, index_mapping):
        # Need a staging area to hold everything till initialized
        self._index_mapping.update(index_mapping)  # Rename _field_df
        self.__field_df = pd.concat([self.__field_df, fields_df])
        self._edits.append({"operation": "save", "ids": self.index})

    def _empty_expanded_region_df(self, *, spec: SpecDict, dims: list[str] | None) -> pd.DataFrame:
        invalid_dims = set(dims) - spec.keys()
        if invalid_dims:
            invalid_dims_str = ", ".join([f"{dim!r}" for dim in sorted(invalid_dims)])
            msg = f"Dimension(s) {invalid_dims_str} not in the spec"
            raise ValueError(msg)

        columns, types = [], []
        for dim in dims:
            region = spec[dim]["region"]
            dtype = pd.NaT if issubclass(t := spec[dim]["type"], dt.date) else t()
            if region == "range":
                columns.extend([f"start[{dim}]", f"end[{dim}]"])
                types.extend([dtype, dtype])
            else:
                columns.append(f"{region}[{dim}]")
                types.append(dtype)

        return pd.DataFrame([types], columns=columns).drop(index=0)

    def _expand_region_df(self, *, spec: SpecDict, dims: list[str] | None = None) -> pd.DataFrame:
        data = self._region_df.pivot(index="_id", columns="dim", values="value")
        dims = list(dims or spec)

        expanded = self._empty_expanded_region_df(spec=spec, dims=dims)
        if data.empty:
            return expanded

        set_index = True
        for dim in dims:
            region = spec[dim]["region"]
            if dim not in data.columns:
                continue
            elif region == "range":
                na_mask = data[dim].isnull()
                data.loc[na_mask, dim] = data.loc[na_mask, dim].apply(lambda *x: (None, None))
                expanded[[f"start[{dim}]", f"end[{dim}]"]] = list(data[dim])
            else:
                dtype = expanded.dtypes[f"{region}[{dim}]"]
                expanded[f"{region}[{dim}]"] = data[dim].astype(dtype)

            if set_index:
                expanded.index = data.index
                set_index = False

        return expanded

    def get_dataframe(
        self, *, spec: SpecDict | None = None, dims: list[str] | None = None
    ) -> pd.DataFrame:
        field_df = self._field_df
        region_df = self._expand_region_df(spec=spec, dims=dims)

        df = region_df.join(field_df, how="left")
        df.index.name = self.index_name
        df = df.reindex(field_df.index)
        return df

    def load_annotation_table(self, conn: Connector, fields: list[str], spec: SpecDict) -> None:
        """Load the AnnotationTable region and field DataFrame from a connector.

        Parameters
        ----------
        conn : Connector
            Database connection
        fields : list[str]
            List of field columns to load from the connector
        spec : SpecDict
            Dictionary of region specifications
        """
        df = conn.transforms["load"](conn.load_dataframe())

        # Load fields dataframe
        fields_df = df[fields].copy()
        self.define_fields(fields_df, {ind: ind for ind in fields_df.index})
        # Replace: self._field_df = pd.concat([self._field_df, fields_df])

        # Load region dataframe
        if df.empty:
            self.__region_df = pd.DataFrame(columns=self.columns)
            return

        data = []
        for kdim, info in spec.items():
            region = info["region"]
            if region == "range":
                value = list(zip(df[f"start_{kdim}"], df[f"end_{kdim}"]))
            else:
                value = df[f"{region}_{kdim}"]

            subdata = pd.DataFrame(
                {"region": region, "dim": kdim, "value": value, "_id": list(df.index)}
            )
            if region == "range":
                empty_mask = subdata["value"] == (None, None)
            else:
                empty_mask = subdata["value"].isnull()

            data.append(subdata[~empty_mask])

        self.__region_df = pd.concat(data, ignore_index=True)

        self.clear_edits()

    @property
    def _region_df(self):
        if self._new_regions:
            new_regions = pd.DataFrame(self._new_regions)
            if self.__region_df.empty:
                self.__region_df = new_regions
            else:
                self.__region_df = pd.concat((self.__region_df, new_regions), ignore_index=True)
            self._new_regions = []

        return self.__region_df

    @property
    def _field_df(self):
        if self._new_fields:
            new_fields = pd.DataFrame(self._new_fields)
            new_fields = new_fields.set_index(self.index_name)
            if self.__field_df.empty:
                self.__field_df[new_fields.columns] = new_fields
            else:
                self.__field_df = pd.concat((self.__field_df, new_fields))
            self._new_fields = []

        return self.__field_df
