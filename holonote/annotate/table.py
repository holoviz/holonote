from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import param

if TYPE_CHECKING:
    from .connector import Connector
    from .typing import SpecDict


class AnnotationTable(param.Parameterized):
    """
    Class that stores and manipulates annotation data, including methods
    to declare annotations and commit edits back to the original data
    source such as a database.
    """
    columns = ("region", "dim", "value", "_id")

    index = param.List(default=[])

    def __init__(self,  **params):
        """
        Either specify annotation fields with filled field columns
        (via connector or dataframe) or declare the expected
        field columns if starting with no annotation data.
        """
        super().__init__(**params)

        self._region_df = pd.DataFrame(columns=self.columns)
        self._field_df = None

        self._edits = []
        self._index_mapping = {}

        self._update_index()
        self._field_df_snapshot, self._region_df_snapshot = None, None

        self._annotators = weakref.WeakValueDictionary()

    def load(self, connector=None, fields_df=None, primary_key_name=None, fields=None, spec=None):
        """
        Load the AnnotationTable from a connector or a fields DataFrame.
        """
        if fields is None:
            fields = []

        if [connector, primary_key_name] == [None,None]:
            raise ValueError('Either a connector instance must be supplied or the primary key name supplied')
        if len(fields) < 1:
            raise ValueError('More than one field column is required')
        primary_key_name = primary_key_name if primary_key_name else connector.primary_key.field_name

        if fields_df:
            fields_df = fields_df[fields].copy() # Primary key/index for annotations
            self._field_df = fields_df
        elif connector:
            self.load_annotation_table(connector, fields, spec)
        elif fields_df is None:
            fields_df = pd.DataFrame(columns=[primary_key_name, *fields])
            fields_df = fields_df.set_index(primary_key_name)
            self._field_df = fields_df

        self.clear_edits()
        self._update_index()

    def register_annotator(self, annotator):
        self._annotators[id(annotator)] = annotator


    # FIXME: Multiple region updates
    def update_annotation_region(self, index):
        region = next(iter(self._annotators.values()))._region
        if region == {}:
            print('No new region selected. Skipping')
            return

        value = region['value']
        mask = self._region_df[self._region_df._id == index]
        assert len(mask)==1, 'TODO: Handle multiple region updates for single index'
        self._region_df.at[mask.index.values[0], 'value'] = value
        self._edits.append({'operation':'update', 'id':index,
                            'fields': None,
                            'region_fields':[]})

    @property
    def has_snapshot(self) -> bool:
        return self._field_df_snapshot is not None

    def revert_to_snapshot(self):
        "Clears outstanding changes and used to implement an basic undo system."
        if self._field_df_snapshot is None:
            raise Exception('Call snapshot method before calling revert_to_snapshot')
        self._field_df = self._field_df_snapshot
        self._region_df = self._region_df_snapshot
        self.clear_edits()


    def snapshot(self):
        "Saves a snapshot. Expected to only be used after a syncing commit"
        self._field_df_snapshot, self._region_df_snapshot = self._snapshot()


    def _snapshot(self):
        return self._field_df.copy(), self._region_df.copy()

    def _update_index(self) -> None:
        if self._field_df is None:
            self.index = []
            return

        self.index = list(self._field_df.index)

    def _expand_commit_by_id(self, id_val, fields=None, region_fields=None):
        kwargs = self._field_df.loc[[id_val]].to_dict('records')[0]
        if fields:
            kwargs = {k:v for k,v in kwargs.items() if k in fields}
        kwargs[self._field_df.index.name] = id_val
        if region_fields == []:
            return kwargs
        items = self._region_df[self._region_df['_id'] == id_val]
        for i in items.itertuples(index=False):
            if i.region == "range":
                kwargs[f"start_{i.dim}"] = i.value[0]
                kwargs[f"end_{i.dim}"] = i.value[1]
            else:
                kwargs[f"{i.region}_{i.dim}"] = i.value
        return kwargs

    def _expand_save_commits(self, ids):
        return {'field_list':[self._expand_commit_by_id(id_val) for id_val in ids]}

    def _create_commits(self):
        "Expands out the commit history into commit operations"
        fields_inds = set(self._field_df.index)
        region_inds = set(self._region_df['_id'].unique())
        unassigned_inds = fields_inds - region_inds
        if unassigned_inds:
            raise ValueError(f'Following annotations have no associated region: {unassigned_inds}')

        commits = []
        for edit in self._edits:
            operation = edit['operation']
            if operation == 'insert':
                # May be missing due to earlier deletion operation - nothing to do
                if edit['id'] not in self._field_df.index:
                    continue
                kwargs = self._expand_commit_by_id(edit['id'])

            elif operation == 'delete':
                kwargs = {'id_val': edit['id']}
            elif operation == 'update':
                if edit['id'] not in self._field_df.index:
                    continue
                kwargs = self._expand_commit_by_id(edit['id'],
                                                   fields=edit['fields'],
                                                   region_fields=edit['region_fields'])
            elif operation == 'save':
                kwargs = self._expand_save_commits(edit['ids'])
            commits.append({'operation':operation, 'kwargs':kwargs})

        return commits

    def commits(self, connector):
        commits = self._create_commits()
        for commit in commits:
            operation = commit['operation']
            kwargs = connector.transforms[operation](commit['kwargs'])
            getattr(connector,connector.operation_mapping[operation])(**kwargs)

        for annotator in self._annotators.values():
            annotator.annotation_table.clear_edits()

        return commits

    def clear_edits(self, edit_type=None):
        "Clear edit state and index mapping"
        self._edits = []
        self._index_mapping = {}

    def add_annotation(self, regions: dict[str, Any], spec: SpecDict, **fields):
        "Takes a list of regions or the special value 'annotation-regions' to use attached annotators"
        index_value = fields.pop(self._field_df.index.name)
        self._add_annotation_fields(index_value, fields=fields)

        data = []
        for kdim, value in regions.items():
            if not value:
                continue

            d = {"region": spec[kdim]["region"], "dim": kdim, "value": value, "_id": index_value}
            data.append(
                # pd.DataFrame(d) does not work because tuples is expanded into multiple rows:
                # pd.DataFrame({'v': (1, 2)})
                pd.DataFrame(d.values(), index=d.keys()).T
            )

        self._region_df = pd.concat((self._region_df, *data), ignore_index=True)

        self._edits.append({'operation':'insert', 'id':index_value})
        self._update_index()

    # def refresh_annotators(self, clear=False):
    #     for annotator in self._annotators.values():
    #         annotator.refresh(clear=clear)

    def _add_annotation_fields(self, index_value, fields=None):

        index_name_set = set() if self._field_df.index.name is None else set([self._field_df.index.name])
        unknown_kwargs = set(fields.keys()) - set(self._field_df.columns)
        if unknown_kwargs - index_name_set:
            raise KeyError(f'Unknown fields columns: {list(unknown_kwargs)}')

        new_fields = pd.DataFrame([dict(fields, **{self._field_df.index.name:index_value})])
        new_fields = new_fields.set_index(self._field_df.index.name)
        self._field_df =   pd.concat((self._field_df, new_fields))

    def delete_annotation(self, index):
        if index is None:
            raise ValueError('Deletion index cannot be None')
        self._region_df = self._region_df[self._region_df['_id'] != index] # Could match multiple rows
        self._field_df = self._field_df.drop(index, axis=0)

        self._edits.append({'operation':'delete', 'id':index})
        self._update_index()

    def update_annotation_fields(self, index, **fields):
        for column, value in fields.items():
            self._field_df.loc[index, column] = value

        self._edits.append({'operation':'update', 'id':index,
                            'fields' : [c for c in fields.keys()],
                            'region_fields' : []})

    # Methods to map into holoviews

    def _validate_index_to_fields(self, series):
        if series.index.name !=  self._field_df.index.name:
            raise ValueError(f'Index name {series.index.name} does not match fields index name {self._field_df.index.name}')
        if series.index.dtype != self._field_df.index.dtype:
            raise ValueError('Index dtype does not match fields index dtype')

    def _assert_indices_match(self, *series):
        if all(s.index is series[0].index for s in series):
            pass
        else:
            index_names = [s.index.name for s in series]
            if not all(name == index_names[0] for name in index_names):
                raise ValueError(f'Index names do not match: {index_names}')
            # TODO: Match dtypes
            match_values = all(all(s.index == series[0].index) for s in series)
            if not match_values:
                raise ValueError('Indices do not match')

        # for s in series:
        #     self._validate_index_to_fields(s)


    def define_fields(self, fields_df, index_mapping):
        # Need a staging area to hold everything till initialized
        self._index_mapping.update(index_mapping)  # Rename _field_df
        self._field_df = pd.concat([self._field_df, fields_df])
        self._edits.append({'operation':'save', 'ids':list(fields_df.index)})

    def define_points(self, dims, posx, posy=None):
        """
        Points in 1- or 2-dimensional space.

        Both posx and posy expect a Series object with an index
        corresponding to the fields_df supplied in the constructor (if
        it was specified).
        """
        if isinstance(dims, str):
            dims = (dims,)
        if posy is None and len(dims)==2:
            raise ValueError('Two dimensions declared but data for only one specified')
        if posy is not None and len(dims)==1:
            raise ValueError('Only one dimensions declared but data for more than one specified.')

        if len(dims)==2:
            self._assert_indices_match(posx, posy)

        mismatches = [el for el in posx.index if self._index_mapping.get(el,el)
                      not in self._field_df.index]
        if any(mismatches):
            raise KeyError(f'Keys {mismatches} do not match any fields entries')

        dim2 = None if len(dims)==1 else dims[1]
        value = zip(posx, [None] * len(posx)) if len(dims)==1 else zip(posx, posy)
        additions = pd.DataFrame({"region_type":'Point',
                                  "dim1":dims[0],
                                  "dim2":dim2,
                                  "value":value,
                                  "_id":[self._index_mapping[ind] for ind in posx.index]})
        self._region_df = pd.concat((self._region_df, additions), ignore_index=True)

    def define_ranges(self, dims, startx, endx, starty=None, endy=None):
        if isinstance(dims, str):
            dims = (dims,)
        if len(dims)==2 and any([el is None for el in [starty, endy]]):
            raise ValueError('Two dimensions declared but insufficient data specified')
        if len(dims)==1 and (starty, endy) != (None, None):
            raise ValueError('Only one dimensions declared but data for more than one specified.')

        if len(dims)==1:
            self._assert_indices_match(startx, endx)
        else:
            self._assert_indices_match(startx, endx, starty, endy)

        mismatches = [el for el in startx.index if self._index_mapping.get(el,el)
                      not in self._field_df.index]
        if any(mismatches):
            raise KeyError(f'Keys {mismatches} do not match any fields entries')


        dim2 = None if len(dims)==1 else dims[1]
        value = zip(startx, endx) if len(dims)==1 else zip(startx, endx, starty, endy)
        additions = pd.DataFrame({"region_type":'Range',
                                  "dim1":dims[0],
                                  "dim2":dim2,
                                  "value":value,
                                  "_id":[self._index_mapping[ind] for ind in startx.index]})
        self._region_df = pd.concat((self._region_df, additions), ignore_index=True)
        self._update_index()

    def _collapse_region_df(self, columns: list[str] | None=None) -> pd.DataFrame:
        # TODO: Move columns filtering to the top!
        regions = self._region_df.groupby("dim")["region"].first()
        data = self._region_df.pivot(index="_id", columns="dim", values="value")

        if data.empty:
            return data

        all_columns = list(data.columns)
        dims = columns or all_columns
        for dim in dims:
            region = regions.get(dim)
            if region is None:
                continue
            elif region == "range":
                na_mask = data[dim].isnull()
                data.loc[na_mask, dim] = data.loc[na_mask, dim].apply(lambda *x: (None, None))
                data[[f"start[{dim}]", f"end[{dim}]"]] = list(data[dim])
            else:
                data[f"{region}[{dim}]"] = data[dim].infer_objects()

        # Clean up
        data = data.drop(all_columns, axis=1)
        data.index.name = None
        data.columns.name = None
        return data

    @property
    def dataframe(self) -> pd.DataFrame:
        field_df = self._field_df
        region_df = self._collapse_region_df()

        df = pd.merge(region_df, field_df, left_index=True, right_index=True)
        df.index.name = self._field_df.index.name
        df = df.reindex(field_df.index)
        return df

    def _filter(self, dim_mask, region_type):
        region_mask = self._region_df["region"] == region_type
        return self._region_df[region_mask & dim_mask]

    def _mask1D(self, kdims):
        return self._region_df["dim"] == str(kdims[0])

    def _mask2D(self, kdims):
        dim1_name, dim2_name = str(kdims[0]), str(kdims[1])
        return np.logical_and(
            self._region_df["dim1"] == dim1_name, self._region_df["dim2"] == dim2_name
        )

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
        df = conn.transforms['load'](conn.load_dataframe())

        # Load fields dataframe
        fields_df = df[fields].copy()
        self.define_fields(fields_df, {ind:ind for ind in fields_df.index})
        # Replace: self._field_df = pd.concat([self._field_df, fields_df])

        # Load region dataframe
        if df.empty:
            self._region_df = pd.DataFrame(columns=self.columns)
            return

        data = []
        for kdim, info in spec.items():
            region = info["region"]
            if region == "range":
                value = list(zip(df[f"start_{kdim}"], df[f"end_{kdim}"]))
            else:
                value = df[f"{region}_{kdim}"]

            d = {"region": region, "dim": kdim, "value": value, "_id": list(df.index)}
            data.append(pd.DataFrame(d))

        rdf = pd.concat(data, ignore_index=True)
        empty_mask = (rdf.value == (None, None)) | rdf.value.isnull()
        self._region_df = rdf[~empty_mask].copy()

        self._update_index()
        self.clear_edits()

    def add_schema_to_conn(self, conn: Connector) -> None:
        field_dtypes = {col: str for col in conn.fields} # FIXME - generalize
        all_region_types = [{v["region"] for v in an.spec.values()} for an in self._annotators.values()]
        all_kdim_dtypes = [{k: v["type"] for k, v in an.spec.items()} for an in self._annotators.values()]
        schema = conn.generate_schema(conn.primary_key, all_region_types, all_kdim_dtypes, field_dtypes)
        conn.add_schema(schema)
