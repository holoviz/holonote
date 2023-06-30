import param
import sys
import weakref

import numpy as np
import pandas as pd

class AnnotationTable(param.Parameterized):
    """
    Class that stores and manipulates annotation data, including methods
    to declare annotations and commit edits back to the original data
    source such as a database.
    """
    columns = ("region_type", "dim1", "dim2", "value", "_id")

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

    def load(self, connector=None, fields_df=None, primary_key_name=None, fields=[]):
        """
        Load the AnnotationTable from a connector or a fields DataFrame.
        """
        if [connector, primary_key_name] == [None,None]:
            raise ValueError('Either a connector instance must be supplied or the primary key name supplied')
        if len(fields) < 1:
            raise ValueError('More than one field column is required')
        primary_key_name = primary_key_name if primary_key_name else connector.primary_key.field_name

        if fields_df:
            fields_df = fields_df[fields].copy() # Primary key/index for annotations
            self._field_df = fields_df
        elif connector and not connector.uninitialized:
            connector.load_annotation_table(self, fields)
        elif fields_df is None:
            fields_df = pd.DataFrame(columns=[primary_key_name] + fields)
            fields_df = fields_df.set_index(primary_key_name)
            self._field_df = fields_df

        # FIXME: Proper solution is to only load relevant columns
        self._field_df = self._field_df #.drop_duplicates(axis=1)
        self._region_df = self._region_df #.drop_duplicates(axis=1)

        self.clear_edits()
        self._update_index()

    def register_annotator(self, annotator):
        self._annotators[id(annotator)] = annotator


    # FIXME: Multiple region updates
    def update_annotation_region(self, index):
        region = list(self._annotators.values())[0]._region
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
    def has_snapshot(self):
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

    def _update_index(self):
        index_list = list(self._region_df['_id'])

        #for ind in self._deletions:
        #    if ind in index_list:
        #        index_list.remove(ind)
        self.index = list(set(index_list))


    def _expand_commit_by_id(self, id_val, fields=None, region_fields=None):
        kwargs = self._field_df.loc[[id_val]].to_dict('records')[0]
        if fields:
            kwargs = {k:v for k,v in kwargs.items() if k in fields}
        kwargs[self._field_df.index.name] = id_val
        if region_fields == []:
            return kwargs
        region_rows = self._region_df[self._region_df['_id'] == id_val]
        for _, region in region_rows.iterrows():
            # TODO: Handle region_fields. Use id? or uuid? or hash?
            # Or value + kdim + region_type?
            dims =  [el for el in [region['dim1'], region['dim2']] if el is not None]
            for dim in dims:
                if region['region_type'] == 'Range':
                    if len(dims)==1:
                        start, end = region['value']
                    elif dim==dims[0]:
                        start, end, _, _ = region['value']
                    elif dim==dims[1]:
                        _, _, start, end = region['value']
                    kwargs[f'start_{dim}'] = start
                    kwargs[f'end_{dim}'] = end
                elif region['region_type'] == 'Point':
                    point_dim1, point_dim2 = region['value']
                    if dim==dims[0]: # FIXME timestamp mapping
                        kwargs[f'point_{dim}'] = region['value'][0]
                    elif dim==dims[1]:
                        kwargs[f'point_{dim}'] = region['value'][1]
        return kwargs

    def _expand_save_commits(self, ids):
        return {'field_list':[self._expand_commit_by_id(id_val) for id_val in ids]}

    def commits(self):
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


    def clear_edits(self, edit_type=None):
        "Clear edit state and index mapping"
        self._edits = []
        self._index_mapping = {}

    def add_annotation(self, regions, **fields):
        "Takes a list of regions or the special value 'annotation-regions' to use attached annotators"
        index_value = fields.pop(self._field_df.index.name)
        if regions == 'annotator-regions':
            regions = [a._region for a in self._annotators.values()]
        if all(el=={} for el in regions):
            print('No new region selected. Skipping')
            return
        self._add_annotation_fields(index_value, fields=fields)

        for region in regions:
            if region == {}:
                continue

            region_type, dim1, dim2, value = (region['region_type'], region['dim1'],
                                              region['dim2'], region['value'])
            if region_type not in ['Range', 'Point']:
                raise Exception('TODO: Currently only supporting Range and Point annotations')

            new_region_row = pd.DataFrame([[region_type, dim1, dim2, value, index_value]], columns=self.columns)
            self._region_df = pd.concat((self._region_df, new_region_row), ignore_index=True)

        self._edits.append({'operation':'insert', 'id':index_value})
        self._update_index()

    def refresh_annotators(self, clear=False):
        for annotator in self._annotators.values():
            annotator.refresh(clear=clear)

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
            self._field_df.loc[index][column] = value

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
        value = zip(posx,  pd.Series([None for el in range(len(posx))])) if len(dims)==1 else zip(posx, posy)
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


    def _filter(self, dim_mask, region_type):
        region_mask = self._region_df["region_type"] == region_type
        return self._region_df[np.logical_and(region_mask, dim_mask)]

    def _mask1D(self, kdims):
        return np.logical_and(
            self._region_df["dim1"] == str(kdims[0]), self._region_df["dim2"].isnull()
        )

    def _mask2D(self, kdims):
        dim1_name, dim2_name = str(kdims[0]), str(kdims[1])
        return np.logical_and(
            self._region_df["dim1"] == dim1_name, self._region_df["dim2"] == dim2_name
        )

