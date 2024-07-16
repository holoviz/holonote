from collections import defaultdict

import numpy as np
import panel as pn
import param

pn.extension(
    "tabulator",
    css_files=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css"],
)


class AnnotatorTable(pn.viewable.Viewer):
    annotator = param.Parameter(allow_refs=False)
    tabulator = param.Parameter(allow_refs=False)
    dataframe = param.DataFrame()
    tabulator_kwargs = param.Dict(
        default={}, doc="""Configurations for the HoloViz Panel Tabulator widget""", precedence=-1
    )

    _updating = False

    def __init__(self, annotator, **params):
        super().__init__(annotator=annotator, **params)
        annotator.snapshot()
        self._create_tabulator()

    def _create_tabulator(self):
        def inner(event, annotator=self.annotator):
            return annotator.df

        def on_edit(event):
            row = self.tabulator.value.iloc[event.row]

            # Extracting specs and fields from row
            spec_dct, field_dct = defaultdict(list), {}
            for k, v in row.items():
                if "[" in k:
                    k = k.split("[")[1][:-1]  # Getting the spec name
                    spec_dct[k].append(v)
                else:
                    field_dct[k] = v

            self.annotator.annotation_table.update_annotation_region(spec_dct, row.name)
            self.annotator.update_annotation_fields(row.name, **field_dct)
            self.annotator.refresh(clear=True)

            # So it is still reactive, as editing overwrites the table
            self.tabulator.value = pn.bind(inner, self.annotator)

        def on_click(event):
            if event.column != "delete":
                return
            index = self.tabulator.value.iloc[event.row].name
            self.annotator.delete_annotation(index)

        def new_style(row):
            changed = [e["id"] for e in self.annotator.annotation_table._edits]
            color = "darkgray" if row.name in changed else "inherit"
            return [f"color: {color}"] * len(row)

        self.tabulator = pn.widgets.Tabulator(
            value=pn.bind(inner, self.annotator),
            buttons={"delete": '<i class="fa fa-trash"></i>'},
            show_index=False,
            selectable=True,
            **self.tabulator_kwargs,
        )
        self.tabulator.on_edit(on_edit)
        self.tabulator.on_click(on_click)
        self.tabulator.style.apply(new_style, axis=1)

        def on_commit(event):
            self.tabulator.param.trigger("value")
            # So it is still reactive, as triggering the value overwrites the table
            self.tabulator.value = pn.bind(inner, self.annotator)

        self.annotator.on_commit(on_commit)

    @param.depends("tabulator.selection", watch=True)
    def _select_table_to_plot(self):
        if self._updating:
            return
        try:
            self._updating = True
            self.annotator.selected_indices = list(
                self.tabulator.value.iloc[self.tabulator.selection].index
            )
        except IndexError:
            pass  # when we delete we select and get an index error if it is the last
        finally:
            self._updating = False

    @param.depends("annotator.selected_indices", watch=True)
    def _select_plot_to_table(self):
        if self._updating:
            return
        try:
            self._updating = True
            # Likely better way to get this mapping
            mask = self.tabulator.value.index.isin(self.annotator.selected_indices)
            self.tabulator.selection = list(map(int, np.where(mask)[0]))

        finally:
            self._updating = False

    def clear(self):
        self.tabulator.selection = []
        self.tabulator.param.trigger("value")

    def __panel__(self):
        return self.tabulator
