import numpy as np
import panel as pn
import param

pn.extension("tabulator")


class AnnotatorTabulator(pn.viewable.Viewer):
    annotator = param.Parameter(allow_refs=False)
    tabulator = param.Parameter(allow_refs=False)
    dataframe = param.DataFrame()
    changed = param.List()

    _updating = False

    def __init__(self, annotator, **params):
        super().__init__(annotator=annotator, **params)
        annotator.snapshot()
        self._create_tabulator()

    def _create_tabulator(self):
        def inner(event, annotator=self.annotator):
            if event:
                self.changed.append(event.index)
            return annotator.df

        def on_edit(event):
            row = self.tabulator.value.iloc[event.row]
            self.changed.append(row.name)

            # Hard-coded for now
            TIME = tuple(row.iloc[:2])
            self.annotator._set_regions(TIME=TIME)
            self.annotator.update_annotation_region(row.name)
            self.annotator.update_annotation_fields(row.name, **dict(row.iloc[2:]))
            self.annotator.refresh(clear=True)

        def on_click(event):
            if event.column != "delete":
                return
            index = self.tabulator.value.iloc[event.row].name
            self.annotator.delete_annotation(index)

        def new_style(row):
            color = "darkgray" if row.name in self.changed else "inherit"
            return [f"color: {color}"] * len(row)

        self.tabulator = pn.widgets.Tabulator(
            value=pn.bind(inner, self.annotator),
            buttons={"delete": '<i class="fa fa-trash"></i>'},
            show_index=False,
            selectable=True,
        )
        self.tabulator.on_edit(on_edit)
        self.tabulator.on_click(on_click)
        self.tabulator.style.apply(new_style, axis=1)

    @param.depends("tabulator.selection", watch=True)
    def _select_table_to_plot(self):
        if self._updating:
            return
        try:
            self._updating = True
            self.annotator.selected_indices = list(
                self.tabulator.value.iloc[self.tabulator.selection].index
            )
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
        self.changed = []
        self.tabulator.selection = []
        self.tabulator.param.trigger("value")

    def __panel__(self):
        return self.tabulator
