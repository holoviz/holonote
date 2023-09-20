from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import panel as pn
import param

if TYPE_CHECKING:
    from holonote.annotate import Annotator


class PanelWidgets:
    mapping = {
        str: pn.widgets.TextInput,
        bool: pn.widgets.Checkbox,
        dt.datetime: pn.widgets.DatePicker,
        dt.date: pn.widgets.DatePicker,
        int: pn.widgets.IntSlider,
        float: pn.widgets.FloatSlider,
    }

    def __init__(self, annotator: Annotator, field_values: dict[str, Any] | None=None):
        self.annotator = annotator
        self.annotator.snapshot()
        self._widget_mode_group = pn.widgets.RadioButtonGroup(
            name="Mode", options=["+", "-", "✏"], width=90
        )
        self._widget_apply_button = pn.widgets.Button(name="✓", width=20)
        self._widget_revert_button = pn.widgets.Button(name="↺", width=20)
        self._widget_commit_button = pn.widgets.Button(name="▲", width=20)

        if field_values is None:
            self._fields_values = {k: "" for k in self.annotator.fields}
        else:
            self._fields_values = {
                k: field_values.get(k, "") for k in self.annotator.fields
            }
        self._fields_widgets = self._create_fields_widgets(self._fields_values)

        self._set_standard_callbacks()

    @property
    def tool_widgets(self):
        return pn.Row(
            self._widget_apply_button,
            pn.Spacer(width=10),
            self._widget_mode_group,
            pn.Spacer(width=10),
            self._widget_revert_button,
            self._widget_commit_button,
        )

    def _create_fields_widgets(self, fields_values):
        fields_widgets = {}
        for widget_name, default in fields_values.items():
            if isinstance(default, param.Parameter):
                parameterized = type(
                    "widgets", (param.Parameterized,), {widget_name: default}
                )
                pane = pn.Param(parameterized)
                fields_widgets[widget_name] = pane.layout[1]
            elif isinstance(default, list):
                fields_widgets[widget_name] = pn.widgets.Select(
                    value=default[0], options=default, name=widget_name
                )
            else:
                widget_type = self.mapping[type(default)]
                if issubclass(widget_type, pn.widgets.TextInput):
                    fields_widgets[widget_name] = widget_type(
                        value=default, placeholder=widget_name, name=widget_name
                    )
                else:
                    fields_widgets[widget_name] = widget_type(
                        value=default, name=widget_name
                    )
        return fields_widgets

    @property
    def fields_widgets(self):
        accordion = False  # Experimental
        widgets = pn.Column(*self._fields_widgets.values())
        if accordion:
            return pn.Accordion(("fields", widgets))
        else:
            return widgets

    def _reset_fields_widgets(self):
        for widget_name, default in self._fields_values.items():
            if isinstance(default, param.Parameter):
                default = default.default
            try:
                self._fields_widgets[widget_name].value = default
            except Exception:
                pass  # TODO: Fix when lists (for categories, not the same as the default!)

    def _callback_apply(self, event):
        selected_ind = (
            self.annotator.selected_indices[0]
            if len(self.annotator.selected_indices) == 1
            else None
        )
        self.annotator.select_by_index()

        if self._widget_mode_group.value in ["+", "✏"]:
            fields_values = {k: v.value for k, v in self._fields_widgets.items()}
            if self._widget_mode_group.value == "+":
                self.annotator.add_annotation(**fields_values)
                self._reset_fields_widgets()
            elif (self._widget_mode_group.value == "✏") and (selected_ind is not None):
                self.annotator.update_annotation_fields(
                    selected_ind, **fields_values
                )  # TODO: Handle only changed
        elif self._widget_mode_group.value == "-":
            if selected_ind is not None:
                self.annotator.delete_annotation(selected_ind)

    def _callback_commit(self, event):
        self.annotator.commit()

    def _watcher_selected_indices(self, event):
        if len(event.new) != 1:
            return
        selected_index = event.new[0]
        # if self._widget_mode_group.value == '✏':
        for name, widget in self._fields_widgets.items():
            value = self.annotator.annotation_table._field_df.loc[selected_index][name]
            widget.value = value

    def _watcher_mode_group(self, event):
        if event.new in ["-", "✏"]:
            self.annotator.selection_enabled = True
            self.annotator.select_by_index()
            self.annotator.editable_enabled = False
        elif event.new == "+":
            self.annotator.editable_enabled = True
            self.annotator.select_by_index()
            self.annotator.selection_enabled = False

        for widget in self._fields_widgets.values():
            widget.disabled = event.new == "-"

    def _set_standard_callbacks(self):
        self._widget_apply_button.on_click(self._callback_apply)
        self._widget_revert_button.on_click(lambda event: self.annotator.revert_to_snapshot())
        self._widget_commit_button.on_click(self._callback_commit)
        self.annotator.param.watch(self._watcher_selected_indices, "selected_indices")
        self._widget_mode_group.param.watch(self._watcher_mode_group, "value")

    def __panel__(self):
        return pn.Column(self.fields_widgets, self.tool_widgets)
