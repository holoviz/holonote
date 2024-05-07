from __future__ import annotations

import contextlib
import datetime as dt
from typing import TYPE_CHECKING, Any

import panel as pn
import param
from packaging.version import Version
from panel.viewable import Viewer

from ..annotate.display import _default_color

if TYPE_CHECKING:
    from holonote.annotate import Annotator

PN13 = Version(pn.__version__) >= Version("1.3.0")


class PanelWidgets(Viewer):
    mapping = {
        str: pn.widgets.TextInput,
        bool: pn.widgets.Checkbox,
        dt.datetime: pn.widgets.DatePicker,
        dt.date: pn.widgets.DatePicker,
        int: pn.widgets.IntSlider,
        float: pn.widgets.FloatSlider,
    }

    def __init__(
        self,
        annotator: Annotator,
        field_values: dict[str, Any] | None = None,
        as_popup: bool = False,
    ):
        self.annotator = annotator
        self.annotator.snapshot()
        self._widget_mode_group = pn.widgets.RadioButtonGroup(
            name="Mode", options=["+", "-", "✏"], width=90
        )
        self._widget_apply_button = pn.widgets.Button(name="✓", width=20)
        self._widget_revert_button = pn.widgets.Button(name="↺", width=20)
        self._widget_commit_button = pn.widgets.Button(name="▲", width=20)
        if PN13:
            self._add_button_description()

        if field_values is None:
            self._fields_values = {k: "" for k in self.annotator.fields}
        else:
            self._fields_values = {k: field_values.get(k, "") for k in self.annotator.fields}
        self._fields_widgets = self._create_fields_widgets(self._fields_values)
        self._create_visible_widget()

        self._set_standard_callbacks()

        self._layout = pn.Column(self.fields_widgets, self.tool_widgets)
        if self.visible_widget is not None:
            self._layout.insert(0, self.visible_widget)

        self._as_popup = as_popup
        if self._as_popup:
            self._layout.visible = False
            displays = self.annotator._displays
            if not displays:
                kdims = list(self.annotator.spec.keys())
                display = self.annotator.get_display(*kdims)
                display.indicators()
            for display in displays.values():
                if display.region_format in ("range", "range-range"):
                    stream = display._edit_streams[0]
                elif display.region_format in ("point", "point-point"):
                    stream = display._edit_streams[1]
                self._register_stream_popup(stream)
                self._register_tap_popup(display)
                self._register_double_tap_clear(display)

    def _create_visible_widget(self):
        if self.annotator.groupby is None:
            self.visible_widget = None
            return
        style = self.annotator.style
        if style.color is None and style._colormap is None:
            data = sorted(self.annotator.df[self.annotator.groupby].unique())
            colormap = dict(zip(data, _default_color))
        else:
            colormap = style._colormap
        if isinstance(colormap, dict):
            stylesheet = """
            option:after {
              content: "";
              width: 10px;
              height: 10px;
              position: absolute;
              border-radius: 50%;
              left: calc(100% - var(--design-unit, 4) * 2px - 3px);
              top: 20%;
              border: 1px solid black;
              opacity: 0.5;
            }"""
            for i, color in enumerate(colormap.values()):
                stylesheet += f"""
            option:nth-child({i + 1}):after {{
                background-color: {color};
            }}"""
        else:
            stylesheet = ""

        options = list(colormap)
        self.visible_widget = pn.widgets.MultiSelect(
            name="Visible",
            options=options,
            value=self.annotator.visible or options,
            stylesheets=[stylesheet],
        )
        self.annotator.visible = self.visible_widget

    def _add_button_description(self):
        from bokeh.models import Tooltip
        from bokeh.models.dom import HTML

        html_table = """
        <h3>Mode:</h3>
        <table style="width:100%">
        <tr>
            <td style="min-width:50px">+</td>
            <td>Add annotation</td>
        </tr>
        <tr>
            <td>-</td>
            <td>Delete annotation</td>
        </tr>
        <tr>
            <td>✏</td>
            <td>Edit annotation</td>
        </tr>
        </table>
        """

        self._widget_mode_group.description = Tooltip(
            content=HTML(html_table),
            position="bottom",
        )
        self._widget_apply_button.description = Tooltip(content="Apply changes", position="bottom")
        self._widget_revert_button.description = Tooltip(
            content="Revert changes not yet saved to database", position="bottom"
        )
        self._widget_commit_button.description = Tooltip(
            content="Save to database", position="bottom"
        )

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
                parameterized = type("widgets", (param.Parameterized,), {widget_name: default})
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
                    fields_widgets[widget_name] = widget_type(value=default, name=widget_name)
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
            if isinstance(default, list):
                default = default[0]
            with contextlib.suppress(Exception):
                # TODO: Fix when lists (for categories, not the same as the default!)
                self._fields_widgets[widget_name].value = default

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
        elif self._widget_mode_group.value == "-" and selected_ind is not None:
            self.annotator.delete_annotation(selected_ind)

        if self._as_popup:
            self._layout.visible = False

    def _register_stream_popup(self, stream):
        def _popup(*args, **kwargs):
            if self._layout.visible:
                self._layout.visible = False

            self._widget_mode_group.value = "+"
            self._layout.visible = True
            return self._layout

        stream.popup = _popup

    def _register_tap_popup(self, display):
        def tap_popup(x, y) -> None:  # Tap tool must be enabled on the element
            if self._layout.visible:
                return

            if self.annotator.selection_enabled:
                self._layout.visible = True
            return self._layout

        display._tap_stream.popup = tap_popup

    def _register_double_tap_clear(self, display):
        def double_tap_toggle(x, y):
            self._layout.visible = not self._layout.visible
            if self._layout.visible:
                return self._layout

        try:
            tools = display._element.opts["tools"]
        except KeyError:
            tools = []
        display._element.opts(tools=[*tools, "doubletap"])
        display._double_tap_stream.popup = double_tap_toggle

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
        with param.parameterized.batch_call_watchers(self):
            if event.new in ("-", "✏"):
                self.annotator.selection_enabled = True
            elif event.new == "+":
                self.annotator.editable_enabled = True
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
        return self._layout.clone(visible=True)
