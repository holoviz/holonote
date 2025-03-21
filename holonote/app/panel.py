from __future__ import annotations

import contextlib
import datetime as dt
from typing import TYPE_CHECKING, Any

import holoviews as hv
import panel as pn
import param
from packaging.version import Version
from panel.viewable import Viewer

from ..annotate.display import _default_color

if TYPE_CHECKING:
    from holonote.annotate import Annotator

PN13 = Version(pn.__version__) >= Version("1.3.0")


class PanelWidgets(Viewer):
    reset_on_apply = param.Boolean(default=True, doc="Reset fields widgets on apply")

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
        **params,
    ):
        super().__init__(**params)
        self._layouts = {}
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
            self._fields_values = dict.fromkeys(self.annotator.fields, "")
        else:
            self._fields_values = {k: field_values.get(k, "") for k in self.annotator.fields}
        self._fields_widgets = self._create_fields_widgets(self._fields_values)
        self._create_visible_widget()
        self.annotator.on_event(self._update_visible_widget)

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
        self.colormap = {}
        options = sorted(set(self.annotator.df[self.annotator.groupby].unique()))
        # if all_options:
        if style.color is None:
            self.colormap = dict(zip(options, _default_color))
        elif isinstance(style.color, str):
            self.colormap = dict(zip(options, [style.color] * len(options)))
        elif isinstance(style.color, hv.dim):
            self.colormap = self.annotator.style.color.ops[0]["kwargs"]["categories"]
            # assign default to any options whose color is unspecified by the user
            for option in options:
                if option not in self.colormap:
                    self.colormap[option] = self.annotator.style.color.ops[0]["kwargs"]["default"]

        self._update_stylesheet()
        self.visible_widget = pn.widgets.MultiSelect(
            name="Visible",
            options=options,
            value=self.annotator.visible or options,
            stylesheets=[self.stylesheet],
        )
        self.annotator.visible = self.visible_widget

    def _update_stylesheet(self):
        self.stylesheet = """
        option {
        position: relative;
        padding-left: 20px;
        }

        option:after {
        content: "";
        width: 8px;
        height: 8px;
        position: absolute;
        border-radius: 50%;
        left: 5px;
        top: 50%; /* Align vertically */
        transform: translateY(-50%); /* Align vertically */
        border: 1px solid black;
        opacity: 0.60;
        }
        """
        for _, (option, color) in enumerate(sorted(self.colormap.items())):
            self.stylesheet += f"""
        option[value="{option}"]:after {{
            background-color: {color};
        }}"""

    def _update_visible_widget(self, event):
        style = self.annotator.style
        old_options = list(self.visible_widget.options)
        old_values = list(self.visible_widget.value)

        if event.type == "create":
            new_option = event.fields[self.annotator.groupby]
            if new_option not in old_options:
                self.visible_widget.param.update(
                    options=sorted([*old_options, new_option]), value=[*old_values, new_option]
                )
            if new_option not in self.colormap:
                if style.color is None:
                    # For now, we need to update the colormap so that
                    # the new sorted order of the keys matches the order of the default_colors
                    new_options = sorted(self.annotator.df[self.annotator.groupby].unique())
                    self.colormap = dict(zip(new_options, _default_color))
                elif isinstance(style.color, str):
                    self.colormap[new_option] = style.color
                elif isinstance(style.color, hv.dim):
                    self.colormap[new_option] = style.color.ops[0]["kwargs"]["default"]
                self._update_stylesheet()
                self.visible_widget.stylesheets = [self.stylesheet]
            return

        if event.type == "delete":
            new_options = sorted(self.annotator.df[self.annotator.groupby].unique())
            # if color was not user-specified, remake colormap in case an anno type was dropped
            if style.color is None:
                self.colormap = dict(zip(new_options, _default_color))
            self.visible_widget.options = list(new_options)
            self._update_stylesheet()
            self.visible_widget.stylesheets = [self.stylesheet]
            return

        if event.type == "update" and event.fields is not None:
            new_option = event.fields[self.annotator.groupby]
            new_options = sorted(self.annotator.df[self.annotator.groupby].unique())
            self.visible_widget.options = new_options
            # Make new vals visible, else inherit visible state
            if new_option not in old_options:
                self.visible_widget.value = [*old_values, new_option]
            if new_option not in self.colormap:
                if style.color is None:
                    # if the color was not user-specified, remake colormap for new anno type
                    self.colormap = dict(zip(new_options, _default_color))
                elif isinstance(style.color, str):
                    self.colormap[new_option] = style.color
                elif isinstance(style.color, hv.dim):
                    # if it's a new annot type but color dim had been specified by the user, it would already
                    # be in the colormap, so otherwise set the new anno type to the default color
                    self.colormap[new_option] = style.color.ops[0]["kwargs"]["default"]
                self._update_stylesheet()
                self.visible_widget.stylesheets = [self.stylesheet]

        if event.type == "update" and event.region is not None:
            return

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
                if self.reset_on_apply:
                    self._reset_fields_widgets()
            elif (self._widget_mode_group.value == "✏") and (selected_ind is not None):
                self.annotator.update_annotation_fields(
                    selected_ind, **fields_values
                )  # TODO: Handle only changed
        elif self._widget_mode_group.value == "-" and selected_ind is not None:
            self.annotator.delete_annotation(selected_ind)

    def _add_layout(self, name):
        """
        Add a layout to the panel, by cloning the root layout, linking the close button,
        and returning it visibly.
        """

        def close_layout(event):
            layout.visible = False

        layout = self._layouts.get(name)
        if not layout:
            layout = self._layout.clone(visible=True)
            self._widget_apply_button.on_click(close_layout)
            self._layouts[name] = layout
        return layout

    def _hide_layouts_except(self, desired_name) -> pn.Column:
        """
        Prevents multiple layouts from being visible at the same time.
        """
        desired_layout = None
        for name, layout in self._layouts.items():
            if name == desired_name:
                layout.visible = True
                desired_layout = layout
            elif name != "__panel__":
                layout.visible = False

        # If the desired layout is not found, create it
        if desired_name is not None and desired_layout is None:
            desired_layout = self._add_layout(desired_name)
        return desired_layout

    def _register_stream_popup(self, stream):
        def _popup(*args, **kwargs):
            # If the annotation widgets are laid out on the side in a Column/Row/etc,
            # while as_popup=True, do not show the popup during subtract or edit mode
            widgets_on_side = any(name == "__panel__" for name in self._layouts)
            if widgets_on_side and self._widget_mode_group.value in ("-", "✏"):
                return
            self._widget_mode_group.value = "+"
            return self._hide_layouts_except(stream.name)

        stream.popup = _popup

    def _register_tap_popup(self, display):
        def tap_popup(x, y) -> None:  # Tap tool must be enabled on the element
            if self.annotator.selected_indices:
                return self._hide_layouts_except("tap")

        display._tap_stream.popup = tap_popup

    def _register_double_tap_clear(self, display):
        def double_tap_toggle(x, y):
            # Toggle the visibility of the doubletap layout
            if any(layout.visible for layout in self._layouts.values()):
                # Clear all open layouts
                self._hide_layouts_except(None)
            else:
                # Open specifically the doubletap layout
                return self._hide_layouts_except("doubletap")

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
        for name, widget in self._fields_widgets.items():
            value = self.annotator.annotation_table._field_df.loc[selected_index][name]
            widget.value = value

    def _watcher_mode_group(self, event):
        with param.parameterized.batch_call_watchers(self):
            if event.new in ("-", "✏"):
                self.annotator.selection_enabled = True
                self.annotator.editable_enabled = False
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
        layout = self._layout.clone(visible=True)
        self._layouts["__panel__"] = layout
        return layout
