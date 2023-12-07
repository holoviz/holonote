import os
from pathlib import Path

import param
from nbsite.shared_conf import *  # noqa: F403
from panel.io.convert import BOKEH_VERSION, PY_VERSION
from panel.io.resources import CDN_DIST

import holonote as hn

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False


project = "HoloNote"
authors = "HoloNote contributors"
copyright_years["start_year"] = "2023"
copyright = copyright_fmt.format(**copyright_years)
description = "Tools to create, edit and persist annotated regions for HoloViews"
version = release = base_version(hn.__version__)
is_dev = any(ext in version for ext in ("a", "b", "rc"))

HOLONOTE_ROOT = Path(hn.__file__).parents[1]
# Remove database files if they exist
for db in (HOLONOTE_ROOT / "examples").rglob("*.db"):
    db.unlink()


# For the interactivity warning box created by nbsite to point to the right
# git tag instead of the default i.e. main.
os.environ["BRANCH"] = f"v{release}"

html_static_path += ["_static"]

html_css_files += [
    "css/custom.css",
]

html_theme = "pydata_sphinx_theme"
# html_favicon = "_static/icons/favicon.ico"

html_theme_options = {
    # "logo": {
    #     "image_light": "_static/logo_horizontal_light_theme.png",
    #     "image_dark": "_static/logo_horizontal_dark_theme.png",
    # },
    "github_url": "https://github.com/holoviz/holonote",
    "icon_links": [
        {
            "name": "Discourse",
            "url": "https://discourse.holoviz.org/",
            "icon": "fa-brands fa-discourse",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/UXdtYyGVQX",
            "icon": "fa-brands fa-discord",
        },
    ],
    "pygment_light_style": "material",
    "pygment_dark_style": "material",
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": [
        "github-stars-button",
        # "jupyterlitelink",
        "page-toc",
    ],
}

extensions += ["sphinx.ext.napoleon", "nbsite.gallery", "sphinx_copybutton", "nbsite.pyodide"]
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]

# gallery_endpoint = "panel-gallery-dev" if is_dev else "panel-gallery"
# gallery_url = f"https://{gallery_endpoint}.pyviz.demo.anaconda.com"
jlite_url = (
    "https://pyviz-dev.github.io/jupyterlite-dev" if is_dev else "https://jupyterlite.holoviz.org"
)
# pyodide_url = (
#     "https://pyviz-dev.github.io/panel/pyodide" if is_dev else "https://panel.holoviz.org/pyodide"
# )

nbsite_gallery_conf = {
    "github_org": "holoviz",
    "github_project": "holonote",
    "galleries": {},
    #     "reference": {
    #         "title": "Component Gallery",
    #         "sections": [
    #             "panes",
    #             "widgets",
    #             "layouts",
    #             # 3 most important by expected usage. Rest alphabetically
    #             "chat",
    #             "global",
    #             "indicators",
    #             "templates",
    #         ],
    #         "titles": {
    #             "Vega": "Altair & Vega",
    #             "DeckGL": "PyDeck & Deck.gl",
    #             "ECharts": "PyEcharts & ECharts",
    #             "IPyWidget": "ipywidgets",
    #             "PanelCallbackHandler": "LangChain CallbackHandler",
    #         },
    #         "as_pyodide": True,
    #         "normalize_titles": False,
    #     }
    # },
    # "thumbnail_url": "https://assets.holoviz.org/panel/thumbnails",
    # "deployment_url": gallery_url,
    # "jupyterlite_url": jlite_url,
}

# if panel.__version__ != version and (PANEL_ROOT / "dist" / "wheels").is_dir():
#     py_version = panel.__version__.replace("-dirty", "")
#     panel_req = f"./wheels/panel-{py_version}-py3-none-any.whl"
#     bokeh_req = f"./wheels/bokeh-{BOKEH_VERSION}-py3-none-any.whl"
# else:
panel_req = f"{CDN_DIST}wheels/panel-{PY_VERSION}-py3-none-any.whl"
bokeh_req = f"{CDN_DIST}wheels/bokeh-{BOKEH_VERSION}-py3-none-any.whl"


# def get_requirements():
#     with open("pyodide_dependencies.json") as deps:
#         dependencies = json.load(deps)
#     requirements = {}
#     for src, deps in dependencies.items():
#         if deps is None:
#             continue
#         src = src.replace(".ipynb", "").replace(".md", "")
#         for name, min_version in MINIMUM_VERSIONS.items():
#             if any(name in req for req in deps):
#                 deps = [f"{name}>={min_version}" if name in req else req for req in deps]
#         requirements[src] = deps
#     return requirements


nbsite_pyodide_conf = {
    "PYODIDE_URL": "https://cdn.jsdelivr.net/pyodide/v0.23.1/full/pyodide.js",
    "requirements": [bokeh_req, panel_req, "pyodide-http", "holonote"],
    # "requires": get_requirements(),
}

templates_path += ["_templates"]

html_context.update(
    {
        "last_release": f"v{release}",
        "github_user": "holoviz",
        "github_repo": "holonote",
        "default_mode": "light",
        # "jupyterlite_endpoint": jlite_url,
        # "gallery_url": gallery_url,
        # "pyodide_url": pyodide_url,
    }
)

nbbuild_patterns_to_take_along = ["simple.html", "*.json", "json_*"]

# Override the Sphinx default title that appends `documentation`
html_title = f"{project} v{version}"


# Patching GridItemCardDirective to be able to substitute the domain name
# in the link option.
# from sphinx_design.cards import CardDirective
# from sphinx_design.grids import GridItemCardDirective

# orig_grid_run = GridItemCardDirective.run


# def patched_grid_run(self):
#     app = self.state.document.settings.env.app
#     existing_link = self.options.get("link")
#     domain = getattr(app.config, "grid_item_link_domain", None)
#     if self.has_content:
#         self.content.replace("|gallery-endpoint|", domain)
#     if existing_link and domain:
#         new_link = existing_link.replace("|gallery-endpoint|", domain)
#         self.options["link"] = new_link
#     return list(orig_grid_run(self))


# GridItemCardDirective.run = patched_grid_run

# orig_card_run = CardDirective.run


# def patched_card_run(self):
#     app = self.state.document.settings.env.app
#     existing_link = self.options.get("link")
#     domain = getattr(app.config, "grid_item_link_domain", None)
#     if existing_link and domain:
#         new_link = existing_link.replace("|gallery-endpoint|", domain)
#         self.options["link"] = new_link
#     return orig_card_run(self)


# CardDirective.run = patched_card_run


# def _get_pyodide_version():
#     if PYODIDE_VERSION.startswith("v"):
#         return PYODIDE_VERSION[1:]
#     msg = f"{PYODIDE_VERSION=} is not valid"
#     raise NotImplementedError(msg)


# def update_versions(app, docname, source):
#     # Inspired by: https://stackoverflow.com/questions/8821511
#     version_replace = {
#         "{{PANEL_VERSION}}": PY_VERSION,
#         "{{BOKEH_VERSION}}": BOKEH_VERSION,
#         "{{PYSCRIPT_VERSION}}": PYSCRIPT_VERSION,
#         "{{PYODIDE_VERSION}}": _get_pyodide_version(),
#     }

#     for old, new in version_replace.items():
#         source[0] = source[0].replace(old, new)


def setup(app) -> None:
    try:
        from nbsite.paramdoc import param_formatter, param_skip

        app.connect("autodoc-process-docstring", param_formatter)
        app.connect("autodoc-skip-member", param_skip)
    except ImportError:
        print("no param_formatter (no param?)")

    # app.connect("source-read", update_versions)
    nbbuild.setup(app)
    app.add_config_value("grid_item_link_domain", "", "html")


# grid_item_link_domain = gallery_endpoint
