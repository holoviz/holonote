import os
from pathlib import Path

import param
from nbsite.shared_conf import *  # noqa: F403

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

html_css_files += ["css/custom.css"]

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
        "jupyterlitelink",
        "page-toc",
    ],
}

extensions += ["sphinx.ext.napoleon", "nbsite.gallery"]
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]

jupyterlite_url = "https://holoviz-dev.github.io/holonote-jupyterlite"

nbsite_gallery_conf = {
    "github_org": "holoviz",
    "github_project": "holonote",
    "jupyterlite_url": jupyterlite_url,
    "galleries": {},
}

templates_path += ["_templates"]

html_context.update(
    {
        "last_release": f"v{release}",
        "github_user": "holoviz",
        "github_repo": "holonote",
        "default_mode": "light",
        "jupyterlite_endpoint": jupyterlite_url,
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


def setup(app) -> None:
    try:
        from nbsite.paramdoc import param_formatter, param_skip

        app.connect("autodoc-process-docstring", param_formatter)
        app.connect("autodoc-skip-member", param_skip)
    except ImportError:
        print("no param_formatter (no param?)")

    nbbuild.setup(app)
    app.add_config_value("grid_item_link_domain", "", "html")
