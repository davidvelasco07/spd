import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

# nbsphinx reads notebooks from docs/notebooks/; keep notebooks/Introduction.ipynb
# as the single tracked source and mirror it here on each build.
_repo_root = Path(__file__).resolve().parent.parent
_intro_src = _repo_root / "notebooks" / "Introduction.ipynb"
_intro_dst = Path(__file__).resolve().parent / "notebooks" / "Introduction.ipynb"
if _intro_src.exists():
    _intro_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_intro_src, _intro_dst)

project = "spd"
author = "spd developers"
release = "0.1"

extensions = [
    "myst_nb",
    "sphinx_thebe",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath", "amsmath"]

nb_execution_mode = "off"
nb_merge_streams = True

thebe_config = {
    "repository_url": "https://github.com/davidvelasco07/spd",
    "repository_branch": "main",
    "selector": ".cell",
    "selector_input": "div.cell_input pre",
    "selector_output": ".cell_output",
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "spd"
html_theme_options = {
    "source_repository": "https://github.com/davidvelasco07/spd",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_baseurl = "https://davidvelasco07.github.io/spd/"
