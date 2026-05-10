"""Sphinx configuration for the SceneRep_for_TAMP documentation site."""
from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the source packages importable so autodoc can resolve them.
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DOCS_DIR)
sys.path.insert(0, REPO_ROOT)

# -- Project ----------------------------------------------------------

project = "Dynamic Scene Graph"
author = "Dynamic Scene Graph authors"
copyright = f"{datetime.now().year}, {author}"
release = "1.0.0"

# -- General ----------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
]

# Markdown + reST both accepted.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Pull docstrings rendered with both Google and NumPy styles.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# autodoc behaviour: keep signatures, fall back to __init__ for class docstrings.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    # Avoid double-documenting dataclass attributes — the class docstring
    # already lists them.
    "exclude-members": "__weakref__",
}
autoclass_content = "class"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True

# Many runtime deps (jax, gtsam, ROS, OWL/SAM2, pyrender) are heavy or
# optional; mock them so autodoc can import the modules without a full env.
autodoc_mock_imports = [
    "cv2",
    "open3d",
    "scipy",
    "scipy.optimize",
    "scipy.spatial",
    "scipy.spatial.transform",
    "jax",
    "jaxlib",
    "jax.numpy",
    "flax",
    "tensorflow",
    "torch",
    "torchvision",
    "PIL",
    "kornia",
    "trimesh",
    "pyrender",
    "rospy",
    "ros_numpy",
    "tf2_ros",
    "sensor_msgs",
    "geometry_msgs",
    "nav_msgs",
    "std_msgs",
    "cv_bridge",
    "rosbag",
    "scenic",
    "big_vision",
    "filterpy",
    "skimage",
    "gtsam",
    "segment_anything",
    "pupil_apriltags",
    "pyvista",
    "pyvistaqt",
    "open3d.t",
    "fastapi",
    "uvicorn",
]

# MyST options.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",   # $...$ inline math, $$...$$ display math
    "amsmath",      # \begin{align}…\end{align}
    "linkify",
    "smartquotes",
    "tasklist",
    "html_admonition",
]
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "macros": {
            "Ad": "\\operatorname{Ad}",
            "se": "\\mathfrak{se}",
            "SE": "\\operatorname{SE}",
            "SO": "\\operatorname{SO}",
        },
    },
}
myst_heading_anchors = 3
myst_url_schemes = ("http", "https", "mailto", "ftp")

# Cross-link to common upstream docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "open3d": ("http://www.open3d.org/docs/release/", None),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "ekf_tracker/latex",  # raw LaTeX sources, not Sphinx content
]

# -- HTML -------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "Dynamic Scene Graph"
html_short_title = "Dynamic Scene Graph"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#8a4a23",
        "color-brand-content": "#8a4a23",
        "color-background-primary": "#fafaf7",
        "color-background-secondary": "#f3f3ee",
        "color-foreground-primary": "#1f2328",
        "font-stack": "-apple-system, BlinkMacSystemFont, 'Segoe UI', "
                      "'Helvetica Neue', Arial, sans-serif",
        "font-stack--monospace": "'SF Mono', Menlo, Consolas, "
                                 "'DejaVu Sans Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#d29867",
        "color-brand-content": "#d29867",
    },
}

# -- Misc -------------------------------------------------------------

# Don't fail the build on warnings yet; flip to True once everything renders.
nitpicky = False
todo_include_todos = True

# The codebase's docstrings predate Sphinx and use Google/NumPy-ish style
# with embedded math (``{α - 2}``, ``|r|``) and ad-hoc indentation that
# trips docutils.  Suppress the noisy categories so the build output stays
# scannable.  The warnings are cosmetic — the rendered HTML is fine.
suppress_warnings = [
    "docutils",
    "myst.xref_missing",
    "myst.header",
    "ref.python",
    "autodoc",
    "autodoc.import_object",
    "misc.highlighting_failure",
    # Dataclass attributes get documented twice when the same module
    # appears via both ``automodule`` and a class autodoc; the rendered
    # output is identical and harmless.
    "app.add_directive",
]

# sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True
