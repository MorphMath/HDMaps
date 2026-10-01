project = "HDMaps"
author = "Felix Granum, Sofus Hesseldahl Laubel"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["pyvista", "cupy", "cupyx"]

html_theme = "alabaster"
