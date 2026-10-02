project = "HDMaps"
author = "Felix Granum, Sofus Hesseldahl Laubel"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

napoleon_google_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["cupy", "cupyx", "matplotlib", "mpl_toolkits"]

nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath"]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/MorphMath/HDMaps",
    "use_repository_button": True,
    "footer_content_items": ["copyright.html", "last-updated.html", "extra-footer.html"],
}
