# doc/gen_api.py

import pathlib
import mkdocs_gen_files

# Excluded packages
EXCLUDED_PACKAGES = ["cellects.icons", "cellects.config", "cellects.paper_illustrations"]

# Root directory
SRC_ROOT = pathlib.Path("src")
PACKAGE_NAME = "cellects"
PACKAGE_ROOT = SRC_ROOT / PACKAGE_NAME

nav = mkdocs_gen_files.Nav()

# Collect only items directly under cellects/
main_items = {}  # {"core": "cellects/core.md", "gui": "cellects/gui/index.md", ...}

# Iterate over all .py files inside src/cellects
for path in sorted(PACKAGE_ROOT.rglob("*.py")):
    # Ignore __main__.py files
    if path.name == "__main__.py":
        continue

    # Skip root package __init__.py to avoid the useless "cellects" page
    if path == (PACKAGE_ROOT / "__init__.py"):
        continue

    # src/cellects/core.py -> ("cellects", "core")
    rel_path = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(rel_path.parts)

    # Ignore excluded packages or modules
    module_name = ".".join(parts)
    if any(module_name.startswith(prefix) for prefix in EXCLUDED_PACKAGES):
        continue

    # Special case for __init__.py: we want "cellects" or "cellects.subpkg",
    # not "cellects.__init__" which does not exist as a Python object.
    if parts[-1] == "__init__":
        module_parts = parts[:-1]  # e.g. ["cellects", "gui"]
        ident = ".".join(module_parts)  # "cellects.gui"

        # Where the generated markdown file lives in the built docs:
        # docs/api/cellects/gui/index.md
        file_doc_path = pathlib.Path("api", *module_parts, "index.md")

        # How the link should appear inside api/index.md:
        # "cellects/gui/index.md" (relative to docs/api/index.md)
        nav_doc_path = pathlib.Path(*module_parts, "index.md")
    else:
        module_parts = parts                     # e.g. ["cellects", "core"]
        ident = ".".join(module_parts)           # "cellects.core"

        # docs/api/cellects/core.md
        file_doc_path = pathlib.Path("api", *module_parts).with_suffix(".md")

        # link from api/index.md: "cellects/core.md"
        nav_doc_path = pathlib.Path(*module_parts).with_suffix(".md")

    # Register entry in the navigation tree (used to build api/index.md)
    nav[module_parts] = nav_doc_path.as_posix()

    # Collect top-level items under cellects/
    if len(module_parts) == 2 and module_parts[0] == PACKAGE_NAME:
        # store display name without "cellects."
        main_items[module_parts[1]] = nav_doc_path.as_posix()

    # Generate the module page
    with mkdocs_gen_files.open(file_doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n\n")
        fd.write(f"::: {ident}\n")

    # Optional: map generated doc to original source file
    mkdocs_gen_files.set_edit_path(file_doc_path, path)

# Generate the API index page
index_path = pathlib.Path("api", "index.md")
with mkdocs_gen_files.open(index_path, "w") as fd:
    fd.write("# API Reference\n\n")
    fd.write("## Main modules\n\n")
    for name in sorted(main_items):
        fd.write(f"- [`cellects.{name}`]({main_items[name]})\n")

# Generate the literate navigation file
summary_path = pathlib.Path("api", "SUMMARY.md")
with mkdocs_gen_files.open(summary_path, "w") as nav_file:
    nav_file.write("* [Overview](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())