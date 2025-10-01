# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

'''
html_theme is usually unchanged (rocm_docs_theme).
flavor defines the site header display, select the flavor for the corresponding portals
flavor options: rocm, rocm-docs-home, rocm-blogs, rocm-ds, instinct, ai-developer-hub, local, generic
'''
html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-docs-home"}

'''
docs_header_version is used to manually configure the version in the header. If
there exists a non-null value mapped to docs_header_version, then the header in
the documentation page will contain the given version string.
'''
html_context = {
    "docs_header_version": "3.15"
}

# This section turns on/off article info
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""


with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'.*\bset\(VERSION\s+\"?([0-9.]+)[^0-9.]+', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"rocAL {version_number} Documentation"

# for PDF output on Read the Docs
project = "rocAL Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"


doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}


extensions = [
    "rocm_docs", 
    "rocm_docs.doxygen",
] 

'''
docs_core = ROCmDocs(left_nav_title)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.enable_api_reference()
docs_core.setup()
'''

html_title = f"{project} {version_number} documentation"

external_projects_current_project = "rocal"

