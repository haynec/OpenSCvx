"""SEO fix-ups that need to reach files MkDocs config options cannot.

Two concerns live here, both stemming from pages/dates that plugins produce after the point where
static config (``exclude_docs``, the sitemap template) is resolved:

* ``scripts/gen_ref_pages.py`` and ``scripts/gen_example_pages.py`` emit ``Reference/SUMMARY.md``
  and ``Examples/SUMMARY.md`` via ``mkdocs-gen-files``. ``mkdocs-literate-nav`` consumes them to
  build the section navigation but leaves them in the file set, so MkDocs renders each as a real
  page (``.../SUMMARY/``) and lists it in ``sitemap.xml`` — a navigation artifact with no reader
  value. Because ``gen-files`` adds them at ``on_files`` time, ``exclude_docs`` (which only filters
  the ``docs_dir`` scan) never sees them; we drop them from the file set instead.

* ``mkdocs-git-revision-date-localized`` stamps each page's real git date into ``page.meta`` but
  not into ``page.update_date``, which is the field MkDocs' ``sitemap.xml`` reads for ``<lastmod>``.
  Left alone, every URL is stamped with the uniform build date. We copy the git date across so the
  sitemap carries a genuine per-page freshness signal.
"""

from mkdocs.plugins import event_priority
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page


@event_priority(-200)  # After mkdocs-literate-nav (-100) has consumed the SUMMARY files.
def on_files(files: Files, config) -> Files:
    for file in list(files):
        if file.src_uri.endswith("SUMMARY.md"):
            files.remove(file)
    return files


@event_priority(-100)  # After git-revision-date-localized has populated page.meta.
def on_page_markdown(markdown: str, page: Page, config, files) -> str:
    git_date = page.meta.get("git_revision_date_localized_raw_iso_date")
    if git_date:
        page.update_date = git_date
    return markdown
