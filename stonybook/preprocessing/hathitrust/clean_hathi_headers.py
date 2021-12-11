from stonybook.preprocessing.hathitrust.htrc.runningheaders import parse_page_structure
from stonybook.preprocessing.hathitrust.htrc.tests import load_vol

def htrc_cleaned_pages(book_dir):
    pages = parse_page_structure(load_vol(book_dir))

    formatted_pages = []
    for n, page in enumerate(pages):
        formatted_pages.append(
            (
                page.header if page.has_header else "",
                page.body if page.has_body else "",
                page.footer if page.has_footer else "",
            )
        )
    return formatted_pages