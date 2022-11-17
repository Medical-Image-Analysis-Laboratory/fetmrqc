from bs4 import BeautifulSoup as bs
from pathlib import Path


def get_html_index(folder):
    index_list = [
        f
        for f in Path(folder).iterdir()
        if f.is_file() and f.suffix == ".html" and "index" not in f.name
    ]
    return index_list


def add_message_to_reports(out_folder, index_list):
    index_list = get_html_index(out_folder)
    for file in index_list:
        # Parse HTML file in Beautiful Soup
        soup = bs(open(file), "html.parser")
        out = soup.find("script", type="text/javascript")
        in_str = out.string[1:]
        nspaces = len(in_str) - len(in_str.lstrip())
        newline = "\n" + " " * nspaces
        script_func = f"{newline}$('#btn-download').click(function () {{{newline}    window.parent.postMessage({{'message': 'rating done'}}, '*');{newline}}});{newline}"
        out.string = script_func + out.string
        with open(file, "w", encoding="utf-8") as f_output:
            f_output.write(str(soup))
