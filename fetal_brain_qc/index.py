from .data.config import IndexTemplate
from pathlib import Path
from .utils import get_html_index, add_message_to_reports


def index_html(
    out_folder,
    index_list,
):
    """Given a folder containing reports, list all html files and generates an index."""
    import os

    out_folder = Path(out_folder)
    index_list = [str(f.relative_to(out_folder)) for f in index_list]
    _config = {
        "index_list": index_list,
    }
    out_file = out_folder / "index.html"
    tpl = IndexTemplate()
    tpl.generate_conf(_config, out_file)
    return out_file


def generate_index(out_folder, add_script_to_reports, index_list=None):
    """TODO"""
    if index_list is None:
        index_list = get_html_index(out_folder)
    if add_script_to_reports:
        add_message_to_reports(out_folder, index_list)

    out = index_html(
        out_folder=out_folder,
        index_list=index_list,
    )
    print(f"Index successfully generated in folder {out_folder}.")
    return out
