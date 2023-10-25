from .data.config import IndexTemplate
from pathlib import Path
from .utils import get_html_index, add_message_to_reports
import os


def index_html(
    out_folder,
    index_list,
    navigate,
):
    """Given a folder containing reports, list all html files and generates an index."""

    out_folder = Path(out_folder)
    index_list = [str(f.relative_to(out_folder)) for f in index_list]
    _config = {
        "index_list": index_list,
        "navigate": navigate,
    }
    out_file = out_folder / "index.html"
    tpl = IndexTemplate()
    tpl.generate_conf(_config, out_file)
    return out_file


def list_out_folders(out_folders):
    """Construct the list of all folders that need
    an index file. Two types of folders can be given as input:
    1. Folders with reports
    2. Folders containing various splits of reports (constructed
        from the randomization step)
    The function returns a list of folders potentially containing reports.
    """
    if not isinstance(out_folders, list):
        out_folders = [out_folders]
    out_folder_list = []
    for folder in out_folders:
        out_folder_list += [Path(x[0]) for x in os.walk(folder)]
    return out_folder_list


def generate_index(
    out_folders, add_script_to_reports, use_ordering_file, navigate
):
    """Generate index.html files in the given folders."""

    out_folder_list = list_out_folders(out_folders)
    out_dict = {}
    for out_folder in out_folder_list:
        index_list = get_html_index(out_folder, use_ordering_file)
        if add_script_to_reports:
            add_message_to_reports(index_list)
        if len(index_list) > 0:
            out = index_html(
                out_folder=out_folder,
                index_list=index_list,
                navigate=navigate,
            )
            out_dict[out_folder] = out
            print(f"Index successfully generated in folder {out_folder}.")
    return out_dict
