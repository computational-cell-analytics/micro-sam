import os
import magicgui
from shutil import rmtree
from typing import Union

from magicgui.widgets import Container
from magicgui.application import use_app

from PyQt5 import QtWidgets


def show_wrong_file_warning(file_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    """Show dialog if the data signature does not match the signature stored in the file.

    The user can choose from the following options in this dialog:
       - Ignore: continue with input file (return file_path).
       - Overwrite: delete file_path and recompute the embeddings at same location.
       - Select a different file
       - Select a new file

    Args:
        file_path: Path of the problematic file.

    Returns:
        Path to a file (new or old) depending on user decision
    """
#    q_app = QtWidgets.QApplication([])
#    msgbox = QtWidgets.QMessageBox()
#    msgbox.setWindowFlags(QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
#    msgbox.setWindowTitle("Warning")
#    msgbox.setText("The input data does not match the embeddings file.")
#    create_btn = msgbox.addButton("Create new file", QtWidgets.QMessageBox.AcceptRole)

    msg_box = None
    new_path = {"value": ""}

    @magicgui.magicgui(call_button="Ignore", labels=False)
    def _ignore():
        msg_box.close()
        new_path["value"] = file_path

    @magicgui.magicgui(call_button="Overwrite file", labels=False)
    def _overwrite():
        msg_box.close()
        rmtree(file_path)
        new_path["value"] = file_path

    @magicgui.magicgui(call_button="Create new file", labels=False)
    def _create():
        msg_box.close()
        # unfortunately there exists no dialog to create a directory so we have
        # to use "create new file" dialog with some adjustments.
        dialog = QtWidgets.QFileDialog(None)
        dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly)
        dialog.setNameFilter("Archives (*.zarr)")
        try_cnt = 0
        while os.path.splitext(new_path["value"])[1] != ".zarr":
            if try_cnt > 3:
                new_path["value"] = file_path
                return
            dialog.exec_()
            res = dialog.selectedFiles()
            new_path["value"] = res[0] if len(res) > 0 else ""
            try_cnt += 1
        os.makedirs(new_path["value"])

    @magicgui.magicgui(call_button="Select different file", labels=False)
    def _select():
        msg_box.close()
        try_cnt = 0
        while not os.path.exists(new_path["value"]):
            if try_cnt > 3:
                new_path["value"] = file_path
                return
            new_path["value"] = QtWidgets.QFileDialog.getExistingDirectory(
                None, "Open a folder", os.path.split(file_path)[0], QtWidgets.QFileDialog.ShowDirsOnly
            )
            try_cnt += 1

    msg_box = Container(widgets=[_select, _ignore, _overwrite, _create], layout='horizontal', labels=False)
    msg_box.root_native_widget.setWindowTitle("The input data does not match the embeddings file")
    msg_box.show(run=True)
    use_app().quit()
    return new_path["value"]
