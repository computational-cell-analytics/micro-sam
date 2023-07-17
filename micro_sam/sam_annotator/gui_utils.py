import os
from shutil import rmtree

from PyQt5 import QtCore, QtWidgets


def show_wrong_file_warning(file_path):
    """If the data signature does not match to the signature,
    the user can choose from the following options in this dialog:
       - Ignore: continue with input file (return file_path).
       - Overwrite: delete file_path and recompute the embeddings at same location.
       - Select a different file
       - Select a new file

    Arguments:
        file_path (string or os.path): path of the problematic file

    Returns:
        string or os.path: path to a file (new or old) depending on user decision
    """
    msgbox = QtWidgets.QMessageBox()
    msgbox.setWindowFlags(QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
    msgbox.setWindowTitle("Warning")
    msgbox.setText("The input data does not match the embeddings file.")
    ignore_btn = msgbox.addButton("Ignore", QtWidgets.QMessageBox.RejectRole)
    overwrite_btn = msgbox.addButton("Overwrite file", QtWidgets.QMessageBox.DestructiveRole)
    select_btn = msgbox.addButton("Select different file", QtWidgets.QMessageBox.AcceptRole)
    create_btn = msgbox.addButton("Create new file", QtWidgets.QMessageBox.AcceptRole)
    msgbox.setDefaultButton(create_btn)

    msgbox.exec()
    msgbox.clickedButton()
    if msgbox.clickedButton() == ignore_btn:
        return file_path
    elif msgbox.clickedButton() == overwrite_btn:
        rmtree(file_path)
        return file_path
    elif msgbox.clickedButton() == create_btn:
        # unfortunately there exists no dialog to create a directory so we have
        # to use "create new file" dialog with some adjustments.
        dialog = QtWidgets.QFileDialog(None)
        dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly)
        dialog.setNameFilter("Archives (*.zarr)")
        new_path = ""
        while os.path.splitext(new_path)[1] != ".zarr":
            dialog.exec()
            new_path = dialog.selectedFiles()[0]
        os.makedirs(new_path)
        return(new_path)
    elif msgbox.clickedButton() == select_btn:
        return QtWidgets.QFileDialog.getExistingDirectory(
            None, "Open a folder", os.path.split(file_path)[0], QtWidgets.QFileDialog.ShowDirsOnly
        )
