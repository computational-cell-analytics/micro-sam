import menuinst
import shutil
import os

conda_dir = os.environ["CONDA_PREFIX"]
menu_dir = os.path.join(conda_dir, "Menu")
os.makedirs(menu_dir, exist_ok=True)
shutil.copy("../doc/logo.ico", menu_dir)
shutil.copy("./windows_menu.json", menu_dir)
menuinst.install(os.path.join(menu_dir, 'windows_menu.json'), prefix=conda_dir)
