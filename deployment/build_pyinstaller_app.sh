#!/bin/bash
rm -rf dist
rm -f annotator.py
echo "from micro_sam.sam_annotator.annotator import main" >> annotator.py
echo "main()" >> annotator.py
PYTHON_SITE=$(python -c "import site; print(''.join(site.getsitepackages()))")
echo $PYTHON_SITE
ls $PYTHON_SITE
pyinstaller --log-level=DEBUG \
--hidden-import=napari_svg \
--hidden-import=napari_plugin_engine \
--hidden-import=napari_console \
--hidden-import=napari_builtins \
--hidden-import=napari._event_loop \
--hidden-import=vispy.app.backends._pyqt5 \
--hidden-import=fontconfig \
--hidden-import=magicgui.backends._qtpy \
--hidden-import=imagecodecs._shared \
--hidden-import=torchvision.io \
--hidden-import=pytorch.jit \
--hidden-import=torchtriton \
--hidden-import=imagecodecs._imcd \
--hidden-import=jpeg \
--hidden-import=libnvjpeg \
--hidden-import=openjpeg \
--hidden-import=libpng \
--hidden-import=micro_sam \
--add-data "${PYTHON_SITE}/napari/resources/icons:napari/resources/icons" \
--add-data "${PYTHON_SITE}/napari/_qt/qt_resources/styles:napari/_qt/qt_resources/styles" \
--add-data "${PYTHON_SITE}/vispy/io/_data:vispy/io/_data" \
--add-data "${PYTHON_SITE}/vispy/util/fonts/data:vispy/util/fonts/data" \
--add-data "${PYTHON_SITE}/napari_builtins:napari_builtins" \
--add-data "${PYTHON_SITE}/PIL:PIL" \
--add-data "${PYTHON_SITE}/torchvision:torchvision" \
--add-data "${PYTHON_SITE}/torch:torch" \
--add-data "${PYTHON_SITE}/numpy:numpy" \
--add-data "${PYTHON_SITE}/vispy/glsl:vispy/glsl" annotator.py

mkdir ./dist/annotator/lib
ls ./dist/annotator
#cp ./dist/annotator/libiomp5.dylib ./dist/annotator/lib
cd ./dist/annotator
./annotator
