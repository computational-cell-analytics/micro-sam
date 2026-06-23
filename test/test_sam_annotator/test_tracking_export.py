import xml.etree.ElementTree as ET

import numpy as np
import pytest

from micro_sam.v1.multi_dimensional_segmentation import (
    export_tracking_result_to_ctc,
    export_tracking_result_to_geff,
    export_tracking_result_to_trackmate_xml,
)


def _tracking_result_with_division():
    segmentation = np.zeros((3, 8, 8), dtype="uint16")
    segmentation[0, 1:3, 1:3] = 1
    segmentation[1, 2:4, 1:3] = 1
    segmentation[2, 1:3, 4:6] = 2
    segmentation[2, 4:6, 4:6] = 3
    lineages = [{1: [2, 3], 2: [], 3: []}]
    return segmentation, lineages


def test_export_tracking_result_to_ctc(tmp_path):
    segmentation, lineages = _tracking_result_with_division()

    export_tracking_result_to_ctc(segmentation, lineages, tmp_path)

    track_file = tmp_path / "TRA" / "man_track.txt"
    assert track_file.exists()
    rows = [line.split() for line in track_file.read_text().splitlines()]
    assert rows == [["1", "0", "1", "0"], ["2", "2", "2", "1"], ["3", "2", "2", "1"]]
    assert (tmp_path / "TRA" / "man_track0000.tif").exists()
    assert (tmp_path / "SEG" / "man_seg0000.tif").exists()


def test_export_tracking_result_to_trackmate_xml(tmp_path):
    segmentation, lineages = _tracking_result_with_division()

    output_path = export_tracking_result_to_trackmate_xml(segmentation, lineages, tmp_path)

    assert output_path == tmp_path / "tracking_result.xml"
    root = ET.parse(output_path).getroot()
    assert root.tag == "TrackMate"

    spots = root.findall(".//Spot")
    assert len(spots) == 4
    assert {int(spot.attrib["FRAME"]) for spot in spots} == {0, 1, 2}
    assert {int(spot.attrib["TRACKLET_ID"]) for spot in spots} == {1, 2, 3}

    spot_ids = {spot.attrib["ID"] for spot in spots}
    edges = root.findall(".//Edge")
    assert len(edges) == 3
    assert all(edge.attrib["SPOT_SOURCE_ID"] in spot_ids for edge in edges)
    assert all(edge.attrib["SPOT_TARGET_ID"] in spot_ids for edge in edges)


def test_export_tracking_result_to_geff(tmp_path):
    pytest.importorskip("geff")
    pytest.importorskip("trackastra")
    import geff
    import zarr

    segmentation, lineages = _tracking_result_with_division()

    output_path = export_tracking_result_to_geff(segmentation, lineages, tmp_path)

    assert output_path == tmp_path / "tracking_result.zarr"
    root = zarr.open(output_path, mode="r")
    assert root["segmentation"].shape == segmentation.shape
    assert np.array_equal(root["segmentation"][:], segmentation)

    graph, _ = geff.read(output_path / "tracking_graph.geff")
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3


def test_trackmate_xml_can_be_converted_to_geff(tmp_path):
    pytest.importorskip("geff")
    from geff.convert import from_trackmate_xml_to_geff

    segmentation, lineages = _tracking_result_with_division()
    xml_path = export_tracking_result_to_trackmate_xml(segmentation, lineages, tmp_path)
    geff_path = tmp_path / "from_trackmate.geff"

    from_trackmate_xml_to_geff(xml_path, geff_path)

    assert geff_path.exists()


def test_ctc_export_can_be_converted_to_geff(tmp_path):
    pytest.importorskip("geff")
    from geff.convert import from_ctc_to_geff

    segmentation, lineages = _tracking_result_with_division()
    ctc_path = tmp_path / "ctc"
    export_tracking_result_to_ctc(segmentation, lineages, ctc_path)
    geff_path = tmp_path / "from_ctc.geff"

    from_ctc_to_geff(ctc_path / "TRA", geff_path)

    assert geff_path.exists()
