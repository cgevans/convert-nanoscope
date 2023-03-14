import pytest
import convert_nanoscope.spm as spm
from convert_nanoscope.image import proc_spm


def test_load_file():
    f = spm.NanoscopeFile("tests/files/basic-test.spm")

    assert len(f.images) == 4

    # process height data for first
    im0 = f.images[0].processed_data()

    # first image is height
    assert f.first_height_image() is f.images[0]

    assert im0.shape == (1024, 1024)


def test_proc():
    proc_spm("tests/files/basic-test.spm")


def test_partial_image_file():
    f = spm.NanoscopeFile("tests/files/partial image.spm")

    assert len(f.images) == 4

    # process height data for first
    im0 = f.images[0].processed_data()

    assert im0.shape == (367, 1024)
