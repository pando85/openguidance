
import nose

from nose.tools import *

import os.path
import re

from guidancelib import pathprovider


def test_abs_path():
    relpath = 'relpath'
    abspath = os.path.abspath(relpath)
    ok_(pathprovider.abs_path(relpath).startswith(pathprovider.get_config_path()))
    eq_(abspath, pathprovider.abs_path(abspath))


if __name__ == '__main__':
    nose.runmodule()
