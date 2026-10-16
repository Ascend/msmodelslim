# -*- coding: UTF-8 -*-
"""
Exclude the dag test directory: its cases are hard-marked with
@pytest.mark.skip() and provide no value when collected, while the
module-level import of mindspore adds significant collection time.
"""

collect_ignore = ["dag"]
