#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
# pylint: disable=no-value-for-parameter

import unittest
import threading

from functools import partial  # noqa: F401
import tempfile  # noqa: F401
import os  # noqa: F401
import torch  # noqa: F401
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager


class TestTimestepManager(unittest.TestCase):
    """Test the TimestepManager class functionality."""

    def setUp(self):
        # Reset the context variable before each test
        TimestepManager._timestep_var.set(None)

    def test_get_set_timestep_idx(self):
        """Test setting and getting timestep index."""
        # Test initial value
        self.assertIsNone(TimestepManager.get_timestep_idx())

        # Test setting and getting a value
        TimestepManager.set_timestep_idx(5)
        self.assertEqual(TimestepManager.get_timestep_idx(), 5)

        # Test changing the value
        TimestepManager.set_timestep_idx(10)
        self.assertEqual(TimestepManager.get_timestep_idx(), 10)

    def test_thread_isolation(self):
        """Test that timestep indices are isolated between threads."""
        # Set timestep in main thread
        TimestepManager.set_timestep_idx(7)

        # Check timestep in another thread
        results = {}

        def check_in_thread():
            results['thread_value'] = TimestepManager.get_timestep_idx()
            # Set a different value in the thread
            TimestepManager.set_timestep_idx(8)

        thread = threading.Thread(target=check_in_thread)
        thread.start()
        thread.join()

        # The thread should initially see None, not the main thread's value
        self.assertIsNone(results['thread_value'])

        # The main thread's value should be unchanged
        self.assertEqual(TimestepManager.get_timestep_idx(), 7)

    def test_set_same_value(self):
        """Test setting the same timestep value consecutively."""
        # Set a value
        TimestepManager.set_timestep_idx(3)
        self.assertEqual(TimestepManager.get_timestep_idx(), 3)

        # Set the same value again
        TimestepManager.set_timestep_idx(3)
        self.assertEqual(TimestepManager.get_timestep_idx(), 3)

    def test_negative_timestep(self):
        """Test setting negative timestep values raises ValueError."""
        # Setting a negative timestep should raise ValueError
        with self.assertRaises(ValueError):
            TimestepManager.set_timestep_idx(-5)

    def test_none_timestep(self):
        with self.assertRaises(TypeError):
            TimestepManager.set_timestep_idx()

    def test_str_timestep(self):
        with self.assertRaises(ValueError):
            TimestepManager.set_timestep_idx("1")

    def test_zero_timestep(self):
        """Test setting timestep to zero."""
        TimestepManager.set_timestep_idx(0)
        self.assertEqual(TimestepManager.get_timestep_idx(), 0)

    def test_multiple_threads(self):
        """Test isolation across multiple threads running concurrently."""
        num_threads = 10
        results = [None] * num_threads

        def thread_func(thread_id):
            # Each thread sets its own timestep index
            TimestepManager.set_timestep_idx(thread_id)
            results[thread_id] = TimestepManager.get_timestep_idx()

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=thread_func, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify each thread had its own isolated value
        for i in range(num_threads):
            self.assertEqual(results[i], i)

    def test_non_int_timestep(self):
        """Test setting non-integer timestep values raises ValueError."""
        # Setting a float timestep should raise ValueError
        with self.assertRaises(ValueError):
            TimestepManager.set_timestep_idx(1.5)

        # Setting a string timestep should raise ValueError
        with self.assertRaises(ValueError):
            TimestepManager.set_timestep_idx("1")


if __name__ == "__main__":
    unittest.main()
