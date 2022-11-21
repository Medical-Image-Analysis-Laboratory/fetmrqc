# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# STATEMENT OF CHANGES: This file is derived from sources licensed under the Apache-2.0 terms,
# and this file has been changed.
# The original file this work derives from is found at:
# https://github.com/nipreps/mriqc/blob/325ab434f7078701a401a6c1ecdc0a56fe0f640e/mriqc/data/__init__.py
#
# [November 2022] CHANGES:
#    * ENH: Added an IndexTemplate to create an index file for the visual reports.
#
# ORIGINAL WORK'S ATTRIBUTION NOTICE:
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#


"""MRIQC resources."""
from .config import IndexTemplate, IndividualTemplate, Template

__all__ = ["Template", "IndexTemplate", "IndividualTemplate"]
