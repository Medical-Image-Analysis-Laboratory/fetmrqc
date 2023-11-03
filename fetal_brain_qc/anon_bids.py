# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
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
import string
import csv
import random


def generate_random_ids(nreports: int, n: int = 5) -> list:
    """Generate a list of nreports unique
    combinations of n uppercase letters to anonymize
    the different reports that are considered.
    """
    id_list = []
    while len(id_list) < nreports:
        id_ = "".join(random.choice(string.ascii_uppercase) for i in range(n))
        if id_ not in id_list:
            id_list.append(id_)
    return id_list


def anonymize_bids_csv(bids_csv, out_bids_csv=None, seed=None):
    """Given a csv file listing the locations of LR series and corresponding
    masks, generates a list of anonymous IDs and adds them to the csv file.
    """
    file_list = []
    if not out_bids_csv:
        out_bids_csv = bids_csv.split(".")[0] + "_anon.csv"

    # Set the seed for the random number generator
    if seed:
        random.seed(seed)

    # Generate anonymous IDs
    anon_id = generate_random_ids(5000)

    # Read and anonymize the name in every row of bids_csv
    reader = csv.DictReader(open(bids_csv))
    for i, line in enumerate(reader):
        line["name"] = "sub-" + str(anon_id[i])
        file_list.append(line)
    # Write to the output bids file
    with open(out_bids_csv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=file_list[0].keys())
        writer.writeheader()
        for data in file_list:
            writer.writerow(data)
