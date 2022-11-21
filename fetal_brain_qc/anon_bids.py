import secrets
import string
import csv


def generate_random_ids(nreports: int, n: int = 5) -> list:
    """Generate a list of nreports unique
    combinations of n uppercase letters to anonymize
    the different reports that are considered.
    """
    id_list = []
    while len(id_list) < nreports:
        id_ = "".join(secrets.choice(string.ascii_uppercase) for i in range(n))
        if id_ not in id_list:
            id_list.append(id_)
    return id_list


def anonymize_bids_csv(bids_csv, out_bids_csv=None):
    """Given a csv file listing the locations of LR series and corresponding
    masks, generates a list of anonymous IDs and adds them to the csv file.
    """
    file_list = []
    if not out_bids_csv:
        out_bids_csv = bids_csv.split(".")[0] + "_anon.csv"

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
