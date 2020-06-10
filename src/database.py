import argparse
from pathlib import Path

import bilby
import pandas as pd
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-f", "--filename", default="kb_database.h5")
    parser.add_argument("-c", "--clean", action="store_true")
    args = parser.parse_args()

    directory = Path(args.directory)

    # Read in existing database
    database_file = directory.joinpath(args.filename)
    if args.clean is False and database_file.exists():
        df = pd.read_hdf(database_file, "kb_database")
    else:
        cols = ["filename", "pulse_number", "n_shapelets", "base_flux_n_polynomial"]
        df = pd.DataFrame(columns=cols)

    filenames = list(directory.glob('**/*result.json'))
    if len(filenames) == 0:
        raise ValueError("No results found to create database")

    # Get all filenames not already in the database
    filenames_to_read = []
    for filename in filenames:
        if filename not in df.filename:
            filenames_to_read.append(filename)

    for ff in tqdm.tqdm(filenames_to_read):
        result = bilby.core.result.read_in_result(ff)
        data = dict(
            filename=ff,
            data_file=result.meta_data["args"]["data_file"],
            pulse_number=result.meta_data["args"]["pulse_number"],
            n_shapelets=result.meta_data["args"]["n_shapelets"],
            base_flux_n_polynomial=result.meta_data["args"]["base_flux_n_polynomial"],
            maxl_normaltest_pvalue=result.meta_data["maxl_normaltest_pvalue"]
        )
        data.update({f"{key}_median": val for key, val in dict(result.posterior.median()).items()})
        data.update({f"{key}_std": val for key, val in dict(result.posterior.std()).items()})
        series = pd.Series(data, name=result.label)
        df = df.append(series)

    df.to_hdf(database_file, "kb_database")


if __name__ == "__main__":
    main()
