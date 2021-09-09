import argparse
from pathlib import Path
import sys

import bilby
import pandas as pd
import tqdm


logger = bilby.core.utils.logger
logger.name = "kb"


def main():
    parser = argparse.ArgumentParser(
        description="Create a database from single pulse analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("directory", help="Directory containing json files")
    parser.add_argument(
        "-f", "--filename", default="kb_database.h5", help="Database filename"
    )
    parser.add_argument("-c", "--clean", action="store_true", help="Start clean")
    args = parser.parse_args()

    directory = Path(args.directory)

    # Read in existing database
    database_file = directory.joinpath(args.filename)
    if args.clean is False and database_file.exists():
        df = pd.read_hdf(database_file, "kb_database")
        logger.info(f"Read in database with {len(df)} records")
    else:
        cols = ["filename", "pulse_number", "n_shapelets", "base_flux_n_polynomial"]
        logger.info("Creating new database")
        df = pd.DataFrame(columns=cols)

    filenames = list(directory.glob("**/*result.json"))
    if len(filenames) == 0:
        raise ValueError("No results found to create/or add to database")
    else:
        logger.info(f"Found {len(filenames)} files to add to the database")

    # Get all filenames not already in the database
    filenames_to_read = []
    for filename in filenames:
        if str(filename) not in df.filename.values:
            filenames_to_read.append(filename)

    if len(filenames_to_read) == 0:
        logger.info("No new results to add to the database, exiting")
        sys.exit()

    logger.info(f"Adding {len(filenames_to_read)} files to the database")
    for ff in tqdm.tqdm(filenames_to_read):
        result = bilby.core.result.read_in_result(ff)
        data = dict(
            filename=str(ff),
            data_file=result.meta_data["args"]["data_file"],
            pulse_number=result.meta_data["args"]["pulse_number"],
            n_shapelets=str(result.meta_data["args"]["n_shapelets"]),
            base_flux_n_polynomial=result.meta_data["args"]["base_flux_n_polynomial"],
            maxl_normaltest_pvalue=result.meta_data["maxl_normaltest_pvalue"],
            log_evidence=result.log_evidence,
            log_evidence_err=result.log_evidence_err,
            log_noise_evidence=result.log_noise_evidence,
            log_bayes_factor=result.log_bayes_factor,
        )
        data.update(
            {
                f"{key}_median": val
                for key, val in dict(result.posterior.median()).items()
            }
        )
        data.update(
            {f"{key}_std": val for key, val in dict(result.posterior.std()).items()}
        )
        series = pd.Series(data, name=result.label)
        df = df.append(series)

    df["pulse_number"] = df["pulse_number"].map(int)
    df["data_file"] = df["data_file"].map(str)
    df["n_shapelets"] = df["n_shapelets"].map(str)
    df["base_flux_n_polynomial"] = df["base_flux_n_polynomial"].map(str)

    df.to_hdf(database_file, "kb_database", format="table")


if __name__ == "__main__":
    main()
