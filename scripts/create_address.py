import click

import dstnx
from dstnx.utils import address

from dstnx import log_utils

LOGGER = log_utils.get_logger(name=__name__)

@click.command()
@click.option("--suffix", default="")
def main(suffix: str):
    df = address.construct_address_data(suffix)
    filename = dstnx.fp.REG_DATA / f"address{suffix}.parquet"
    df.to_parquet(filename)
    LOGGER.info(f"Saved addresses to: {filename}")


if __name__ == "__main__":
    main()