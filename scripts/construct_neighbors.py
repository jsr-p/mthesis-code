import click

from dstnx.data import neighbors


@click.command()
@click.option("--start", default=1985, help="Start year")
@click.option("--end", default=2000, help="End year")
@click.option("--force", default=False, is_flag=True, help="Force query again")
@click.option("--geo-only", default=False, is_flag=True, help="End year")
def main(start, end, force, geo_only):
    for year in range(start, end + 1):
        neighbors.construct_geo(year, force, geo_only)


if __name__ == "__main__":
    main()
