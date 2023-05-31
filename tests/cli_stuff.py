import click


@click.command(name="neighbors")
@click.option("--start", default=None, help="Batch size", type=int)
@click.option("--end", default=None, help="Batch size", type=int)
def main_multiple_periods(
    start: int,
    end: int,
):
    if not (start and end):
        start, end = 1, 2
    print(start, end)


main_multiple_periods()
