import click

from dstnx.utils.address_sim import main_simulate


@click.command()
@click.option("--size", default=1000)
@click.option("--suffix", default="")
@click.option("--large", default=False, is_flag=True)
def main(size, suffix, large):
    main_simulate(size, suffix, large)


if __name__ == "__main__":
    main()
    # main_simulate(50_000, "_large", False)
