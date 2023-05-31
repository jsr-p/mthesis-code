import click

from dstnx.data import school_siblings


@click.command()
@click.option("--start", default=1985, help="Start year")
@click.option("--end", default=2000, help="End year")
@click.option("--ninth-grade-only", default=False, help="Ninth grade only")
def main(start, end, ninth_grade_only):
    school_data = school_siblings.SchoolData(ninthgrade_only=ninth_grade_only)
    school_data.proc_all(start=start, end=end)


if __name__ == "__main__":
    main()
