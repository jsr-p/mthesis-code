import click
import seaborn as sns
import matplotlib.pyplot as plt

from dstnx.features import education
from dstnx import data_utils


@click.command()
@click.option("--suffix", default="", help="Suffix for data files")
def main(suffix: str):
    """Constructs a categorical variable for 9th grade school."""
    school = education.school_types("_new", full_info=True)
    gp = school.groupby(["INSTNR", "YEAR"]).PERSON_ID.count().sort_values()
    gp = school.merge(
        gp[gp > 2].to_frame("inst_year_count"), how="left", on=["INSTNR", "YEAR"]
    )
    ninthgrade = gp.query("audd_name in ['9. klasse', '9. klasse, efterskole']")
    ninthgrade_insts = (
        ninthgrade.loc[(ninthgrade.groupby("PERSON_ID").YEAR.idxmax())][
            ["PERSON_ID", "INSTNR", "YEAR"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    assert ninthgrade_insts.duplicated(subset=["PERSON_ID", "INSTNR"]).sum() == 0
    data_utils.log_save_pq(
        filename=f"inst9thgrade{suffix}",
        df=ninthgrade_insts,
    )
    fig, ax = plt.subplots()
    counts = ninthgrade_insts.groupby(["INSTNR", "YEAR"]).PERSON_ID.count()
    sns.histplot(counts, ax=ax)
    data_utils.log_save_fig(
        filename=f"instyear9thgradedist",
        fig=fig,
    )


if __name__ == "__main__":
    main()
