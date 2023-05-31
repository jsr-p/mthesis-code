from dstnx import dst_tools

HOEJKVALITETSVARIABLE = [
    "Sociale forhold, sundhed og retsvæsen",
    "Beskæftigelse",
    "Uddannelse",
    "Privatøkonomi",
    "Befolkning",
    "Virksomheder, virksomhedsøkonomi og udenrigshandel",
    "Ejendomme, Boliger og Biler",
    "Lønforhold",
    "Ledighed og beskæftigelsesforanstaltninger",
    "Arbejdssteder",
]


def test_times_cats():
    cats = dst_tools.get_hq_cats()
    assert sorted(cats) == sorted(HOEJKVALITETSVARIABLE)


if __name__ == "__main__":
    test_times_cats()
