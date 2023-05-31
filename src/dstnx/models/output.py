def _by_coef(df):
    return df.assign(abs_coef=lambda df: df["Coef."].abs()).sort_values(
        by="abs_coef", ascending=False
    )


def _by_t_val(df):
    return df.assign(abs_t_table=lambda df: df["t"].abs()).sort_values(
        by="abs_t_table", ascending=False
    )


TRANS = {
    "coef": _by_coef,
    "t": _by_t_val,
}


def abs_sum2(mod, by: str = "t"):
    trans_fn = TRANS[by]
    return mod.summary2().tables[1].pipe(trans_fn)
