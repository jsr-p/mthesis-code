"""
Gather information on table to pick variables to choose.
"""
import dstnx
from dstnx import dst_tools


def collect(name: str = "KRAF"):
    dst_tables = dst_tools.DSTTables()
    dst_table = dst_tables.get_table(name)
    variables = dst_table.table_info.variabel.tolist()
    with open(dstnx.fp.DATA / "tableinfo" / f"{name}_vars.txt", "w") as file:
        for var in variables:
            file.write(f"{var}\n")
    dst_table.table_info[["variabel", "label"]].to_csv(
        dstnx.fp.DATA / "tableinfo" / f"{name}_table.csv", index=False
    )
    return dst_table


if __name__ == "__main__":
    collect("KRAF")
    collect("KRSI")
