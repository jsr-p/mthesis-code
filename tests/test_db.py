from dstnx import db


def test_global_tmp():
    dst_db = db.DSTDB()
    dst_db.reset_global_tmp_table(name="temp3")
    name = "temp3"
    dtypes = ["id INT", "description VARCHAR2(100)"]
    dst_db.create_global_tmp_table(name=name, dtypes=dtypes)
    rows = [
        (1, "First"),
        (2, "Second"),
        (3, "Third"),
        (4, "Fourth"),
        (5, "Fifth"),
        (6, "Sixth"),
        (7, "Seventh"),
    ]
    dst_db.insert_global_tmp_table(
        "insert into temp3(id, description) values (:1, :2)", rows
    )
    df = dst_db.extract_data("select * from temp3")
    ids = [row[0] for row in rows]
    descs = [row[1] for row in rows]
    assert df["ID"].isin(ids).all()
    assert df["DESCRIPTION"].isin(descs).all()


if __name__ == "__main__":
    test_global_tmp()
