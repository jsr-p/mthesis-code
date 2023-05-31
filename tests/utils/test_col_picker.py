import re

import dstnx
from dstnx import data_utils
from dstnx.features import utils as feat_utils

RE_SES_VARS = re.compile(r"(inc_avg|inc_kont|crimes|highest_edu_pria|arblos)")
RE_ADULT_OTHER = re.compile(r"highest_(eu|gs)")
RE_SES = re.compile(r"(ses)")


if __name__ == "__main__":
    cols = data_utils.load_json(
        dstnx.fp.DATA / "feature-columns/columns-k30_defaultk_defaultk30.json"
    )
    reduced = feat_utils.cols_picker(cols, "reduced")
    all = feat_utils.cols_picker(cols, "all")
    print(all)
    print(reduced)
