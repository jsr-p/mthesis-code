import dstnx
from dstnx import data_utils
from dstnx.data import psyk
from dstnx.features import utils as feat_utils

PSYK_SPEC = psyk.NON_SPEC_ABV
PSYK_ANY = ["any_psyk"]

EXTRA_FEATS = [
    "klasse_10",
    "klasse_11",
    "efterskole",
    "imm",
    "gpa",
    "female",
    "own_crimes",
]

PAR_CONDENSED = [
    "par_inc_avg",
    "par_edu_avg",
]

GROUP_FEATS = [
    # "KOM",
    "INSTNR"
]
COHORT_DUMMY = ["cohort"]
SES_Q_FEATS = [
    "SES_Q1",
    "SES_Q2",
    "SES_Q3",
    "SES_Q4",
]
SES_Q_DEC_FEATS = [
    "SES_Q10",
    "SES_Q20",
    "SES_Q30",
    "SES_Q40",
    "SES_Q50",
    "SES_Q60",
    "SES_Q70",
    "SES_Q80",
    "SES_Q90",
]
PEER_FEATS = ["all_ses_large", "all_ses_q99", "all_ses_q95", "all_ses_q90"]

EXTRA_SETS = [
    ["all_ses_small"],
    ["all_ses_large"],
    # []
]

targets = [
    "eu_grad",
    "gym_grad",
    "gs_grad",
    "eg_grad",
    "us_grad",
    "eu_apply",
    "gym_apply",
    "gs_apply",
    "eg_apply",
    "us_apply",
    "not_neet",
    "real_neet",
]

non_features = [
    "not_neet",
    "real_neet",
    "socio_edu",
    "socio_work",
    "socio_not",
    "beskst_work",
    "beskst_not",
    "PERSON_ID",
    # Drop targets
    "eu_grad",
    "gym_grad",
    "eu_apply",
    "eg_apply",
    "gs_grad",
    "gs_apply",
    "eg_grad",
    "gym_apply",
    "us_grad",
    "us_apply",
]
_TARGETS = [
    "gym_apply",
    "gym_grad",
    "eu_apply",
    "eu_grad",
    "us_apply",
    "us_grad",
    "real_neet",
]
TARGETS = _TARGETS + [f"{target}_y20" for target in _TARGETS]


class FeatureSets:
    FEATURES = EXTRA_FEATS + SES_Q_FEATS + PSYK_SPEC

    def __init__(self, feature_suffix: str):
        self.feature_suffix = feature_suffix
        self.weighted_cols = data_utils.load_json(
            dstnx.fp.DATA / "feature-columns" / f"columns_w-{self.feature_suffix}.json"
        )
        self.normal_cols = data_utils.load_json(
            dstnx.fp.DATA / "feature-columns" / f"columns-{self.feature_suffix}.json"
        )

    def get(self, fset: str):
        match fset:  # NOTE: All cases should have an underscore and not dash
            case "all":
                feat_cols = (
                    feat_utils.cols_picker(self.weighted_cols, "all")
                    + ["all_ses_large"]
                    + SES_Q_FEATS
                    + PSYK_ANY
                    + EXTRA_FEATS
                )
            case "all_excfamily":
                feat_cols = (
                    feat_utils.cols_picker(
                        self.weighted_cols, "all", exclude_family=True
                    )
                    + ["all_ses_large"]
                    + PSYK_ANY
                    + EXTRA_FEATS
                )
            case "all_excneighborhood":
                feat_cols = (
                    feat_utils.cols_picker(
                        self.weighted_cols,
                        "all",
                        exclude_adults=True,
                        exclude_youth=True,
                    )
                    + ["all_ses_large"]
                    + PSYK_ANY
                    + EXTRA_FEATS
                )
            case "all_exccontrols":
                feat_cols = feat_utils.cols_picker(self.weighted_cols, "all") + [
                    "all_ses_large"
                ]
            case "fam_only":
                feat_cols = feat_utils.cols_picker(
                        self.weighted_cols, "all", exclude_family=False, exclude_adults=True, exclude_youth=True
                    )
            case "neighborhood_only":
                feat_cols = feat_utils.cols_picker(
                        self.weighted_cols, "all", exclude_family=True, exclude_adults=False, exclude_youth=False
                    )
            case "controls_only":
                feat_cols = (
                    PSYK_ANY
                    + EXTRA_FEATS
                )
            case "gpa_only":
                feat_cols = ["gpa"]
            case "reduced":
                feat_cols = (
                    feat_utils.cols_picker(self.weighted_cols, "reduced")
                    + ["all_ses_large"]
                    + SES_Q_FEATS
                    + PSYK_ANY
                    + EXTRA_FEATS
                )
        return feat_cols


def get(feat_set: str, feature_suffix: str) -> list[str]:
    FEAT_SETS = FeatureSets(feature_suffix)
    return FEAT_SETS.get(feat_set)
