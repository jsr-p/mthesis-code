def outcome_file_name_extra_suffix(max_age: int) -> str:
    match max_age:
        case 25:  # For edus
            return ""
        case 24:  # For jobs
            return ""
        case 21:  # For edus  (we query up until but not including 21)
            return "_y20"
        case 20:  # For jobs (we query up including 20)
            return "_y20"
        case _:
            raise ValueError
