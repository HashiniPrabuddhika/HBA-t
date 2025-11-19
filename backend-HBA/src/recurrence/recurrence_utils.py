def build_rrule_from_extracted(data: dict) -> str:
    freq_map = {"daily": "DAILY", "weekly": "WEEKLY", "monthly": "MONTHLY"}
    freq = freq_map.get(data.get("frequency"))
    if not freq:
        raise ValueError("Invalid frequency")

    byday = ""
    if freq == "WEEKLY" and "days_of_week" in data:
        day_map = {
            "Monday": "MO",
            "Tuesday": "TU",
            "Wednesday": "WE",
            "Thursday": "TH",
            "Friday": "FR",
            "Saturday": "SA",
            "Sunday": "SU",
        }
        days = [day_map[d] for d in data["days_of_week"] if d in day_map]
        if days:
            byday = ";BYDAY=" + ",".join(days)

    rrule = f"FREQ={freq}{byday}"
    return rrule
