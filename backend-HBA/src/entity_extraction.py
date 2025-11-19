import spacy
import re
from dateutil.parser import parse as parse_date
from datetime import datetime

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


# Define known room names (extend as needed)
KNOWN_ROOMS = {"LT1", "LT2", "MainHall", "Lab1", "Auditorium"}

DELETE_KEYWORDS = {"delete", "remove", "cancel", "unbook", "clear", "drop", "eliminate"}

def extract_time(text):
    """Extract time using regex and convert to HH:MM format."""
    time_matches = re.findall(r'\b\d{1,2}[:.]?\d{0,2}\s?(?:am|pm|AM|PM)?\b', text)
    times = []
    for t in time_matches:
        try:
            parsed = datetime.strptime(t.replace(".", ":"), "%I:%M %p") if ':' in t or '.' in t else datetime.strptime(t, "%I %p")
            times.append(parsed.strftime("%H:%M"))
        except:
            try:
                parsed = parse_date(t)
                times.append(parsed.strftime("%H:%M"))
            except:
                continue
    return times[:2]  # Return up to 2 times (start_time, end_time)

def extract_entities(text: str) -> dict:
    doc = nlp(text)
    entities = {}

    # Extract room name
    for token in doc:
        if token.text in KNOWN_ROOMS:
            entities["room_name"] = token.text

    # Extract date
    for ent in doc.ents:
        if ent.label_ == "DATE":
            try:
                parsed_date = parse_date(ent.text)
                entities["date"] = parsed_date.strftime("%Y-%m-%d")
                break
            except:
                continue

    # Extract start_time and end_time
    times = extract_time(text)
    if len(times) >= 1:
        entities["start_time"] = times[0]
    if len(times) >= 2:
        entities["end_time"] = times[1]

    return entities