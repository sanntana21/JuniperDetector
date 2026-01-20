import re

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else -1