
import io
import json


def write_log(path, logs):
    with io.open(path, 'a', encoding="utf8", newline="\n") as tgt:
        print(json.dumps(logs), file=tgt)
