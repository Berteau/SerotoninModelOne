#!/usr/bin/env python3

from absl import app
from absl import flags
from absl import logging
from base64 import b64encode
from numpy import arange
from params import loadParams
from string import Template

import os
import sys


"""
Ranges are iterated over to generate the appropriate flags files. All unique
combinations will be generated. See `flagfile`, `flagfile.template`, and
`params.py` for flags that are available. Edit this block when starting a new
swarm run.
"""
RANGES = {
    "maxTime": range(100, 200, 50),
    "tau": arange(.1, 1, .3),
    "popCount": range(10, 50, 10),
}


sys.setrecursionlimit(1000)
FLAGS = flags.FLAGS

flags.DEFINE_string("flags_template", "flagfile.template", "Template file for generating flags.")
flags.DEFINE_string("flags_directory", "flags", "Template file for generating flags.")

def load_template():
    with open(FLAGS.flags_template, "r") as flags_template:
        return Template(flags_template.read())


def build_table_matrix(ranges, accumulated = {}): 
    if not ranges:
        yield accumulated
        return

    for key in ranges:
        rang = ranges[key]
        for value in rang:
            next_accumulated = {**accumulated}
            next_accumulated[key] = value
            next_ranges = {**ranges}
            del next_ranges[key]
            yield from build_table_matrix(next_ranges, accumulated=next_accumulated)
        return


def write_template(template, build_id, default_params, matrix_params):
    params = {
        **default_params,
        **matrix_params,
    } # Merge param dictionaries

    params["build_id"] = build_id

    try:
        flags_file_contents = template.substitute(params)
        path = os.path.join(FLAGS.flags_directory, "flags.%s" % build_id)
        logging.info("Writing flags file %s" % path)
        with open(path, "w") as flags_file:
            flags_file.write(flags_file_contents)
    except Exception as err:
        logging.warn("Failed to write template for %s" % build_id)
        logging.info(err)


def encode_params(matrix_params):
    encoded_params = ''
    for key in matrix_params:
        encoded_params = "%s__%s_%s" % (encoded_params, key, matrix_params[key])
    return encoded_params[2:] # Strip leading __


def main(argv):
    if not os.path.exists(FLAGS.flags_directory):
        os.makedirs(FLAGS.flags_directory)

    template = load_template()

    default_params = loadParams()

    for matrix_params in build_table_matrix(RANGES):
        build_id = encode_params(matrix_params)
        write_template(template, build_id, default_params, matrix_params)


if __name__ == '__main__':
    app.run(main)
