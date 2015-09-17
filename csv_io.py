""" CSV input-output functions """

import csv
import numpy as np

COMMA_DELIM = {'delimiter': ',' , 'lineterminator' : '\n'}

######################## Output to CSV file ########################

def dump_stream_to_csv(iterable, csv_filename, csv_flavor = COMMA_DELIM):
    """ Write *iterable* stream into file *csv_filename*. """    
    with open(csv_filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile,  **csv_flavor)
        for row in iterable:        
             spamwriter.writerow(row)
    
def dump_list_to_csv(_list, csv_filename,  csv_flavor = COMMA_DELIM):
    dump_stream_to_csv(iter(_list), csv_filename, csv_flavor)

######################## Input from CSV file ########################

def yield_csv_rows(csv_filename, csv_flavor = COMMA_DELIM):
    """ Open *csv_filename* and return rows as iterable."""
    with open(csv_filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, **csv_flavor)
        for row in spamreader:
            yield row

def get_csv_rows_as_list(csv_filename, csv_flavor = COMMA_DELIM):
    return [x for x in yield_csv_rows(csv_filename, csv_flavor)]

def get_csv_rows_as_array(csv_filename, csv_flavor = COMMA_DELIM):
    return np.array(get_csv_rows_as_list(csv_filename, csv_flavor))

def get_csv_rows_as_matrix(csv_filename, csv_flavor = COMMA_DELIM):
    return np.matrix(get_csv_rows_as_array(csv_filename, csv_flavor), 
                     dtype = 'float64') 
                     