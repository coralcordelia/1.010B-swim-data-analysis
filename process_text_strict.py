
import numpy as np
import csv

raw_text = open('raw_text.txt', 'r').read()


columns = """Obs	Place	State	Gender	Age	Age2	AgeGrp	Year	Seed	Time	Diff	Split-1	Split-2	Split-3	Split-4	Split-5	Split-6	Split-7	Split-8	Split-9	Split-10"""

def process_entire_string(input_string):
    all_lines = input_string.split('\n')
    return [process_one_line(line) for line in all_lines]

def process_one_line(line):
    return line.split('\t')

def remove_spaces_one_line(input_string):
    doubles_gone = remove_doubles(input_string)
    return doubles_gone.split('\t')
    
def remove_doubles(input_string):
    if '  ' not in input_string:
        return input_string
    return remove_doubles(input_string.replace('  ', ' '))

col_list = remove_spaces_one_line(columns)
col_data = process_entire_string(raw_text)

data_list = [
    {
        col_list[index]: line[index]
        for index in range(len(col_list))
    } for line in col_data if '*' not in line
]



with open('swim_data_strict.csv', 'w') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=col_list)
    writer.writeheader()
    writer.writerows(data_list)
