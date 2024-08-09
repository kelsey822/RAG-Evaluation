""" Reformats a .csv file of responses to multiple arguments of a command line.
"""

import csv

def string_format(query: str):
    """Takes a query from the queries.csv file and formats it into a command line argument.
    """
    query = query.strip()
    query = query.translate(str.maketrans("", "", "" + "''"))
    query = '"' + query + '"'
    return query



def parse(input_f: str):
    """Takes a csv file of queries and parses them into a single command line argument.
    """
    args = []
    #open the queries file to read
    with open(input_f, mode='r', newline="") as f_in:
        queries = csv.reader(f_in)
        for query in queries:
            query = ''.join(query)
            formatted = string_format(query)
            args.append(formatted)


    #remove the commas in the list
    delim = " "
    args = delim.join([ele for ele in args])
    return args

if __name__ == "__main__":
    input_f = "queries.csv"
    print(parse(input_f))


