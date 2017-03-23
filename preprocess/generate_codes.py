from contextlib import ExitStack

A_STOCKS = './Data/A_stocks.txt'
A_CODES = './Data/A_codes.txt'

def generate_A_codes():
    with ExitStack() as stack:
        stocks = stack.enter_context(open(A_STOCKS, encoding='utf8'))
        codes = stack.enter_context(open(A_CODES, 'w'))
        for line in stocks:
            codes.write(line.split()[0][2:])
            codes.write('\n')

def main():
    generate_A_codes()

if __name__ == '__main__':
    pass
    # main()