from contextlib import ExitStack

def generate_A_codes():
    with ExitStack() as stack:
        stocks = stack.enter_context(open('./Data/A_stocks.txt', encoding= 'utf8'))
        codes = stack.enter_context(open('./Data/A_codes.txt', 'w'))
        for line in stocks:
            codes.write(line.split()[0][2:])
            codes.write('\n')

def main():
    generate_A_codes()

if __name__ == '__main__':
    main()
