
SOURCE = '../data/dataset64_rough.dat'
DESTINATION = '../data/dataset64.dat'

def main():
    lines = []
    with open(SOURCE, mode="r") as fdataset:
        print('Restructuring...')
        lines = fdataset.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip().split(' ')
            lines[i] = [str.strip() for str in lines[i]]
            lines[i] = ';'.join(filter(lambda el : el != '', lines[i]))
            lines[i] += "\n"
        fdataset.close()
    with open(DESTINATION, mode="w") as fdataset:
        fdataset.writelines(lines)
        fdataset.close()
    print('finish')


# entry point
if __name__ == '__main__':
    main()
