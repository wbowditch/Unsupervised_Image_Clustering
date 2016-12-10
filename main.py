import sys

def main(argv):
    print "hello world"
    for args in argv:
        print argv

if __name__ == '__main__':
    main(sys.argv)