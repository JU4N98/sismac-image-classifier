import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()