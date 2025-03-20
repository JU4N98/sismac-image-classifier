import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("inception", 7, 16, offset="(0,0,0)", to="(0,0,0)", caption="Inception", height=64, depth=64, width=32 ),
    to_Conv("flatten", 1, 1, offset="(3,0,0)", to="(inception-east)", caption="Aplanamiento", height=2, depth=64, width=2 ),
    to_connection("inception","flatten"),
    to_Conv("dense", 1, 1, offset="(3,0,0)", to="(flatten-east)", caption="Densa", height=2, depth=32, width=2 ),
    to_connection("flatten", "dense"),
    to_SoftMax("softmax", 1 ,"(3,0,0)", "(dense-east)", caption="SOFTMAX", height=2, depth=16, width=2),
    to_connection("dense", "softmax"),
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()