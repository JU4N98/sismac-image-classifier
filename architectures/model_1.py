import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 7, 16, offset="(0,0,0)", to="(0,0,0)", caption="Conv2D", height=64, depth=64, width=2 ),
    to_Conv("conv2", 5, 32, offset="(3,0,0)", to="(conv1-east)", caption="Conv2D", height=60, depth=60, width=4 ),
    to_connection("conv1", "conv2"),
    to_Conv("conv3", 3, 64, offset="(3,0,0)", to="(conv2-east)", caption="Conv2D", height=56, depth=56, width=8 ),
    to_connection("conv2", "conv3"),
    to_Pool("pool1","(3,0,0)","(conv3-east)",caption="Agrupacion Max", height=52, depth=52, width=2),
    to_connection("conv3", "pool1"),
    to_Conv("flatten", 1, 1, offset="(3,0,0)", to="(pool1-east)", caption="Aplanamiento", height=2, depth=64, width=2 ),
    to_connection("pool1", "flatten"),
    to_Conv("dense1", 1, 1, offset="(3,0,0)", to="(flatten-east)", caption="Densa", height=2, depth=32, width=2 ),
    to_connection("flatten", "dense1"),
    to_SoftMax("sigmoid", 1 ,"(3,0,0)", "(dense1-east)", caption="SIGMOIDE", height=2, depth=16, width=2),
    to_connection("dense1", "sigmoid"),
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()