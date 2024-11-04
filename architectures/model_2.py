import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 7, 16, offset="(0,0,0)", to="(0,0,0)", caption="Conv1D", height=2, depth=64, width=2 ),
    to_Conv("conv2", 5, 32, offset="(3,0,0)", to="(conv1-east)", caption="Conv1D", height=2, depth=60, width=4 ),
    to_connection("conv1", "conv2"),
    to_Conv("conv3", 3, 64, offset="(3,0,0)", to="(conv2-east)", caption="Conv1D", height=2, depth=56, width=8 ),
    to_connection("conv2", "conv3"),
    to_Conv("flatten", 1, 1, offset="(3,0,0)", to="(conv3-east)", caption="flatten", height=2, depth=64, width=2 ),
    to_connection("conv3", "flatten"),
    to_Conv("dense1", 1, 1, offset="(3,0,0)", to="(flatten-east)", caption="dense", height=2, depth=32, width=2 ),
    to_connection("flatten", "dense1"),
    to_SoftMax("sigmoid", 1 ,"(3,0,0)", "(dense1-east)", caption="SIGMOID", height=2, depth=16, width=2),
    to_connection("dense1", "sigmoid"),
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()