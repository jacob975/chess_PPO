import numpy as np

def createZobristHashingTable(size:tuple =(8,8,12)):
    '''
    Creates a Zobrist Hashing Table.
    '''
    return np.random.randint(0, 2**64, size, dtype=np.uint64)

def getZobristHashingValue(table:np.ndarray, board:np.ndarray):
    '''
    Returns the Zobrist Hashing Value of the board.
    '''
    return np.bitwise_xor.reduce(table[board])