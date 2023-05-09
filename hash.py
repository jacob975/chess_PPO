import chess
import chess.engine
import hashlib
import hash
import numpy as np
import random
import time

# Zobrist hashing used to hash a chess FEN string
# FEN getting from chess.board.fen() is a string of the form:
# rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq
# where the first part is the board state and the second part is the turn
# and castling rights
class ZobristHasing():
    def __init__(self):
        self.piece_keys = np.zeros((12, 64), dtype=np.uint64)
        self.castling_keys = np.zeros((4), dtype=np.uint64)
        self.enpassant_keys = np.zeros((8), dtype=np.uint64)
        self.turn_key = np.uint64(0)
        self.init_keys()
    def init_keys(self):
        for i in range(12):
            for j in range(64):
                self.piece_keys[i][j] = np.uint64(random.getrandbits(64))
        for i in range(4):
            self.castling_keys[i] = np.uint64(random.getrandbits(64))
        for i in range(8):
            self.enpassant_keys[i] = np.uint64(random.getrandbits(64))
        self.turn_key = np.uint64(random.getrandbits(64))
    def hash(self, board):
        h = np.uint64(0)
        for i in range(64):
            if board.piece_at(i) != None:
                piece = board.piece_at(i)
                piece_type = piece.piece_type
                piece_color = piece.color
                piece_index = piece_type - 1 + 6 * piece_color
                h ^= self.piece_keys[piece_index][i]
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self.castling_keys[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self.castling_keys[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self.castling_keys[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self.castling_keys[3]
        if board.ep_square != None:
            h ^= self.enpassant_keys[board.ep_square % 8]
        if board.turn == chess.WHITE:
            h ^= self.turn_key
        return h
    def hash_str(self, board):
        return str(self.hash(board))