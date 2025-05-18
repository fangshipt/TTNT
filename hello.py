import numpy as np
import os
from copy import deepcopy
from time import sleep

test_brd = np.zeros((8, 8), dtype='int8')
test_brd[0][1] = 1
test_brd[0][3] = 1
test_brd[2][2] = 1

def get_empty_line():
    line = '|'
    for c in range(8):
        line += ' '*3 + '|'
    return line

def put_queen_on_the_line(queen_position, line=''):
    if line == '':
        line = get_empty_line()

    pos_adj = 4*queen_position + 2

    return line[:pos_adj] + 'Q' + line[pos_adj+1:]

def print_board(board):
    print('     0   1   2   3   4   5   6   7')
    print('   +-------------------------------+')

    for r in range(8):
        new_line = get_empty_line()
        for c in range(8):
            if board[r][c] == 1:
                new_line = put_queen_on_the_line(c, new_line)
        new_line = f' {r} ' + new_line
        print(new_line)
        print('   +---+---+---+---+---+---+---+---+')

    # print('   +-------------------------------+')
    print('     a   b   c   d   e   f   g   h')

def clear():
    os.system('cls')

def is_possible(board, pos):
    r, c = pos
    # Check in column
    for i in range(8):
        if board[i][c] == 1:
            return False
    # Check in row
    for i in range(8):
        if board[r][i] == 1:
            return False
    # Check in diagonals
    # Getting upper left starting point
    r0 = r
    c0 = c
    while (r0!=0) and (c0!=0):
        r0 -= 1
        c0 -= 1
    # Checking to right and down
    while (r0!=8) and (c0!=8):
        if board[r0][c0] == 1:
            return False
        r0 += 1
        c0 += 1

    # Getting upper right starting point
    r0 = r
    c0 = c
    while (r0!=0) and (c0!=7):
        r0 -= 1
        c0 += 1
    # Checking to left and down
    while (r0!=8) and (c0!=-1):
        if board[r0][c0] == 1:
            return False
        r0 += 1
        c0 -= 1

    return True


SOLUTIONS = []

def solve_eight_queen_puzzle(board):
    for r in range(8):
        for c in range(8):
            if is_possible(board, (r, c)):
                board[r][c] = 1
                solve_eight_queen_puzzle(board)
                board[r][c] = 0

                if True:
                    clear()
                    print_board(board)
                    sleep(1)
                return

    SOLUTIONS.append(board.copy())
counter = 0
def solve(n, board, row):
    global counter
    if counter >= 113:
        return

    if row >= 8:
        counter += 1
        print(f'Solution #{counter}')
        print_board(board)
        print()
        SOLUTIONS.append(deepcopy(board))
        return


    for col in range(n):
        if is_possible(board, (row, col)):
            board[row][col] = 1
            clear()

            print(f'Solutions: {counter}')
            print_board(board)

            solve(n, board, row+1)
            board[row][col] = 0

            clear()
            print(f'Solutions: {counter}')
            print_board(board)


if __name__ == '__main__':
    # print_board(test_brd)
    # assert not is_possible(test_brd, (0, 0))
    # assert not is_possible(test_brd, (0, 1))
    # assert not is_possible(test_brd, (1, 1))
    # assert not is_possible(test_brd, (7, 2))
    # assert is_possible(test_brd, (7, 6))
    # assert not is_possible(test_brd, (2, 2))
    # assert not is_possible(test_brd, (7, 7))
    # assert not is_possible(test_brd, (1, 0))

    # print(is_possible(test_brd, (7, 5)))
    # print()
    # print()

    # test_2 = np.zeros((8, 8), dtype='int8')
    # test_2[3:5, 3:5] = np.array([[1, 1], [1, 1]])
    # print_board(test_2)

    # for r in range(8):
    #     for c in range(8):
    #         if is_possible(test_2, (r, c)):
    #             print(f'Pissoble loc is at {r} {c}')



    brd = np.zeros((8, 8), dtype='int8')
    clear()
    print('Watch and enjoy how the script solves the Eight Queen puzzle ;) ')
    print()
    print_board(brd)
    sleep(5)

    solve(8, brd, 0)
    clear()

    print(f'Num of solutions {len(SOLUTIONS)}')
    for i, item in enumerate(SOLUTIONS):
        print(f'Solution #{i+1}')
        print_board(item)
        print()








####

