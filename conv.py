'''
prev -> previous letter
b3v  -> the base-3 value
'''

def convert_value_sequence(prev:str, seq:str)->str:
    next_letter = lambda p, v: 'ACGT'[('ACGT'.find(p) + v + 1)%4]
    res = next_letter(prev, int(seq[0]))
    for k in seq[1:]:
        v = int(k)
        res += next_letter(res[-1], v)
    return res

# input 1 is 'A', 'T', 'C', 'G'
# input 2 is a sequence of 0,1,2 (base-3 value)
prev = input()
vals = input()
print(convert_value_sequence(prev, vals))

