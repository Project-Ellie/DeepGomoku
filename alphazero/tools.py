import numpy as np


def print_bin(*_args, **_kwargs):
    print("Deprecated. Use print_channels instead.")


def print_channels(binary_sample, combine=True):
    binary_sample = np.squeeze(binary_sample)
    print(f'shape: {binary_sample.shape}')
    if combine:
        data = np.rollaxis(binary_sample, 2, 0)[0]\
               + 2 * np.rollaxis(binary_sample, 2, 0)[1]
        if binary_sample.shape[-1] == 3:
            data += 3 * np.rollaxis(binary_sample, 2, 0)[2]
        print(data)

    else:  # 3rd channel is the boarder that nobody would want to see
        print(np.rollaxis(binary_sample, 2, 0)[0])
        print()
        print(np.rollaxis(binary_sample, 2, 0)[1])


def vis(tensor, scale=1):
    npa = np.squeeze(tensor[0, :, :, 0].numpy())
    print((npa * scale).astype(int))


def string_to_stones(encoded):
    """
    returns an array of pairs for a string-encoded sequence
    e.g. [('A',1), ('M',14)] for 'a1m14'
    """
    x, y = encoded[0].upper(), 0
    stones = []
    for c in encoded[1:]:
        if c.isdigit():
            y = 10 * y + int(c)
        else:
            stones.append((x, y))
            x = c.upper()
            y = 0
    stones.append((x, y))
    return stones


def stones_to_string(stones):
    """
    returns a string-encoded sequence for an array of pairs.
    e.g. 'a1m14' for [('A',1), ('M',14)]
    """
    return "".join([(s[0].lower() if type(s[0]) == str else chr(96 + s[0])) + str(s[1]) for s in stones])


def str_base(number, base, width=8):
    def _str_base(n, b):
        (d, m) = divmod(n, b)
        if d > 0:
            return _str_base(d, b) + str(m)
        return str(m)


    s = _str_base(number, base)
    return '0' * (width - len(s)) + s


def base2_to_xo(number):
    return str_base(number, 3).replace('2', 'o').replace('1', 'x').replace('0', '.')


def mask(offensive, defensive):
    n = defensive
    ll = n & 0xF0
    ll = (ll | ll << 1 | ll << 2 | ll << 3) & 0xF0

    r = n & 0x0F
    r = (r | r >> 1 | r >> 2 | r >> 3) & 0x0F

    the_mask = (~(ll | r)) & 0xFF
    free_stones = the_mask & offensive

    return free_stones, the_mask


def num_offensive(o, d):
    s, l, offset = mask2(o, d)
    m2o_bits = as_bit_array(s)[:l]
    max_count = 0
    for w in [2, 1, 0]:
        i = 0
        while i <= len(m2o_bits) - 2 - w:
            count = sum(m2o_bits[i:i + w + 2])
            count = 3 * count - (w + 2)
            if count > max_count:
                max_count = count
            i += 1
    if m2o_bits[0] == 0:
        max_count += 1
    if m2o_bits[-1] == 0:
        max_count += 1

    # Criticality correction for the fatal double-open 3
    if max_count == 8:
        max_count = 13
    return max_count


def mask2(offensive, defensive):
    n = defensive
    ll = n & 0xF0
    ll = (ll | ll << 1 | ll << 2 | ll << 3) & 0xF0

    r = n & 0x0F
    r = (r | r >> 1 | r >> 2 | r >> 3) & 0x0F

    m = (~(ll | r))
    free_stones = m & offensive

    free_length = np.sum([(m >> i) & 1 for i in range(8)], axis=0)
    l_offset = np.sum([(ll >> i) & 1 for i in range(8)], axis=0)
    # free_length = (free_length > 5) * 5 + (free_length <= 5) * free_length
    return free_stones << l_offset, free_length, l_offset


def m2b(m, size):
    """matrix index to board position"""
    r, c = m
    return np.array([c + 1, size - r])


def m2b2(m, size):
    """matrix index to board position"""
    r, c = m
    return chr(c + 65), size - r


def as_bit_array(n):
    """
    Returns an array of int 0 or 1
    """
    assert (0 <= n <= 255)
    return [np.sign(n & (1 << i)) for i in range(7, -1, -1)]


def line_for_xo(xo_string):
    """
    return a 2x8 int array representing the 'x..o..' xo_string
    """
    powers = np.array([2 ** i for i in range(7, -1, -1)])
    return [sum([1 if (ch == 'x' and c == 0)
                or (ch == 'o' and c == 1)
                 else 0 for ch in xo_string] * powers) for c in [0, 1]]
