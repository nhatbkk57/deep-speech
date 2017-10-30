# -*- coding: utf-8 -*-
def load_vocab():
    char_map_str = """
    <SPACE> 1
    a 2
    à 3
    á 4
    ả 5
    ạ 6
    ã 7
    ă 8
    ằ 9
    ắ 10
    ẳ 11
    ặ 12
    ẵ 13
    â 14
    ầ 15
    ấ 16
    ẩ 17
    ậ 18
    ẫ 19
    b 20
    c 21
    d 22
    đ 23
    e 24
    è 25
    é 26
    ẻ 27
    ẹ 28
    ẽ 29
    ê 30
    ề 31
    ế 32
    ể 33
    ệ 34
    ễ 35
    g 36
    h 37
    i 38
    ì 39
    í 40
    ỉ 41
    ị 42
    ĩ 43
    k 44
    l 45
    m 46
    n 47
    o 48
    ò 49
    ó 50
    ỏ 51
    ọ 52
    õ 53
    ô 54
    ồ 55
    ố 56
    ổ 57
    ộ 58
    ỗ 59
    ơ 60
    ờ 61
    ớ 62
    ở 63
    ợ 64
    ỡ 65
    p 66
    q 67
    r 68
    s 69
    t 70
    u 71
    ù 72
    ú 73
    ủ 74
    ụ 75
    ũ 76
    ư 77
    ừ 78
    ứ 79
    ử 80
    ự 81
    ữ 82
    v 83
    x 84
    y 85
    ỳ 86
    ý 87
    ỷ 88
    ỵ 89
    ỹ 90
    E 91
    G 92
    """
    char_map = {}
    index_map = {}
    for line in char_map_str.strip().split('\n'):
        ch, index = line.split()

        char_map[ch] = int(index)
        index_map[int(index)] = ch
        index_map[1] = ' '
    return index_map,char_map

index_map,char_map = load_vocab()