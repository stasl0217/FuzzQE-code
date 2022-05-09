
query_name_dict = {
    ('e',('r',)): '1p',
    ('e', ('r', 'r')): '2p',
    ('e', ('r', 'r', 'r')): '3p',
    (('e', ('r',)), ('e', ('r',))): '2i',
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
}
query_structure_list = list(query_name_dict.keys())  # query_structure_list[0] -> query_structure of index 0
query_structure2idx = {s: i for i, s in enumerate(query_structure_list)}  # {('e',('r',)):0}
