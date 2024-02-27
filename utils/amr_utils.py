def get_amr_pairs(triple_path):  # split should be one of ["train", "dev", "test"]

    def read_amrs(path):
        f = open(path, 'r')
        sentences = f.readlines()
        f.close()
        return sentences

    sent_file, lin_graph_file, amr_graph_file = triple_path

    file_streams = [read_amrs(sent_file), read_amrs(lin_graph_file)]

    mapped = map(lambda zipped: (zipped[0].rstrip(), zipped[1].rstrip()),
                    zip(*file_streams))

    return list(mapped)

if __name__ == '__main__':
    from settings import TASK2PATH
    task2path = TASK2PATH()
    triple_path = task2path.get_path('en-amr', "dev")
    li = get_amr_pairs(triple_path)
    print(li[0])

