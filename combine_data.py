def main(in_files, out_filepath):
    all_data = []
    for single_file in in_files:
        data = open(single_file, 'r')
        lines = data.readlines()
        all_data.extend(lines)
        data.close()
    out_file = open(out_filepath, 'a')
    for item in all_data:
        out_file.write(item)
    out_file.close()


if __name__ == '__main__':
    main(['leleketang_middle_school_clean.tsv', 'leleketang_high_school_clean.tsv'], 'leleketang_combined_clean.tsv')
