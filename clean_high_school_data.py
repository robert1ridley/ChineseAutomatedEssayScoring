def get_ratings_maps(in_rating):
    rating_map_dict = {
        'rating2': 2,
        'rating3': 3,
        'rating4': 4,
        'rating5': 5,
    }
    return rating_map_dict[in_rating]


def get_data(read_location, write_location):
    write_data_file = open(write_location, 'a')
    data_file = open(read_location, 'r')
    data = data_file.readlines()
    for line in data:
        essay_data = {}
        items = line.split('\t')
        if not items[5].startswith('rating'):
            for i, item in enumerate(items):
                if item.startswith('rating'):
                    items[4:i] = [''.join(items[4:i])]
                    break
        essay_data['id'] = items[1]
        essay_data['url'] = items[2]
        essay_data['title'] = items[3]
        essay_data['essay'] = items[4]
        essay_data['rating'] = get_ratings_maps(items[5])
        topics = []

        for item in items[6:]:
            item = item.strip()
            if '字' in item:
                essay_data['num_characters'] = item
            elif item in ['高一', '高二', '高三', '高考']:
                essay_data['age_group'] = item
            else:
                topics.append(item)
        essay_data['topics'] = '\t'.join(topics)
        new_line = essay_data['id'] + '\t' + essay_data['url'] + '\t' + essay_data['title'] + '\t' + \
                   essay_data['essay'] + '\t' + str(essay_data['rating']) + '\t' + essay_data['age_group'] + '\t' + \
                   essay_data['num_characters'] + '\t' + essay_data['topics'] + '\n'
        write_data_file.write(new_line)
    write_data_file.close()


if __name__ == '__main__':
    get_data('leleketang_high_school_raw.tsv', 'leleketang_high_school_clean.tsv')
