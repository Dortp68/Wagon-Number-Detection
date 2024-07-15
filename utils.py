

def save_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('numplate_id', 'x1', 'y1', 'x2', 'y2', 'score', 'text', 'text_score'))

        for l in results:
            f.write('{},{},{},{},{},{},{},{}\n'.format(l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7]))

        f.close()

def preprocess(data):
    track, ocr = data
    ocr_result = ocr[0][-1]
    x1, y1, x2, y2, score = list(ocr[0][:-1])
    if ocr_result != []:
        id_ = track[0][4]
        return[id_, x1, y1, x2, y2, score, ocr_result[0][0], ocr_result[0][1]]
