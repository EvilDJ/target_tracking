cdd = open('./ground.csv', 'w')
with open('./groundtruth_rect.txt', 'r') as f:
    num = 0
    for line in f:
        if '\t' in line:
            line = line.replace('\t', ',')
        cdd.write(str(num) + ',' + line)
        print(num)
        num += 1
        # if num >= 100:
        #     break
cdd.closed
