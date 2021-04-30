f = open("res/data/base_data/label.csv", "w")

for i in range(361):
    f.write(f'{i}.jpg,\n')
f.close()