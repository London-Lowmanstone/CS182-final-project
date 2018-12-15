import csv
with open("data/dem_tweets.txt", "w+") as dem_save_file:
    with open("data/rep_tweets.txt", "w+") as rep_save_file:
        with open('data/tweets.csv', newline='') as csv_file:
            spamreader = csv.reader(csv_file, delimiter=',')
            for row_index, row in enumerate(spamreader):
                print(row_index)
                if len(row) > 2:
                    file_to_write_to = None
                    if row[1] == "HillaryClinton":
                        file_to_write_to = dem_save_file
                    elif row[1] == "realDonaldTrump":
                        file_to_write_to = rep_save_file
                    else:
                        print("Unrecognized handle: {}".format(row[1]))
                    if file_to_write_to:
                        file_to_write_to.write(row[2])
                        file_to_write_to.write("\n\n\n")
# 0 gives ID
# 1 gives name
# 2 gives tweet
