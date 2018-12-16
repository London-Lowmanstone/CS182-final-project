import csv
accounts_id_index = 0
tags_index = 9
tweets_id_index = 1
tweet_index = 3
democrat = "democrat"
republican = "republican"
max_count = 500

def lower_list(l):
    return list(map(lambda s: s.lower(), l))

def string_dict_to_list(s):
    return split_by_comma(take_off_outer_braces(s))
    
def string_dict_to_lower_list(s):
    return lower_list(string_dict_to_list(s))

def take_off_outer_braces(s):
    return s.strip()[1:-1]
    
def split_by_comma(s):
    return s.split(",")
    
party_counters = {democrat: 0, republican: 0}

with open("log.txt", "a+") as log:
    with open("data/test_dem_tweets.txt", "w+") as dem_save_file:
        with open("data/test_rep_tweets.txt", "w+") as rep_save_file:
            save_files = {"democrat": dem_save_file, "republican": rep_save_file}
            with open('bkey-politician-tweets/pol_accounts.csv', newline='', encoding="utf-8") as accounts_csv:
                accounts_reader = csv.reader(accounts_csv, delimiter=";")
                # {ID: party}
                # no ID = not a major party, so ignore
                ids_to_party = {}
                
                try:
                    for row_index, row in enumerate(accounts_reader):
                        parties = string_dict_to_lower_list(row[tags_index])
                        if democrat in parties and not republican in parties:
                            ids_to_party[row[accounts_id_index]] = democrat
                        elif republican in parties and not democrat in parties:
                            ids_to_party[row[accounts_id_index]] = republican
                except:
                    print(row_index)
                    raise
                    
            with open('bkey-politician-tweets/pol_tweets.csv', newline='', encoding="utf-8") as tweets_csv:
                tweet_reader = csv.reader(tweets_csv, delimiter=";")
                for row_index, row in enumerate(tweet_reader):
                    try:
                        party = ids_to_party[row[tweets_id_index]]
                    except KeyError:
                        # There will be many of these because there are many people who are not republican or democrat
                        # print("Failed to find ID {}".format(row[tweets_id_index]))
                        party = None
                    
                    try:
                        if party_counters[party] < max_count:
                            party_counters[party] += 1
                            file_to_write_to = save_files[party]
                        else:
                            raise RuntimeError("Party {} is done".format(party))
                    except (RuntimeError, KeyError):
                        file_to_write_to = None
                        
                    if file_to_write_to:
                        file_to_write_to.write(row[tweet_index])
                        file_to_write_to.write("\n\n\n")
                    
                    if False not in [counter == max_count for party, counter in party_counters.items()]:
                        break

print(party_counters)
