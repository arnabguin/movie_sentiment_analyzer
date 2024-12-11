#!python
import twitter_search

with open('movies.list', 'r') as reader:
    f = open("labeled_data","w")
    f.write("review" + "\t" + "sentiment" + "\n")
    for line in reader.readlines():
        results = twitter_search.TwitterSearchEngine(line, 100, "security.json").get_results()
        for key, value in results.items():
            length = len(value)
            for i in range(length):
                s = ""
                for key2, value2 in value[i].items():
                    if (key2 == "subjectivity"):
                        continue
                    sentiment = 0
                    if (key2 == "polarity"):
                        if (value2 > 0.25):
                            sentiment = 1
                        s = s + str(sentiment)
                    else:
                        s = s + value2 + "\t"
                #print (s)
                s = s + "\n"
                f.write(s)
    f.close()
