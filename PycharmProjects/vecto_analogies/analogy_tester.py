import pandas
import vecto2.vecto
import vecto2.vecto.embeddings
from vecto2.vecto.utils.data import save_json
from vecto2.vecto.benchmarks.analogy import LRCos


class embeddings:
    def __init__(self, path):
        i = path
        self.embedding_name = i.split('/')[5]
        self.embedding_directory = i
        self.embeddings = vecto2.vecto.embeddings.load_from_dir(self.embedding_directory)

    def get_analogy(self, dataset):
        self.dataset_path = dataset
        analogy = LRCos()
        self.result = analogy.get_result(self.embeddings, dataset)
        save_json(self.result, "./res.json")

    def get_row(self):
        self.row = {}
        for subcategory in self.subcategories:
            self.row[subcategory] = 1

    def get_dictionary(self):
        dictionary = {}
        dictionary['embedding'] = self.embedding_name
        for i in self.result:
            subcategory = i["experiment_setup"]["subcategory"]
            results = i['result']
            missing_pairs = self.get_length(subcategory) - i["experiment_setup"]["cnt_questions_total"]
            dictionary[subcategory+' results'] = results
            dictionary[subcategory + ' missing pairs'] = missing_pairs
        dictionary['citation'] = str(self.get_citation())
        dictionary['description'] = str(self.get_description())
        return dictionary


    def get_citation(self):
        citation = self.result[0]['experiment_setup']['embeddings']['vocabulary']['corpus']['bib']
        return citation


    def get_description(self):
        description = ''
        if 'description' in self.result[0]['experiment_setup']['embeddings']:
            description += self.result[0]['experiment_setup']['embeddings']['description']+ ", "
            #description += "corpus name: "+self.result[0]['experiment_setup']['embeddings']['vocabulary']['corpus']['name']
            #description += ", language: "+self.result[0]['experiment_setup']['embeddings']['vocabulary']['corpus']['language']
            #description += ", size: "+str(self.result[0]['experiment_setup']['embeddings']['vocabulary']['corpus']['size'])
            description += ", size: " + str(self.result[0]['experiment_setup']['embeddings']['vocabulary']['cnt_words'])
        return description


    def get_length(self, filename):
        i = 0
        file = open(self.dataset_path + filename, 'r')
        for line in file:
            i+=1
        return i


def main(csv_name, embedding_index):
    directories = ["/home/downey/PycharmProjects/vecto_analogies/embeddings/structured_deps_cbow_500d",
                   "/home/downey/PycharmProjects/vecto_analogies/embeddings/structured_linear_cbow_500d",
                   "/home/downey/PycharmProjects/vecto_analogies/embeddings/word_deps_cbow_500d",
                   "/home/downey/PycharmProjects/vecto_analogies/embeddings/word_linear_cbow_500d",
                   "/home/downey/PycharmProjects/vecto_analogies/embeddings/!Demo2/embeddings/bnc/",
                   "/home/downey/PycharmProjects/vecto_analogies/embeddings/lstm/1/word/",
                   "/home/downey/PycharmProjects/vecto_analogies/embeddings/glove.6b/glove.6B/"]

    embedding = embeddings(directories[embedding_index])
    results = embedding.get_analogy("/home/downey/PycharmProjects/vecto_analogies/BATS/BATS_collective/")
    dict = embedding.get_dictionary()
    print(dict)
    df = pandas.DataFrame([dict]).set_index('embedding')
    df.to_csv('/home/downey/PycharmProjects/vecto_analogies/tables/' + csv_name)

main('NN.csv', 4)